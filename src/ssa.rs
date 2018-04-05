use ::prelude::{BTreeSet, VecDeque};
use module::{Module, Type};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FlowGraph {
    pub blocks: Vec<BasicBlock>
}

#[derive(Default, Serialize, Deserialize, Debug, Clone)]
pub struct BasicBlock {
    pub ops: Vec<(Option<ValueId>, Opcode)>,
    pub br: Option<Branch>
}

#[derive(Serialize, Deserialize, Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub struct ValueId(pub usize);

#[derive(Serialize, Deserialize, Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub struct BlockId(pub usize);

#[derive(Serialize, Deserialize, Debug, Clone, Eq, PartialEq)]
pub enum Branch {
    Br(BlockId),
    BrEither(ValueId, BlockId, BlockId),
    BrTable(ValueId, Vec<BlockId>, BlockId),
    Return(Option<ValueId>)
}

#[derive(Serialize, Deserialize, Debug, Clone, Eq, PartialEq)]
pub enum Opcode {
    Phi(Vec<ValueId>),

    Select(ValueId, ValueId, ValueId), // (cond, if_true, if_false)

    GetLocal(u32),
    SetLocal(u32, ValueId),
    GetGlobal(u32),
    SetGlobal(u32, ValueId),

    CurrentMemory,
    GrowMemory(ValueId),

    Unreachable,

    Call(u32, Vec<ValueId>),
    CallIndirect(u32, ValueId, Vec<ValueId>),

    I32Const(i32),
}

#[derive(Default, Clone, Debug)]
struct BlockInfo {
    pre: BTreeSet<BlockId>,
    outgoing_values: Vec<ValueId>,
    scan_completed: bool
}

struct DedupBfs<T: Ord + PartialOrd + Copy> {
    scanned: BTreeSet<T>,
    queue: VecDeque<T>
}

impl<T: Ord + PartialOrd + Copy> DedupBfs<T> {
    fn new() -> DedupBfs<T> {
        DedupBfs {
            scanned: BTreeSet::new(),
            queue: VecDeque::new()
        }
    }

    fn next(&mut self) -> Option<T> {
        self.queue.pop_back()
    }

    fn push(&mut self, val: T) {
        if self.scanned.contains(&val) {
            return;
        }

        self.queue.push_front(val);
        self.scanned.insert(val);
    }

    fn is_scanned(&self, val: &T) -> bool {
        self.scanned.contains(val)
    }
}

impl FlowGraph {
    pub fn from_cfg(cfg: &::cfgraph::CFGraph, m: &Module) -> FlowGraph {
        let mut output: Vec<BasicBlock> = vec! [ BasicBlock::default(); cfg.blocks.len() ];
        let mut block_info: Vec<BlockInfo> = vec! [ BlockInfo::default(); cfg.blocks.len() ];
        let mut bfs: DedupBfs<BlockId> = DedupBfs::new();

        let mut value_id_feed = 0..0xffffffff as usize;

        collect_graph_info(cfg, &mut block_info);

        bfs.push(BlockId(0));

        while let Some(BlockId(id)) = bfs.next() {
            let blk = &cfg.blocks[id];
            let blk_info = &block_info[id];
            let out = &mut output[id];

            use opcode::Opcode as RawOp;

            let mut outgoing_values: Vec<ValueId> = Vec::new();

            // Find out the current stack state.
            if blk_info.pre.len() > 0 {
                for i in 0..0xffffffffusize {
                    let mut should_break: bool = false;
                    let mut last_val_id: Option<ValueId> = None;

                    for pre in &blk_info.pre {
                        let pre_info = &block_info[pre.0];
                        if i >= pre_info.outgoing_values.len() {
                            should_break = true;
                            break;
                        }
                        if let Some(last_val_id) = last_val_id {
                            if last_val_id != pre_info.outgoing_values[i] {
                                should_break = true;
                                break;
                            }
                        }
                        last_val_id = Some(pre_info.outgoing_values[i]);
                    }
                    if should_break {
                        break;
                    }

                    outgoing_values.push(last_val_id.unwrap());
                }

                let mut phi_incoming: Vec<ValueId> = Vec::new();

                let mut is_cycle_begin: bool = false;

                for pre in &blk_info.pre {
                    let pre_info = &block_info[pre.0];

                    // Cycles?
                    if !pre_info.scan_completed {
                        is_cycle_begin = true;
                        break;
                    }

                    if pre_info.outgoing_values.len() == outgoing_values.len() {
                    } else if pre_info.outgoing_values.len() == outgoing_values.len() + 1 {
                        phi_incoming.push(*pre_info.outgoing_values.last().unwrap());
                    } else {
                        panic!("Invalid stack state");
                    }
                }

                if is_cycle_begin {
                    assert_eq!(outgoing_values.len(), 0);
                } else {
                    if phi_incoming.len() == 0 {
                    } else if phi_incoming.len() == blk_info.pre.len() {
                        let new_value = ValueId(value_id_feed.next().unwrap());

                        out.ops.push((Some(new_value), Opcode::Phi(phi_incoming)));
                        outgoing_values.push(new_value);
                    } else {
                        panic!("phi_incoming length mismatch");
                    }
                }
            }

            block_info[id].outgoing_values = outgoing_values;

            for op in &blk.opcodes {
                let outgoing = &mut block_info[id].outgoing_values;
                let mut terminate: bool = false;

                match *op {
                    RawOp::Drop => {
                        outgoing.pop().unwrap();
                    },
                    RawOp::Select => {
                        let c = outgoing.pop().unwrap();
                        let val2 = outgoing.pop().unwrap();
                        let val1 = outgoing.pop().unwrap();

                        let val_id = ValueId(value_id_feed.next().unwrap());
                        outgoing.push(val_id);

                        out.ops.push((Some(val_id), Opcode::Select(c, val1, val2)));
                    },

                    RawOp::GetLocal(id) => {
                        let val_id = ValueId(value_id_feed.next().unwrap());
                        outgoing.push(val_id);

                        out.ops.push((Some(val_id), Opcode::GetLocal(id)));
                    },
                    RawOp::SetLocal(id) => {
                        let val = outgoing.pop().unwrap();
                        out.ops.push((None, Opcode::SetLocal(id, val)));
                    },
                    RawOp::TeeLocal(id) => {
                        let val = *outgoing.last().unwrap();
                        out.ops.push((None, Opcode::SetLocal(id, val)));
                    },
                    RawOp::GetGlobal(id) => {
                        let val_id = ValueId(value_id_feed.next().unwrap());
                        outgoing.push(val_id);

                        out.ops.push((Some(val_id), Opcode::GetGlobal(id)));
                    },
                    RawOp::SetGlobal(id) => {
                        let val = outgoing.pop().unwrap();
                        out.ops.push((None, Opcode::SetGlobal(id, val)));
                    },
                    RawOp::CurrentMemory => {
                        let val_id = ValueId(value_id_feed.next().unwrap());
                        outgoing.push(val_id);

                        out.ops.push((Some(val_id), Opcode::CurrentMemory));
                    },
                    RawOp::GrowMemory => {
                        let val = outgoing.pop().unwrap();

                        let val_id = ValueId(value_id_feed.next().unwrap());
                        outgoing.push(val_id);

                        out.ops.push((Some(val_id), Opcode::GrowMemory(val)));
                    },
                    RawOp::Nop => {},
                    RawOp::Unreachable => {
                        out.ops.push((None, Opcode::Unreachable));
                        terminate = true;
                    },
                    RawOp::Call(funcidx) => {
                        let f = &m.functions[funcidx as usize];
                        let Type::Func(ref ty_args, ref ty_ret) = &m.types[f.typeidx as usize];

                        let mut args: Vec<ValueId> = Vec::with_capacity(ty_args.len());
                        for _ in 0..ty_args.len() {
                            args.push(outgoing.pop().unwrap());
                        }
                        args.reverse();

                        out.ops.push((
                            match ty_ret.len() {
                                0 => None,
                                _ => {
                                    let val_id = ValueId(value_id_feed.next().unwrap());
                                    outgoing.push(val_id);
                                    Some(val_id)
                                }
                            },
                            Opcode::Call(funcidx, args)
                        ));
                    },
                    RawOp::CallIndirect(typeidx) => {
                        let Type::Func(ref ty_args, ref ty_ret) = &m.types[typeidx as usize];

                        let fn_index = outgoing.pop().unwrap();

                        let mut args: Vec<ValueId> = Vec::with_capacity(ty_args.len());
                        for _ in 0..ty_args.len() {
                            args.push(outgoing.pop().unwrap());
                        }
                        args.reverse();

                        out.ops.push((
                            match ty_ret.len() {
                                0 => None,
                                _ => {
                                    let val_id = ValueId(value_id_feed.next().unwrap());
                                    outgoing.push(val_id);
                                    Some(val_id)
                                }
                            },
                            Opcode::CallIndirect(typeidx, fn_index, args)
                        ));
                    },
                    RawOp::I32Const(v) => {
                        let val_id = ValueId(value_id_feed.next().unwrap());
                        outgoing.push(val_id);

                        out.ops.push((Some(val_id), Opcode::I32Const(v)));
                    },
                    _ => unimplemented!()
                }

                if terminate {
                    break;
                }
            }

            out.br = Some(match *blk.br.as_ref().unwrap() {
                ::cfgraph::Branch::Jmp(::cfgraph::BlockId(id)) => {
                    bfs.push(BlockId(id));
                    Branch::Br(BlockId(id))
                },
                ::cfgraph::Branch::JmpEither(
                    ::cfgraph::BlockId(a),
                    ::cfgraph::BlockId(b)
                ) => {
                    bfs.push(BlockId(a));
                    bfs.push(BlockId(b));

                    Branch::BrEither(
                        block_info[id].outgoing_values.pop().unwrap(),
                        BlockId(a),
                        BlockId(b)
                    )
                },
                ::cfgraph::Branch::JmpTable(
                    ref targets,
                    otherwise
                ) => {
                    let mut out_targets: Vec<BlockId> = Vec::with_capacity(targets.len());

                    for t in targets {
                        bfs.push(BlockId(t.0));
                        out_targets.push(BlockId(t.0));
                    }

                    bfs.push(BlockId(otherwise.0));

                    Branch::BrTable(
                        block_info[id].outgoing_values.pop().unwrap(),
                        out_targets,
                        BlockId(otherwise.0)
                    )
                },
                ::cfgraph::Branch::Return => {
                    Branch::Return(block_info[id].outgoing_values.pop())
                }
            });

            block_info[id].scan_completed = true;
        }

        FlowGraph {
            blocks: output
        }
    }
}

fn collect_graph_info(cfg: &::cfgraph::CFGraph, out: &mut [BlockInfo]) {
    for (i, blk) in cfg.blocks.iter().enumerate() {
        use cfgraph::Branch;
        let mut br_unreachable: bool = false;
        for op in &blk.opcodes {
            match *op {
                ::opcode::Opcode::Unreachable => {
                    br_unreachable = true;
                    break;
                },
                _ => {}
            }
        }

        // The branch will never be executed if the block includes an `unreachable` opcode.
        if br_unreachable {
            continue;
        }

        match *blk.br.as_ref().unwrap() {
            Branch::Jmp(::cfgraph::BlockId(id)) => {
                out[id].pre.insert(BlockId(i));
            },
            Branch::JmpEither(::cfgraph::BlockId(a), ::cfgraph::BlockId(b)) => {
                out[a].pre.insert(BlockId(i));
                out[b].pre.insert(BlockId(i));
            },
            Branch::JmpTable(ref targets, ::cfgraph::BlockId(otherwise)) => {
                for t in targets {
                    out[t.0].pre.insert(BlockId(i));
                }
                out[otherwise].pre.insert(BlockId(i));
            }
            Branch::Return => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_ssa_transform() {
        use ::opcode::Opcode as RawOp;
        let opcodes = &[
            RawOp::I32Const(42),
            RawOp::JmpIf(4),
            RawOp::I32Const(1),
            RawOp::Jmp(5),
            RawOp::I32Const(2),
            RawOp::Return
        ];

        let cfg = ::cfgraph::CFGraph::from_function(opcodes).unwrap();
        let ssa = FlowGraph::from_cfg(&cfg, &Module::default());
        println!("{:?}", ssa);
        assert_eq!(ssa.blocks.len(), 4);
        assert_eq!(ssa.blocks[3].ops[0], (
            Some(ValueId(3)),
            Opcode::Phi(vec! [ ValueId(2), ValueId(1) ])
        ));
    }

    #[test]
    fn test_circular_transform() {
        use ::opcode::Opcode as RawOp;
        let opcodes = &[
            RawOp::I32Const(42),
            RawOp::JmpIf(3),
            RawOp::Jmp(0),
            //RawOp::I32Const(2),
            RawOp::Jmp(0),
            RawOp::Return
        ];

        let cfg = ::cfgraph::CFGraph::from_function(opcodes).unwrap();
        let ssa = FlowGraph::from_cfg(&cfg, &Module::default());
        println!("{:?}", ssa);
        assert_eq!(ssa.blocks.len(), 4);
        assert_eq!(ssa.blocks[0].br, Some(Branch::BrEither(
            ValueId(0),
            BlockId(2),
            BlockId(1)
        )));
        assert_eq!(ssa.blocks[1].br, Some(Branch::Br(BlockId(0))));
        assert_eq!(ssa.blocks[2].br, Some(Branch::Br(BlockId(0))));
    }

    #[test]
    fn test_unreachable() {
        use ::opcode::Opcode as RawOp;
        let opcodes = &[
            RawOp::I32Const(42),
            RawOp::JmpIf(12),
            RawOp::Unreachable,
            RawOp::Drop,
            RawOp::Drop,
            RawOp::Drop,
            RawOp::I32Const(100),
            RawOp::I32Const(100),
            RawOp::I32Const(100),
            RawOp::I32Const(100),
            RawOp::I32Const(100),
            RawOp::I32Const(100),
            RawOp::Return
        ];

        let cfg = ::cfgraph::CFGraph::from_function(opcodes).unwrap();
        let ssa = FlowGraph::from_cfg(&cfg, &Module::default());
        //println!("{:?}", ssa);
    }

    #[test]
    fn test_call() {
        use ::module::{Function, FunctionBody, ValType};
        use ::opcode::Opcode as RawOp;

        let opcodes = &[
            RawOp::I32Const(42),
            RawOp::Call(0),
            RawOp::Return
        ];

        let cfg = ::cfgraph::CFGraph::from_function(opcodes).unwrap();

        let mut m = Module::default();
        m.types.push(Type::Func(vec! [ ValType::I32 ], vec! [ ValType::I32 ]));
        m.functions.push(Function {
            name: None,
            typeidx: 0,
            locals: vec! [],
            body: FunctionBody { opcodes: vec! [] }
        });
        let ssa = FlowGraph::from_cfg(&cfg, &m);
        println!("{:?}", ssa);

        assert_eq!(ssa.blocks.len(), 1);
        assert_eq!(
            ssa.blocks[0].ops[0],
            (Some(ValueId(0)), Opcode::I32Const(42))
        );
        assert_eq!(
            ssa.blocks[0].ops[1],
            (Some(ValueId(1)), Opcode::Call(0, vec! [ ValueId(0) ]))
        );
        assert_eq!(
            ssa.blocks[0].br,
            Some(Branch::Return(Some(ValueId(1))))
        );
    }
}
