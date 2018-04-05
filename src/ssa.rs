use ::prelude::{BTreeSet, VecDeque};

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
    I32Const(i32)
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
    pub fn from_cfg(cfg: &::cfgraph::CFGraph) -> FlowGraph {
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
                out.ops.push(match *op {
                    RawOp::I32Const(v) => {
                        let val_id = ValueId(value_id_feed.next().unwrap());
                        block_info[id].outgoing_values.push(val_id);
                        (Some(val_id), Opcode::I32Const(v))
                    },
                    _ => unimplemented!()
                });
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
        let ssa = FlowGraph::from_cfg(&cfg);
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
        let ssa = FlowGraph::from_cfg(&cfg);
        println!("{:?}", ssa);
        assert_eq!(ssa.blocks.len(), 4);
        assert_eq!(ssa.blocks[1].br, Some(Branch::BrEither(
            ValueId(0),
            BlockId(2),
            BlockId(1)
        )));
        assert_eq!(ssa.blocks[2].br, Some(Branch::Br(BlockId(0))));
        assert_eq!(ssa.blocks[3].br, Some(Branch::Br(BlockId(0))));
    }
}
