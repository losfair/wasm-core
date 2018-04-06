use ::prelude::{BTreeSet, VecDeque};
use module::{Module, Type};
use opcode::Memarg;
use std::ops::Range;

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

    I32Clz(ValueId),
    I32Ctz(ValueId),
    I32Popcnt(ValueId),

    I32Add(ValueId, ValueId),
    I32Sub(ValueId, ValueId),
    I32Mul(ValueId, ValueId),
    I32DivU(ValueId, ValueId),
    I32DivS(ValueId, ValueId),
    I32RemU(ValueId, ValueId),
    I32RemS(ValueId, ValueId),
    I32And(ValueId, ValueId),
    I32Or(ValueId, ValueId),
    I32Xor(ValueId, ValueId),
    I32Shl(ValueId, ValueId),
    I32ShrU(ValueId, ValueId),
    I32ShrS(ValueId, ValueId),
    I32Rotl(ValueId, ValueId),
    I32Rotr(ValueId, ValueId),

    I32Eqz(ValueId),

    I32Eq(ValueId, ValueId),
    I32Ne(ValueId, ValueId),
    I32LtU(ValueId, ValueId),
    I32LtS(ValueId, ValueId),
    I32LeU(ValueId, ValueId),
    I32LeS(ValueId, ValueId),
    I32GtU(ValueId, ValueId),
    I32GtS(ValueId, ValueId),
    I32GeU(ValueId, ValueId),
    I32GeS(ValueId, ValueId),

    I32WrapI64(ValueId),

    I32Load(Memarg, ValueId),
    I32Store(Memarg, ValueId /* index */, ValueId /* value */),
    I32Load8U(Memarg, ValueId),
    I32Load8S(Memarg, ValueId),
    I32Load16U(Memarg, ValueId),
    I32Load16S(Memarg, ValueId),
    I32Store8(Memarg, ValueId /* index */, ValueId /* value */),
    I32Store16(Memarg, ValueId /* index */, ValueId /* value */),

    I64Const(i64),

    I64Clz(ValueId),
    I64Ctz(ValueId),
    I64Popcnt(ValueId),

    I64Add(ValueId, ValueId),
    I64Sub(ValueId, ValueId),
    I64Mul(ValueId, ValueId),
    I64DivU(ValueId, ValueId),
    I64DivS(ValueId, ValueId),
    I64RemU(ValueId, ValueId),
    I64RemS(ValueId, ValueId),
    I64And(ValueId, ValueId),
    I64Or(ValueId, ValueId),
    I64Xor(ValueId, ValueId),
    I64Shl(ValueId, ValueId),
    I64ShrU(ValueId, ValueId),
    I64ShrS(ValueId, ValueId),
    I64Rotl(ValueId, ValueId),
    I64Rotr(ValueId, ValueId),

    I64Eqz(ValueId),

    I64Eq(ValueId, ValueId),
    I64Ne(ValueId, ValueId),
    I64LtU(ValueId, ValueId),
    I64LtS(ValueId, ValueId),
    I64LeU(ValueId, ValueId),
    I64LeS(ValueId, ValueId),
    I64GtU(ValueId, ValueId),
    I64GtS(ValueId, ValueId),
    I64GeU(ValueId, ValueId),
    I64GeS(ValueId, ValueId),

    I64ExtendI32U(ValueId),
    I64ExtendI32S(ValueId),

    I64Load(Memarg, ValueId),
    I64Store(Memarg, ValueId /* index */, ValueId /* value */),
    I64Load8U(Memarg, ValueId),
    I64Load8S(Memarg, ValueId),
    I64Load16U(Memarg, ValueId),
    I64Load16S(Memarg, ValueId),
    I64Load32U(Memarg, ValueId),
    I64Load32S(Memarg, ValueId),
    I64Store8(Memarg, ValueId /* index */, ValueId /* value */),
    I64Store16(Memarg, ValueId /* index */, ValueId /* value */),
    I64Store32(Memarg, ValueId /* index */, ValueId /* value */),

    F32Const(u32),
    F64Const(u64),
    F32ReinterpretI32(ValueId),
    F64ReinterpretI64(ValueId),
    I32ReinterpretF32(ValueId),
    I64ReinterpretF64(ValueId),
    I32TruncSF32(ValueId),
    I32TruncUF32(ValueId),
    I32TruncSF64(ValueId),
    I32TruncUF64(ValueId),
    I64TruncSF32(ValueId),
    I64TruncUF32(ValueId),
    I64TruncSF64(ValueId),
    I64TruncUF64(ValueId),
    F32ConvertSI32(ValueId),
    F32ConvertUI32(ValueId),
    F32ConvertSI64(ValueId),
    F32ConvertUI64(ValueId),
    F64ConvertSI32(ValueId),
    F64ConvertUI32(ValueId),
    F64ConvertSI64(ValueId),
    F64ConvertUI64(ValueId),
    F32DemoteF64(ValueId),
    F64PromoteF32(ValueId),
    F32Abs(ValueId),
    F32Neg(ValueId),
    F32Ceil(ValueId),
    F32Floor(ValueId),
    F32Trunc(ValueId),
    F32Nearest(ValueId),
    F32Sqrt(ValueId),

    F32Add(ValueId, ValueId),
    F32Sub(ValueId, ValueId),
    F32Mul(ValueId, ValueId),
    F32Div(ValueId, ValueId),
    F32Min(ValueId, ValueId),
    F32Max(ValueId, ValueId),
    F32Copysign(ValueId, ValueId),
    F32Eq(ValueId, ValueId),
    F32Ne(ValueId, ValueId),
    F32Lt(ValueId, ValueId),
    F32Gt(ValueId, ValueId),
    F32Le(ValueId, ValueId),
    F32Ge(ValueId, ValueId),

    F64Abs(ValueId),
    F64Neg(ValueId),
    F64Ceil(ValueId),
    F64Floor(ValueId),
    F64Trunc(ValueId),
    F64Nearest(ValueId),
    F64Sqrt(ValueId),

    F64Add(ValueId, ValueId),
    F64Sub(ValueId, ValueId),
    F64Mul(ValueId, ValueId),
    F64Div(ValueId, ValueId),
    F64Min(ValueId, ValueId),
    F64Max(ValueId, ValueId),
    F64Copysign(ValueId, ValueId),
    F64Eq(ValueId, ValueId),
    F64Ne(ValueId, ValueId),
    F64Lt(ValueId, ValueId),
    F64Gt(ValueId, ValueId),
    F64Le(ValueId, ValueId),
    F64Ge(ValueId, ValueId),

    NativeInvoke(u32, Vec<ValueId>),
    Memcpy(ValueId, ValueId, ValueId) // (dest, src, n_bytes)
}

#[derive(Default, Clone, Debug)]
struct BlockInfo {
    pre: BTreeSet<BlockId>,
    succ: BTreeSet<BlockId>,

    backedges: BTreeSet<BlockId>,

    outgoing_values: Vec<ValueId>,

    // Indicates whether this block begins a cycle.
    cycle: bool,

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

macro_rules! impl_generic_binop {
    ($name:ident, $outgoing:expr, $value_id_feed:expr, $out:expr) => {
        {
            let b = $outgoing.pop().unwrap();
            let a = $outgoing.pop().unwrap();
            let val_id = ValueId($value_id_feed.next().unwrap());
            $outgoing.push(val_id);
            $out.ops.push((Some(val_id), Opcode::$name(a, b)));
        }
    }
}

macro_rules! impl_generic_unop {
    ($name:ident, $outgoing:expr, $value_id_feed:expr, $out:expr) => {
        {
            let v = $outgoing.pop().unwrap();
            let val_id = ValueId($value_id_feed.next().unwrap());
            $outgoing.push(val_id);
            $out.ops.push((Some(val_id), Opcode::$name(v)));
        }
    }
}

macro_rules! impl_mem_load {
    ($name:ident, $mem_arg:expr, $outgoing:expr, $value_id_feed:expr, $out:expr) => {
        {
            let index = $outgoing.pop().unwrap();
            let val_id = ValueId($value_id_feed.next().unwrap());
            $outgoing.push(val_id);
            $out.ops.push((Some(val_id), Opcode::$name($mem_arg, index)));
        }
    }
}

macro_rules! impl_mem_store {
    ($name:ident, $mem_arg:expr, $outgoing:expr, $value_id_feed:expr, $out:expr) => {
        {
            let val = $outgoing.pop().unwrap();
            let index = $outgoing.pop().unwrap();
            $out.ops.push((None, Opcode::$name($mem_arg, index, val)));
        }
    }
}

struct Analyzer<'a> {
    cfg: &'a ::cfgraph::CFGraph,
    m: &'a Module,
    value_id_feed: Range<usize>,
    block_info: Vec<BlockInfo>,
    basic_blocks: Vec<BasicBlock>
}

impl<'a> Analyzer<'a> {
    fn new(cfg: &'a ::cfgraph::CFGraph, m: &'a Module) -> Analyzer<'a> {
        let mut analyzer = Analyzer {
            cfg: cfg,
            m: m,
            value_id_feed: (0..0xffffffffusize),
            block_info: vec! [ BlockInfo::default(); cfg.blocks.len() ],
            basic_blocks: vec! [ BasicBlock::default(); cfg.blocks.len() ]
        };

        collect_graph_info(cfg, &mut analyzer.block_info);

        analyzer
    }

    fn analyze_cycles(&mut self) {
        if self.block_info.len() == 0 {
            return;
        }

        let mut dfs_stack: Vec<BlockId> = Vec::new();
        let mut dfs_reached: Vec<bool> = vec! [ false; self.block_info.len() ];

        dfs_stack.push(BlockId(0));
        dfs_reached[0] = true;

        while let Some(id) = dfs_stack.pop() {
            // FIXME: O(n^2).
            let mut inner_bfs: DedupBfs<BlockId> = DedupBfs::new();
            inner_bfs.push(id);

            let mut is_cycle = false;

            while let Some(that) = inner_bfs.next() {
                if self.block_info[that.0].cycle {
                    continue;
                }
                for t in &self.block_info[that.0].succ {
                    if *t == id {
                        is_cycle = true;
                        break;
                    }
                    inner_bfs.push(*t);
                }

                if is_cycle {
                    self.block_info[that.0].backedges.insert(id);
                    break;
                }
            }

            if is_cycle {
                self.block_info[id.0].cycle = true;
            }

            for t in &self.block_info[id.0].succ {
                if !dfs_reached[t.0] {
                    dfs_stack.push(*t);
                    dfs_reached[t.0] = true;
                }
            }
        }
    }

    fn longest_common_stack_sequence(incoming: &[&[ValueId]]) -> (Vec<ValueId>, Option<Vec<ValueId>>) {
        if incoming.len() == 0 {
            return (vec! [], None);
        }

        let common_len = incoming[0].len();
        for s in incoming {
            if s.len() != common_len {
                panic!("Invalid stack state: All incoming states must have the same depth");
            }
        }

        if common_len == 0 {
            return (vec! [], None);
        }

        let mut seq: Vec<ValueId> = Vec::new();

        for i in 0..common_len {
            let mut valid = true;
            for (a, b) in incoming.iter().zip(incoming.iter().skip(1)) {
                if a[i] != b[i] {
                    valid = false;
                    break;
                }
            }
            if valid {
                seq.push(incoming[0][i]);
            } else {
                break;
            }
        }

        if seq.len() == common_len {
            (seq, None)
        } else if seq.len() == common_len - 1 {
            let last_elements: Vec<ValueId> = incoming.iter().map(|v| *v.last().unwrap()).collect();
            (seq, Some(last_elements))
        } else {
            panic!("Invalid stack state: Got more than one different elements");
        }
    }

    fn generate_all(&mut self) {
        self.analyze_cycles();

        let n_blocks = self.basic_blocks.len();
        for i in 0..n_blocks {
            self.analyze_and_generate(BlockId(i));
        }
    }

    fn analyze_and_generate(&mut self, blk_id: BlockId) {
        if self.block_info[blk_id.0].scan_completed {
            return;
        }

        self.block_info[blk_id.0].scan_completed = true;

        // Require all previous blocks to be analyzed and code-generated first.
        // Just skip it if we've detected a cycle (?)
        // (Loops are guaranteed to preserve the stack state)
        let pres: Vec<BlockId> = self.block_info[blk_id.0].pre
            .iter()
            .filter(|v| !self.block_info[v.0].backedges.contains(&blk_id))
            .map(|v| *v)
            .collect();

        let pre_stack_seqs: Vec<&[ValueId]> = pres
            .iter()
            .map(|b| self.block_info[b.0].outgoing_values.as_slice())
            .collect();

        let (lcss, incoming) = Self::longest_common_stack_sequence(&pre_stack_seqs);

        self.block_info[blk_id.0].outgoing_values = lcss;

        if let Some(values) = incoming {
            let new_value = ValueId(self.value_id_feed.next().unwrap());
            self.basic_blocks[blk_id.0].ops.push((Some(new_value), Opcode::Phi(values)));
            self.block_info[blk_id.0].outgoing_values.push(new_value);
        }

        Self::gen_opcodes(
            self.m,
            &mut self.value_id_feed,
            &self.cfg.blocks[blk_id.0],
            &mut self.basic_blocks[blk_id.0],
            &mut self.block_info[blk_id.0]
        );
/*
        for pre in &pres {
            if self.block_info[pre.0].backedges.contains(blk_id) {
                continue;
            }
            self.analyze_one(*pre);
        }*/

/*
        let pre_block_ids = self.block_info[blk_id].pre.clone();

        for pre in pre_block_ids {
            self.analyze_one(pre);
        }*/
    }

    fn gen_opcodes(
        m: &Module,
        value_id_feed: &mut Range<usize>,
        blk: &::cfgraph::BasicBlock,
        out: &mut BasicBlock,
        blk_info: &mut BlockInfo
    ) {
        use ::opcode::Opcode as RawOp;

        for op in &blk.opcodes {
            let outgoing = &mut blk_info.outgoing_values;
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

                RawOp::I32Clz => impl_generic_unop!(I32Clz, outgoing, value_id_feed, out),
                RawOp::I32Ctz => impl_generic_unop!(I32Ctz, outgoing, value_id_feed, out),
                RawOp::I32Popcnt => impl_generic_unop!(I32Popcnt, outgoing, value_id_feed, out),

                RawOp::I32Add => impl_generic_binop!(I32Add, outgoing, value_id_feed, out),
                RawOp::I32Sub => impl_generic_binop!(I32Sub, outgoing, value_id_feed, out),
                RawOp::I32Mul => impl_generic_binop!(I32Mul, outgoing, value_id_feed, out),
                RawOp::I32DivU => impl_generic_binop!(I32DivU, outgoing, value_id_feed, out),
                RawOp::I32DivS => impl_generic_binop!(I32DivS, outgoing, value_id_feed, out),
                RawOp::I32RemU => impl_generic_binop!(I32RemU, outgoing, value_id_feed, out),
                RawOp::I32RemS => impl_generic_binop!(I32RemS, outgoing, value_id_feed, out),
                RawOp::I32And => impl_generic_binop!(I32And, outgoing, value_id_feed, out),
                RawOp::I32Or => impl_generic_binop!(I32Or, outgoing, value_id_feed, out),
                RawOp::I32Xor => impl_generic_binop!(I32Xor, outgoing, value_id_feed, out),
                RawOp::I32Shl => impl_generic_binop!(I32Shl, outgoing, value_id_feed, out),
                RawOp::I32ShrU => impl_generic_binop!(I32ShrU, outgoing, value_id_feed, out),
                RawOp::I32ShrS => impl_generic_binop!(I32ShrS, outgoing, value_id_feed, out),
                RawOp::I32Rotl => impl_generic_binop!(I32Rotl, outgoing, value_id_feed, out),
                RawOp::I32Rotr => impl_generic_binop!(I32Rotr, outgoing, value_id_feed, out),

                RawOp::I32Eqz => impl_generic_unop!(I32Eqz, outgoing, value_id_feed, out),

                RawOp::I32Eq => impl_generic_binop!(I32Eq, outgoing, value_id_feed, out),
                RawOp::I32Ne => impl_generic_binop!(I32Ne, outgoing, value_id_feed, out),
                RawOp::I32LtU => impl_generic_binop!(I32LtU, outgoing, value_id_feed, out),
                RawOp::I32LtS => impl_generic_binop!(I32LtS, outgoing, value_id_feed, out),
                RawOp::I32LeU => impl_generic_binop!(I32LeU, outgoing, value_id_feed, out),
                RawOp::I32LeS => impl_generic_binop!(I32LeS, outgoing, value_id_feed, out),
                RawOp::I32GtU => impl_generic_binop!(I32GtU, outgoing, value_id_feed, out),
                RawOp::I32GtS => impl_generic_binop!(I32GtS, outgoing, value_id_feed, out),
                RawOp::I32GeU => impl_generic_binop!(I32GeU, outgoing, value_id_feed, out),
                RawOp::I32GeS => impl_generic_binop!(I32GeS, outgoing, value_id_feed, out),

                RawOp::I32WrapI64 => impl_generic_unop!(I32WrapI64, outgoing, value_id_feed, out),

                RawOp::I32Load(m) => impl_mem_load!(I32Load, m, outgoing, value_id_feed, out),
                RawOp::I32Store(m) => impl_mem_store!(I32Store, m, outgoing, value_id_feed, out),
                RawOp::I32Load8U(m) => impl_mem_load!(I32Load8U, m, outgoing, value_id_feed, out),
                RawOp::I32Load8S(m) => impl_mem_load!(I32Load8S, m, outgoing, value_id_feed, out),
                RawOp::I32Load16U(m) => impl_mem_load!(I32Load16U, m, outgoing, value_id_feed, out),
                RawOp::I32Load16S(m) => impl_mem_load!(I32Load16S, m, outgoing, value_id_feed, out),
                RawOp::I32Store8(m) => impl_mem_store!(I32Store8, m, outgoing, value_id_feed, out),
                RawOp::I32Store16(m) => impl_mem_store!(I32Store16, m, outgoing, value_id_feed, out),

                RawOp::I64Const(v) => {
                    let val_id = ValueId(value_id_feed.next().unwrap());
                    outgoing.push(val_id);

                    out.ops.push((Some(val_id), Opcode::I64Const(v)));
                },

                RawOp::I64Clz => impl_generic_unop!(I64Clz, outgoing, value_id_feed, out),
                RawOp::I64Ctz => impl_generic_unop!(I64Ctz, outgoing, value_id_feed, out),
                RawOp::I64Popcnt => impl_generic_unop!(I64Popcnt, outgoing, value_id_feed, out),

                RawOp::I64Add => impl_generic_binop!(I64Add, outgoing, value_id_feed, out),
                RawOp::I64Sub => impl_generic_binop!(I64Sub, outgoing, value_id_feed, out),
                RawOp::I64Mul => impl_generic_binop!(I64Mul, outgoing, value_id_feed, out),
                RawOp::I64DivU => impl_generic_binop!(I64DivU, outgoing, value_id_feed, out),
                RawOp::I64DivS => impl_generic_binop!(I64DivS, outgoing, value_id_feed, out),
                RawOp::I64RemU => impl_generic_binop!(I64RemU, outgoing, value_id_feed, out),
                RawOp::I64RemS => impl_generic_binop!(I64RemS, outgoing, value_id_feed, out),
                RawOp::I64And => impl_generic_binop!(I64And, outgoing, value_id_feed, out),
                RawOp::I64Or => impl_generic_binop!(I64Or, outgoing, value_id_feed, out),
                RawOp::I64Xor => impl_generic_binop!(I64Xor, outgoing, value_id_feed, out),
                RawOp::I64Shl => impl_generic_binop!(I64Shl, outgoing, value_id_feed, out),
                RawOp::I64ShrU => impl_generic_binop!(I64ShrU, outgoing, value_id_feed, out),
                RawOp::I64ShrS => impl_generic_binop!(I64ShrS, outgoing, value_id_feed, out),
                RawOp::I64Rotl => impl_generic_binop!(I64Rotl, outgoing, value_id_feed, out),
                RawOp::I64Rotr => impl_generic_binop!(I64Rotr, outgoing, value_id_feed, out),

                RawOp::I64Eqz => impl_generic_unop!(I64Eqz, outgoing, value_id_feed, out),

                RawOp::I64Eq => impl_generic_binop!(I64Eq, outgoing, value_id_feed, out),
                RawOp::I64Ne => impl_generic_binop!(I64Ne, outgoing, value_id_feed, out),
                RawOp::I64LtU => impl_generic_binop!(I64LtU, outgoing, value_id_feed, out),
                RawOp::I64LtS => impl_generic_binop!(I64LtS, outgoing, value_id_feed, out),
                RawOp::I64LeU => impl_generic_binop!(I64LeU, outgoing, value_id_feed, out),
                RawOp::I64LeS => impl_generic_binop!(I64LeS, outgoing, value_id_feed, out),
                RawOp::I64GtU => impl_generic_binop!(I64GtU, outgoing, value_id_feed, out),
                RawOp::I64GtS => impl_generic_binop!(I64GtS, outgoing, value_id_feed, out),
                RawOp::I64GeU => impl_generic_binop!(I64GeU, outgoing, value_id_feed, out),
                RawOp::I64GeS => impl_generic_binop!(I64GeS, outgoing, value_id_feed, out),

                RawOp::I64ExtendI32U => impl_generic_unop!(I64ExtendI32U, outgoing, value_id_feed, out),
                RawOp::I64ExtendI32S => impl_generic_unop!(I64ExtendI32S, outgoing, value_id_feed, out),

                RawOp::I64Load(m) => impl_mem_load!(I64Load, m, outgoing, value_id_feed, out),
                RawOp::I64Store(m) => impl_mem_store!(I64Store, m, outgoing, value_id_feed, out),
                RawOp::I64Load8U(m) => impl_mem_load!(I64Load8U, m, outgoing, value_id_feed, out),
                RawOp::I64Load8S(m) => impl_mem_load!(I64Load8S, m, outgoing, value_id_feed, out),
                RawOp::I64Load16U(m) => impl_mem_load!(I64Load16U, m, outgoing, value_id_feed, out),
                RawOp::I64Load16S(m) => impl_mem_load!(I64Load16S, m, outgoing, value_id_feed, out),
                RawOp::I64Load32U(m) => impl_mem_load!(I64Load32U, m, outgoing, value_id_feed, out),
                RawOp::I64Load32S(m) => impl_mem_load!(I64Load32S, m, outgoing, value_id_feed, out),
                RawOp::I64Store8(m) => impl_mem_store!(I64Store8, m, outgoing, value_id_feed, out),
                RawOp::I64Store16(m) => impl_mem_store!(I64Store16, m, outgoing, value_id_feed, out),
                RawOp::I64Store32(m) => impl_mem_store!(I64Store32, m, outgoing, value_id_feed, out),

                RawOp::F32Const(v) => {
                    let val_id = ValueId(value_id_feed.next().unwrap());
                    outgoing.push(val_id);

                    out.ops.push((Some(val_id), Opcode::F32Const(v)));
                },
                RawOp::F64Const(v) => {
                    let val_id = ValueId(value_id_feed.next().unwrap());
                    outgoing.push(val_id);

                    out.ops.push((Some(val_id), Opcode::F64Const(v)));
                },

                RawOp::F32ReinterpretI32 => impl_generic_unop!(F32ReinterpretI32, outgoing, value_id_feed, out),
                RawOp::F64ReinterpretI64 => impl_generic_unop!(F64ReinterpretI64, outgoing, value_id_feed, out),
                RawOp::I32ReinterpretF32 => impl_generic_unop!(I32ReinterpretF32, outgoing, value_id_feed, out),
                RawOp::I64ReinterpretF64 => impl_generic_unop!(I64ReinterpretF64, outgoing, value_id_feed, out),
                RawOp::I32TruncSF32 => impl_generic_unop!(I32TruncSF32, outgoing, value_id_feed, out),
                RawOp::I32TruncUF32 => impl_generic_unop!(I32TruncUF32, outgoing, value_id_feed, out),
                RawOp::I32TruncSF64 => impl_generic_unop!(I32TruncSF64, outgoing, value_id_feed, out),
                RawOp::I32TruncUF64 => impl_generic_unop!(I32TruncUF64, outgoing, value_id_feed, out),
                RawOp::I64TruncSF32 => impl_generic_unop!(I64TruncSF32, outgoing, value_id_feed, out),
                RawOp::I64TruncUF32 => impl_generic_unop!(I64TruncUF32, outgoing, value_id_feed, out),
                RawOp::I64TruncSF64 => impl_generic_unop!(I64TruncSF64, outgoing, value_id_feed, out),
                RawOp::I64TruncUF64 => impl_generic_unop!(I64TruncUF64, outgoing, value_id_feed, out),
                RawOp::F32ConvertSI32 => impl_generic_unop!(F32ConvertSI32, outgoing, value_id_feed, out),
                RawOp::F32ConvertUI32 => impl_generic_unop!(F32ConvertUI32, outgoing, value_id_feed, out),
                RawOp::F32ConvertSI64 => impl_generic_unop!(F32ConvertSI64, outgoing, value_id_feed, out),
                RawOp::F32ConvertUI64 => impl_generic_unop!(F32ConvertUI64, outgoing, value_id_feed, out),
                RawOp::F64ConvertSI32 => impl_generic_unop!(F64ConvertSI32, outgoing, value_id_feed, out),
                RawOp::F64ConvertUI32 => impl_generic_unop!(F64ConvertUI32, outgoing, value_id_feed, out),
                RawOp::F64ConvertSI64 => impl_generic_unop!(F64ConvertSI64, outgoing, value_id_feed, out),
                RawOp::F64ConvertUI64 => impl_generic_unop!(F64ConvertUI64, outgoing, value_id_feed, out),
                RawOp::F32DemoteF64 => impl_generic_unop!(F32DemoteF64, outgoing, value_id_feed, out),
                RawOp::F64PromoteF32 => impl_generic_unop!(F64PromoteF32, outgoing, value_id_feed, out),
                RawOp::F32Abs => impl_generic_unop!(F32Abs, outgoing, value_id_feed, out),
                RawOp::F32Neg => impl_generic_unop!(F32Neg, outgoing, value_id_feed, out),
                RawOp::F32Ceil => impl_generic_unop!(F32Ceil, outgoing, value_id_feed, out),
                RawOp::F32Floor => impl_generic_unop!(F32Floor, outgoing, value_id_feed, out),
                RawOp::F32Trunc => impl_generic_unop!(F32Trunc, outgoing, value_id_feed, out),
                RawOp::F32Nearest => impl_generic_unop!(F32Nearest, outgoing, value_id_feed, out),
                RawOp::F32Sqrt => impl_generic_unop!(F32Sqrt, outgoing, value_id_feed, out),

                RawOp::F32Add => impl_generic_binop!(F32Add, outgoing, value_id_feed, out),
                RawOp::F32Sub => impl_generic_binop!(F32Sub, outgoing, value_id_feed, out),
                RawOp::F32Mul => impl_generic_binop!(F32Mul, outgoing, value_id_feed, out),
                RawOp::F32Div => impl_generic_binop!(F32Div, outgoing, value_id_feed, out),
                RawOp::F32Min => impl_generic_binop!(F32Min, outgoing, value_id_feed, out),
                RawOp::F32Max => impl_generic_binop!(F32Max, outgoing, value_id_feed, out),
                RawOp::F32Copysign => impl_generic_binop!(F32Copysign, outgoing, value_id_feed, out),
                RawOp::F32Eq => impl_generic_binop!(F32Eq, outgoing, value_id_feed, out),
                RawOp::F32Ne => impl_generic_binop!(F32Ne, outgoing, value_id_feed, out),
                RawOp::F32Lt => impl_generic_binop!(F32Lt, outgoing, value_id_feed, out),
                RawOp::F32Gt => impl_generic_binop!(F32Gt, outgoing, value_id_feed, out),
                RawOp::F32Le => impl_generic_binop!(F32Le, outgoing, value_id_feed, out),
                RawOp::F32Ge => impl_generic_binop!(F32Ge, outgoing, value_id_feed, out),

                RawOp::F64Abs => impl_generic_unop!(F64Abs, outgoing, value_id_feed, out),
                RawOp::F64Neg => impl_generic_unop!(F64Neg, outgoing, value_id_feed, out),
                RawOp::F64Ceil => impl_generic_unop!(F64Ceil, outgoing, value_id_feed, out),
                RawOp::F64Floor => impl_generic_unop!(F64Floor, outgoing, value_id_feed, out),
                RawOp::F64Trunc => impl_generic_unop!(F64Trunc, outgoing, value_id_feed, out),
                RawOp::F64Nearest => impl_generic_unop!(F64Nearest, outgoing, value_id_feed, out),
                RawOp::F64Sqrt => impl_generic_unop!(F64Sqrt, outgoing, value_id_feed, out),

                RawOp::F64Add => impl_generic_binop!(F64Add, outgoing, value_id_feed, out),
                RawOp::F64Sub => impl_generic_binop!(F64Sub, outgoing, value_id_feed, out),
                RawOp::F64Mul => impl_generic_binop!(F64Mul, outgoing, value_id_feed, out),
                RawOp::F64Div => impl_generic_binop!(F64Div, outgoing, value_id_feed, out),
                RawOp::F64Min => impl_generic_binop!(F64Min, outgoing, value_id_feed, out),
                RawOp::F64Max => impl_generic_binop!(F64Max, outgoing, value_id_feed, out),
                RawOp::F64Copysign => impl_generic_binop!(F64Copysign, outgoing, value_id_feed, out),
                RawOp::F64Eq => impl_generic_binop!(F64Eq, outgoing, value_id_feed, out),
                RawOp::F64Ne => impl_generic_binop!(F64Ne, outgoing, value_id_feed, out),
                RawOp::F64Lt => impl_generic_binop!(F64Lt, outgoing, value_id_feed, out),
                RawOp::F64Gt => impl_generic_binop!(F64Gt, outgoing, value_id_feed, out),
                RawOp::F64Le => impl_generic_binop!(F64Le, outgoing, value_id_feed, out),
                RawOp::F64Ge => impl_generic_binop!(F64Ge, outgoing, value_id_feed, out),

                RawOp::Jmp(_) | RawOp::JmpIf(_)
                | RawOp::JmpEither(_, _) | RawOp::JmpTable(_, _)
                | RawOp::Return => unreachable!(),

                RawOp::NativeInvoke(native_idx) => {
                    let f = &m.natives[native_idx as usize];
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
                        Opcode::NativeInvoke(native_idx, args)
                    ));
                },

                RawOp::Memcpy => {
                    let n_bytes = outgoing.pop().unwrap();
                    let src = outgoing.pop().unwrap();
                    let dest = outgoing.pop().unwrap();
                    out.ops.push((None, Opcode::Memcpy(dest, src, n_bytes)));
                },

                RawOp::NotImplemented(ref s) => {
                    out.ops.push((None, Opcode::Unreachable));
                    terminate = true;
                }
            }

            if terminate {
                break;
            }
        }

        out.br = Some(match *blk.br.as_ref().unwrap() {
            ::cfgraph::Branch::Jmp(::cfgraph::BlockId(id)) => {
                Branch::Br(BlockId(id))
            },
            ::cfgraph::Branch::JmpEither(
                ::cfgraph::BlockId(a),
                ::cfgraph::BlockId(b)
            ) => {
                Branch::BrEither(
                    blk_info.outgoing_values.pop().unwrap(),
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
                    out_targets.push(BlockId(t.0));
                }

                Branch::BrTable(
                    blk_info.outgoing_values.pop().unwrap(),
                    out_targets,
                    BlockId(otherwise.0)
                )
            },
            ::cfgraph::Branch::Return => {
                Branch::Return(blk_info.outgoing_values.pop())
            }
        });
    }
}

impl FlowGraph {
    pub fn from_cfg(cfg: &::cfgraph::CFGraph, m: &Module) -> FlowGraph {
        let mut analyzer = Analyzer::new(cfg, m);
        analyzer.generate_all();

        FlowGraph {
            blocks: analyzer.basic_blocks
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
                out[i].succ.insert(BlockId(id));
            },
            Branch::JmpEither(::cfgraph::BlockId(a), ::cfgraph::BlockId(b)) => {
                out[a].pre.insert(BlockId(i));
                out[b].pre.insert(BlockId(i));
                out[i].succ.insert(BlockId(a));
                out[i].succ.insert(BlockId(b));
            },
            Branch::JmpTable(ref targets, ::cfgraph::BlockId(otherwise)) => {
                for t in targets {
                    out[t.0].pre.insert(BlockId(i));
                    out[i].succ.insert(BlockId(t.0));
                }
                out[otherwise].pre.insert(BlockId(i));
                out[i].succ.insert(BlockId(otherwise));
            }
            Branch::Return => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cycle_analysis() {
        use ::opcode::Opcode as RawOp;
        let opcodes = &[
            RawOp::I32Const(42), // 0

            RawOp::Jmp(2), // 1

            RawOp::I32Const(0), // 2
            RawOp::JmpEither(4, 7), // 3

            RawOp::Jmp(5), // 4

            RawOp::I32Const(1), // 5
            RawOp::Jmp(9), // 6

            RawOp::I32Const(2), // 7
            RawOp::Jmp(9), // 8

            RawOp::Jmp(10), // 9

            RawOp::JmpEither(2, 11), // 10

            RawOp::Return // 11
        ];

        let cfg = ::cfgraph::CFGraph::from_function(opcodes).unwrap();
        let m = Module::default();

        let mut analyzer = Analyzer::new(&cfg, &m);
        analyzer.generate_all();

        println!("{:?}", analyzer.block_info);
        println!("{:?}", analyzer.basic_blocks);

        /*assert_eq!(analyzer.block_info.len(), 7);
        assert_eq!(analyzer.block_info[0].cycle, false);
        assert_eq!(analyzer.block_info[1].cycle, true);
        assert_eq!(analyzer.block_info[2].cycle, false);
        assert_eq!(analyzer.block_info[3].cycle, false);
        assert_eq!(analyzer.block_info[4].cycle, false);
        assert_eq!(analyzer.block_info[5].cycle, false);
        assert_eq!(analyzer.block_info[6].cycle, false);*/
    }

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
            Opcode::Phi(vec! [ ValueId(1), ValueId(2) ])
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
