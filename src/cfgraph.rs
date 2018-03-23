use opcode::Opcode;
use prelude::{BTreeMap, BTreeSet};

#[derive(Clone, Debug)]
pub struct CFGraph {
    pub blocks: Vec<BasicBlock>
}

#[derive(Clone, Debug)]
pub struct BasicBlock {
    pub opcodes: Vec<Opcode>,
    pub br: Option<Branch> // must be Some in a valid control graph
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Branch {
    Jmp(BlockId),
    JmpEither(BlockId, BlockId), // (if_true, if_false)
    JmpTable(Vec<BlockId>, BlockId),
    Return
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct BlockId(pub usize);

pub type OptimizeResult<T> = Result<T, OptimizeError>;

#[derive(Clone, Debug)]
pub enum OptimizeError {
    InvalidBranchTarget,
    Custom(String)
}

pub trait Optimizer {
    type Return;

    fn optimize(&self, cfg: &mut CFGraph) -> OptimizeResult<Self::Return>;
}

fn _assert_optimizer_trait_object_safe() {
    struct Opt {}
    impl Optimizer for Opt {
        type Return = ();
        fn optimize(&self, _: &mut CFGraph) -> OptimizeResult<Self::Return> { Ok(()) }
    }

    let _obj: Box<Optimizer<Return = ()>> = Box::new(Opt {});
}

trait CheckedBranchTarget {
    type TValue;

    fn checked_branch_target(&self) -> OptimizeResult<Self::TValue>;
}

impl<'a> CheckedBranchTarget for Option<&'a BlockId> {
    type TValue = BlockId;

    fn checked_branch_target(&self) -> OptimizeResult<BlockId> {
        match *self {
            Some(v) => Ok(*v),
            None => Err(OptimizeError::InvalidBranchTarget)
        }
    }
}

impl CFGraph {
    pub fn from_function(fops: &[Opcode]) -> OptimizeResult<CFGraph> {
        Ok(CFGraph {
            blocks: scan_basic_blocks(fops)?
        })
    }

    pub fn validate(&self) -> OptimizeResult<()> {
        for blk in &self.blocks {
            for op in &blk.opcodes {
                if op.is_branch() {
                    return Err(OptimizeError::Custom(
                        "Branch instruction(s) found in the middle of a basic block".into()
                    ));
                }
            }
            let br = if let Some(ref br) = blk.br {
                br
            } else {
                return Err(OptimizeError::Custom(
                    "Empty branch target(s) found".into()
                ));
            };
            let br_ok = match *br {
                Branch::Jmp(id) => {
                    if id.0 >= self.blocks.len() {
                        false
                    } else {
                        true
                    }
                },
                Branch::JmpEither(a, b) => {
                    if a.0 >= self.blocks.len() || b.0 >= self.blocks.len() {
                        false
                    } else {
                        true
                    }
                },
                Branch::JmpTable(ref targets, otherwise) => {
                    let mut ok = true;
                    for t in targets {
                        if t.0 >= self.blocks.len() {
                            ok = false;
                            break;
                        }
                    }
                    if ok {
                        if otherwise.0 >= self.blocks.len() {
                            false
                        } else {
                            true
                        }
                    } else {
                        false
                    }
                },
                Branch::Return => true
            };
            if !br_ok {
                return Err(OptimizeError::Custom(
                    "Invalid branch target(s)".into()
                ));
            }
        }

        Ok(())
    }

    /// Generate sequential opcodes.
    pub fn gen_opcodes(&self) -> Vec<Opcode> {
        enum OpOrBr {
            Op(Opcode),
            Br(Branch) // pending branch to basic block
        }

        let mut seq: Vec<OpOrBr> = Vec::new();
        let mut begin_instrs: Vec<u32> = Vec::with_capacity(self.blocks.len());

        for (i, bb) in self.blocks.iter().enumerate() {
            begin_instrs.push(seq.len() as u32);
            for op in &bb.opcodes {
                seq.push(OpOrBr::Op(op.clone()));
            }
            seq.push(OpOrBr::Br(bb.br.as_ref().unwrap().clone()));
        }

        seq.into_iter().map(|oob| {
            match oob {
                OpOrBr::Op(op) => op,
                OpOrBr::Br(br) => {
                    match br {
                        Branch::Jmp(BlockId(id)) => Opcode::Jmp(begin_instrs[id]),
                        Branch::JmpEither(BlockId(if_true), BlockId(if_false)) => {
                            Opcode::JmpEither(
                                begin_instrs[if_true],
                                begin_instrs[if_false]
                            )
                        },
                        Branch::JmpTable(targets, BlockId(otherwise)) => Opcode::JmpTable(
                            targets.into_iter().map(|BlockId(id)| begin_instrs[id]).collect(),
                            begin_instrs[otherwise]
                        ),
                        Branch::Return => Opcode::Return
                    }
                }
            }
        }).collect()
    }

    pub fn optimize<
        T: Optimizer<Return = R>,
        R
    >(&mut self, optimizer: T) -> OptimizeResult<R> {
        optimizer.optimize(self)
    }
}

impl BasicBlock {
    pub fn new() -> BasicBlock {
        BasicBlock {
            opcodes: vec! [],
            br: None
        }
    }
}

impl Opcode {
    fn is_branch(&self) -> bool {
        match *self {
            Opcode::Jmp(_) | Opcode::JmpIf(_) | Opcode::JmpEither(_, _) | Opcode::JmpTable(_, _) | Opcode::Return => true,
            _ => false
        }
    }
}

/// Constructs a Vec of basic blocks.
fn scan_basic_blocks(ops: &[Opcode]) -> OptimizeResult<Vec<BasicBlock>> {
    if ops.len() == 0 {
        return Ok(Vec::new());
    }

    let mut jmp_targets: BTreeSet<u32> = BTreeSet::new();

    // Entry point.
    jmp_targets.insert(0);

    {
        // Detect jmp targets
        for (i, op) in ops.iter().enumerate() {
            if op.is_branch() {
                match *op {
                    Opcode::Jmp(id) => {
                        jmp_targets.insert(id);
                    },
                    Opcode::JmpIf(id) => {
                        jmp_targets.insert(id);
                    },
                    Opcode::JmpEither(a, b) => {
                        jmp_targets.insert(a);
                        jmp_targets.insert(b);
                    },
                    Opcode::JmpTable(ref targets, otherwise) => {
                        for t in targets {
                            jmp_targets.insert(*t);
                        }
                        jmp_targets.insert(otherwise);
                    },
                    Opcode::Return => {},
                    _ => unreachable!()
                }

                // The instruction following a branch starts a new basic block.
                jmp_targets.insert((i + 1) as u32);
            }
        }
    }

    // Split opcodes into basic blocks
    let (bb_ops, instr_mappings): (Vec<&[Opcode]>, BTreeMap<u32, BlockId>) = {
        let mut bb_ops: Vec<&[Opcode]> = Vec::new();
        let mut instr_mappings: BTreeMap<u32, BlockId> = BTreeMap::new();

        // jmp_targets.len() >= 1 holds here because of `jmp_targets.insert(0)`
        let mut jmp_targets: Vec<u32> = jmp_targets.iter().map(|v| *v).collect();

        // [start, end) ...
        // ops.len
        {
            let last = *jmp_targets.last().unwrap() as usize;
            if last > ops.len() {
                return Err(OptimizeError::InvalidBranchTarget);
            }

            // ops.len() >= 1 holds here.
            // if last == 0 (same as jmp_targets.len() == 1) then a new jmp target will still be pushed
            // so that jmp_targets.len() >= 2 always hold after this.
            if last < ops.len() {
                jmp_targets.push(ops.len() as u32);
            }
        }

        for i in 0..jmp_targets.len() - 1 {
            // [st..ed)
            let st = jmp_targets[i] as usize;
            let ed = jmp_targets[i + 1] as usize;
            instr_mappings.insert(st as u32, BlockId(bb_ops.len()));
            bb_ops.push(&ops[st..ed]);
        }

        (bb_ops, instr_mappings)
    };

    let mut bbs: Vec<BasicBlock> = Vec::new();

    for (i, bb) in bb_ops.iter().enumerate() {
        let mut bb = bb.to_vec();

        let br: Option<Branch> = if let Some(op) = bb.last() {
            if op.is_branch() {
                Some(match *op {
                    Opcode::Jmp(target) => Branch::Jmp(instr_mappings.get(&target).checked_branch_target()?),
                    Opcode::JmpIf(target) => Branch::JmpEither(
                        instr_mappings.get(&target).checked_branch_target()?, // if true
                        BlockId(i + 1) // otherwise
                    ),
                    Opcode::JmpEither(a, b) => Branch::JmpEither(
                        instr_mappings.get(&a).checked_branch_target()?,
                        instr_mappings.get(&b).checked_branch_target()?
                    ),
                    Opcode::JmpTable(ref targets, otherwise) => {
                        let mut br_targets: Vec<BlockId> = Vec::new();
                        for t in targets {
                            br_targets.push(instr_mappings.get(t).checked_branch_target()?);
                        }
                        Branch::JmpTable(
                            br_targets,
                            instr_mappings.get(&otherwise).checked_branch_target()?
                        )
                    },
                    Opcode::Return => Branch::Return,
                    _ => unreachable!()
                })
            } else {
                None
            }
        } else {
            None
        };

        let br: Branch = if let Some(v) = br {
            bb.pop().unwrap();
            v
        } else {
            Branch::Jmp(BlockId(i + 1))
        };

        let mut result = BasicBlock::new();
        result.opcodes = bb;
        result.br = Some(br);

        bbs.push(result);
    }

    Ok(bbs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jmp() {
        let opcodes: Vec<Opcode> = vec! [
            // bb 0
            Opcode::I32Const(100), // 0
            Opcode::Jmp(3), // 1
            // bb 1, implicit fallthrough
            Opcode::I32Const(50), // 2
            // bb 2 (due to jmp)
            Opcode::I32Const(25), // 3
            Opcode::Return // 4
        ];

        let cfg = CFGraph::from_function(opcodes.as_slice()).unwrap();
        cfg.validate().unwrap();

        assert_eq!(cfg.blocks.len(), 3);
        assert_eq!(cfg.blocks[0].br, Some(Branch::Jmp(BlockId(2))));
        assert_eq!(cfg.blocks[1].br, Some(Branch::Jmp(BlockId(2))));
        assert_eq!(cfg.blocks[2].br, Some(Branch::Return));

        eprintln!("{:?}", cfg);

        eprintln!("{:?}", cfg.gen_opcodes());
    }

    #[test]
    fn test_jmp_if() {
        let opcodes: Vec<Opcode> = vec! [
            // bb 0
            Opcode::I32Const(100), // 0
            Opcode::JmpIf(3), // 1
            // bb 1, implicit fallthrough
            Opcode::I32Const(50), // 2
            // bb 2 (due to jmp)
            Opcode::I32Const(25), // 3
            Opcode::Return // 4
        ];

        let cfg = CFGraph::from_function(opcodes.as_slice()).unwrap();
        cfg.validate().unwrap();

        assert_eq!(cfg.blocks.len(), 3);
        assert_eq!(cfg.blocks[0].br, Some(Branch::JmpEither(BlockId(2), BlockId(1))));
        assert_eq!(cfg.blocks[1].br, Some(Branch::Jmp(BlockId(2))));
        assert_eq!(cfg.blocks[2].br, Some(Branch::Return));

        eprintln!("{:?}", cfg);

        eprintln!("{:?}", cfg.gen_opcodes());
    }

    #[test]
    fn test_circular() {
        let opcodes: Vec<Opcode> = vec! [
            // bb 1
            Opcode::I32Const(100), // 0
            Opcode::JmpIf(0),
            // bb 2
            Opcode::Return // 4
        ];

        let cfg = CFGraph::from_function(opcodes.as_slice()).unwrap();
        cfg.validate().unwrap();

        assert_eq!(cfg.blocks.len(), 2);
        assert_eq!(cfg.blocks[0].br, Some(Branch::JmpEither(BlockId(0), BlockId(1))));

        eprintln!("{:?}", cfg);

        eprintln!("{:?}", cfg.gen_opcodes());
    }

    #[test]
    fn test_invalid_branch_target() {
        let opcodes: Vec<Opcode> = vec! [ Opcode::Jmp(10) ];
        match CFGraph::from_function(opcodes.as_slice()) {
            Err(OptimizeError::InvalidBranchTarget) => {},
            _ => panic!("Expecting an InvalidBranchTarget error")
        }
    }
}
