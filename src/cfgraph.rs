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
    // The "initial" basic block only jumps to the real first one.
    let mut bbs: Vec<BasicBlock> = vec! [ BasicBlock::new() ];

    let mut mappings: BTreeMap<u32, BlockId> = BTreeMap::new();
    let mut jmp_targets: BTreeSet<u32> = BTreeSet::new();

    // Entry point.
    jmp_targets.insert(0);

    {
        // Detect jmp targets
        for op in ops.iter() {
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
            }
        }

        let mut is_first_instr: bool = true;

        for (i, op) in ops.iter().enumerate() {
            // If we are at the first instruction after a jmp or the current
            // instruction is a target of another jmp, then we are in a new
            // basic block
            if is_first_instr || jmp_targets.contains(&(i as u32)) {
                // Add it into mappings
                mappings.insert(i as u32, BlockId(bbs.len()));

                // The newly-entered basic block
                bbs.push(BasicBlock::new());

                is_first_instr = false;
            }

            // We will be in a new basic block after branching
            if op.is_branch() {
                is_first_instr = true;
            }
        }
    }

    {
        let mut current_bb: usize = 0;

        for (i, op) in ops.iter().enumerate() {
            if op.is_branch() {
                bbs[current_bb].br = Some(match *op {
                    Opcode::Jmp(target) => Branch::Jmp(mappings.get(&target).checked_branch_target()?),
                    Opcode::JmpIf(target) => Branch::JmpEither(
                        mappings.get(&target).checked_branch_target()?, // if true
                        mappings.get(&((i + 1) as u32)).checked_branch_target()? // otherwise
                    ),
                    Opcode::JmpEither(a, b) => Branch::JmpEither(
                        mappings.get(&a).checked_branch_target()?,
                        mappings.get(&b).checked_branch_target()?
                    ),
                    Opcode::JmpTable(ref targets, otherwise) => {
                        let mut br_targets: Vec<BlockId> = Vec::new();
                        for t in targets {
                            br_targets.push(mappings.get(t).checked_branch_target()?);
                        }
                        Branch::JmpTable(
                            br_targets,
                            mappings.get(&otherwise).checked_branch_target()?
                        )
                    },
                    Opcode::Return => Branch::Return,
                    _ => unreachable!()
                });
                current_bb += 1;
            } else {
                if jmp_targets.contains(&(i as u32)) { // implicit fallthrough
                    bbs[current_bb].br = Some(Branch::Jmp(BlockId(current_bb + 1)));
                    current_bb += 1;
                }
                bbs[current_bb].opcodes.push(op.clone());
            }
        }

        assert_eq!(current_bb, bbs.len());
    }

    Ok(bbs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jmp() {
        let opcodes: Vec<Opcode> = vec! [
            // bb 1
            Opcode::I32Const(100), // 0
            Opcode::Jmp(3), // 1
            // bb 2, implicit fallthrough
            Opcode::I32Const(50), // 2
            // bb 3 (due to jmp)
            Opcode::I32Const(25), // 3
            Opcode::Return // 4
        ];

        let cfg = CFGraph::from_function(opcodes.as_slice()).unwrap();

        assert_eq!(cfg.blocks.len(), 4);
        assert_eq!(cfg.blocks[1].br, Some(Branch::Jmp(BlockId(3))));
        assert_eq!(cfg.blocks[2].br, Some(Branch::Jmp(BlockId(3))));
        assert_eq!(cfg.blocks[3].br, Some(Branch::Return));

        eprintln!("{:?}", cfg);

        eprintln!("{:?}", cfg.gen_opcodes());
    }

    #[test]
    fn test_jmp_if() {
        let opcodes: Vec<Opcode> = vec! [
            // bb 1
            Opcode::I32Const(100), // 0
            Opcode::JmpIf(3), // 1
            // bb 2, implicit fallthrough
            Opcode::I32Const(50), // 2
            // bb 3 (due to jmp)
            Opcode::I32Const(25), // 3
            Opcode::Return // 4
        ];

        let cfg = CFGraph::from_function(opcodes.as_slice()).unwrap();

        assert_eq!(cfg.blocks.len(), 4);
        assert_eq!(cfg.blocks[1].br, Some(Branch::JmpEither(BlockId(3), BlockId(2))));
        assert_eq!(cfg.blocks[2].br, Some(Branch::Jmp(BlockId(3))));
        assert_eq!(cfg.blocks[3].br, Some(Branch::Return));

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

        assert_eq!(cfg.blocks.len(), 3);
        assert_eq!(cfg.blocks[1].br, Some(Branch::JmpEither(BlockId(1), BlockId(2))));

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
