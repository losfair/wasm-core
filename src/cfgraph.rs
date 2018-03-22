use opcode::Opcode;
use prelude::{BTreeMap, BTreeSet};

#[derive(Clone, Debug)]
pub struct CFGraph {
    blocks: Vec<BasicBlock>
}

#[derive(Clone, Debug)]
pub struct BasicBlock {
    opcodes: Vec<Opcode>,
    br: Option<Branch> // must be Some in a valid control graph
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Branch {
    Jmp(BlockId),
    JmpIf(BlockId),
    JmpTable(Vec<BlockId>, BlockId),
    Return
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct BlockId(usize);

impl CFGraph {
    pub fn from_function(fops: &[Opcode]) -> CFGraph {
        CFGraph {
            blocks: scan_basic_blocks(fops)
        }
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
                        Branch::JmpIf(BlockId(id)) => Opcode::JmpIf(begin_instrs[id]),
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
            Opcode::Jmp(_) | Opcode::JmpIf(_) | Opcode::JmpTable(_, _) | Opcode::Return => true,
            _ => false
        }
    }
}

/// Constructs a Vec of basic blocks.
fn scan_basic_blocks(ops: &[Opcode]) -> Vec<BasicBlock> {
    let mut bbs: Vec<BasicBlock> = vec! [ ];
    let mut mappings: BTreeMap<u32, BlockId> = BTreeMap::new();
    let mut jmp_targets: BTreeSet<u32> = BTreeSet::new();

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
                    Opcode::Jmp(target) => Branch::Jmp(*mappings.get(&target).unwrap()),
                    Opcode::JmpIf(target) => Branch::JmpIf(*mappings.get(&target).unwrap()),
                    Opcode::JmpTable(ref targets, otherwise) => {
                        Branch::JmpTable(
                            targets.iter().map(|t| *mappings.get(t).unwrap()).collect(),
                            *mappings.get(&otherwise).unwrap()
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

    bbs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cfg_build() {
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

        let cfg = CFGraph::from_function(opcodes.as_slice());

        assert_eq!(cfg.blocks.len(), 3);
        assert_eq!(cfg.blocks[0].br, Some(Branch::Jmp(BlockId(2))));
        assert_eq!(cfg.blocks[1].br, Some(Branch::Jmp(BlockId(2))));
        assert_eq!(cfg.blocks[2].br, Some(Branch::Return));

        eprintln!("{:?}", cfg);

        eprintln!("{:?}", cfg.gen_opcodes());
    }
}
