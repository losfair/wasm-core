use cfgraph::*;
use prelude::{BTreeSet, BTreeMap};

pub struct RemoveDeadBasicBlocks;

impl Optimizer for RemoveDeadBasicBlocks {
    type Return = ();

    fn optimize(&self, cfg: &mut CFGraph) -> OptimizeResult<()> {
        if cfg.blocks.len() == 0 {
            return Ok(());
        }

        let mut reachable: BTreeSet<BlockId> = BTreeSet::new();

        // Perform a depth-first search on the CFG to figure out reachable blocks.
        {
            let mut dfs_stack: Vec<BlockId> = vec! [ BlockId(0) ];

            while let Some(blk_id) = dfs_stack.pop() {
                if reachable.contains(&blk_id) {
                    continue;
                }

                reachable.insert(blk_id);

                let blk = &cfg.blocks[blk_id.0];
                match *blk.br.as_ref().unwrap() {
                    Branch::Jmp(t) => {
                        dfs_stack.push(t);
                    },
                    Branch::JmpEither(a, b) => {
                        dfs_stack.push(a);
                        dfs_stack.push(b);
                    },
                    Branch::JmpTable(ref targets, otherwise) => {
                        for t in targets {
                            dfs_stack.push(*t);
                        }
                        dfs_stack.push(otherwise);
                    },
                    Branch::Return => {}
                }
            }
        }

        // Maps old block ids to new ones.
        let mut block_id_mappings: BTreeMap<BlockId, BlockId> = BTreeMap::new();

        // Reachable basic blocks
        let mut new_basic_blocks = Vec::with_capacity(reachable.len());

        {
            // Old basic blocks
            let mut old_basic_blocks = ::prelude::mem::replace(&mut cfg.blocks, Vec::new());

            // reachable is a Set so blk_id will never duplicate.
            for (i, blk_id) in reachable.iter().enumerate() {
                block_id_mappings.insert(*blk_id, BlockId(i));
                new_basic_blocks.push(
                    ::prelude::mem::replace(
                        &mut old_basic_blocks[blk_id.0],
                        BasicBlock::new()
                    )
                );
            }
        }

        for bb in &mut new_basic_blocks {
            let old_br = bb.br.take().unwrap();
            bb.br = Some(match old_br {
                Branch::Jmp(id) => Branch::Jmp(*block_id_mappings.get(&id).unwrap()),
                Branch::JmpEither(a, b) => Branch::JmpEither(
                    *block_id_mappings.get(&a).unwrap(),
                    *block_id_mappings.get(&b).unwrap()
                ),
                Branch::JmpTable(targets, otherwise) => Branch::JmpTable(
                    targets.into_iter().map(|t| *block_id_mappings.get(&t).unwrap()).collect(),
                    *block_id_mappings.get(&otherwise).unwrap()
                ),
                Branch::Return => Branch::Return
            });
        }

        cfg.blocks = new_basic_blocks;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opcode::Opcode;

    #[test]
    fn test_remove_dead_basic_blocks() {
        let opcodes: Vec<Opcode> = vec! [
            // bb 1
            Opcode::I32Const(100), // 0
            Opcode::Jmp(3), // 1
            // bb 2, never reached
            Opcode::I32Const(50), // 2
            // bb 3 (due to jmp)
            Opcode::I32Const(25), // 3
            Opcode::JmpIf(0), // 4
            // bb 4
            Opcode::Return // 5
        ];

        let mut cfg = CFGraph::from_function(opcodes.as_slice()).unwrap();
        cfg.optimize(RemoveDeadBasicBlocks).unwrap();

        assert_eq!(cfg.blocks.len(), 4);
        assert_eq!(cfg.blocks[1].br, Some(Branch::Jmp(BlockId(2))));
        assert_eq!(cfg.blocks[2].br, Some(Branch::JmpEither(BlockId(1), BlockId(3))));
        assert_eq!(cfg.blocks[3].br, Some(Branch::Return));

        eprintln!("{:?}", cfg);

        eprintln!("{:?}", cfg.gen_opcodes());
    }
}
