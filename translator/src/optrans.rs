use wasm_core;
use wasm_core::opcode::Memarg;
use parity_wasm::elements;

struct Continuation {
    opcode_index: usize,
    brtable_index: Option<usize>
}

impl Continuation {
    fn with_opcode_index(index: usize) -> Continuation {
        Continuation {
            opcode_index: index,
            brtable_index: None
        }
    }

    fn brtable(index: usize, brt_index: usize) -> Continuation {
        Continuation {
            opcode_index: index,
            brtable_index: Some(brt_index)
        }
    }

    fn write(&self, target: usize, opcodes: &mut [wasm_core::opcode::Opcode]) {
        use self::wasm_core::opcode::Opcode;

        let op_index = self.opcode_index;

        let new_op = match ::std::mem::replace(
            &mut opcodes[op_index],
            Opcode::Unreachable
        ) {
            Opcode::Jmp(_) => Opcode::Jmp(target as u32),
            Opcode::JmpIf(_) => Opcode::JmpIf(target as u32),
            Opcode::JmpTable(mut table, otherwise) => {
                let table_index = self.brtable_index.unwrap();
                if table_index < table.len() {
                    table[table_index] = target as u32;
                    Opcode::JmpTable(table, otherwise)
                } else if table_index == table.len() {
                    Opcode::JmpTable(table, target as u32)
                } else {
                    panic!("Table index out of bound");
                }
            },
            _ => panic!("Expecting Jmp*")
        };
        opcodes[op_index] = new_op;
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum LabelType {
    Block,
    Loop(usize), // begin
    If(usize), // branch-if-false instr
    Else
}

struct Label {
    continuations: Vec<Continuation>,
    ty: LabelType
}

impl Label {
    fn new(ty: LabelType) -> Label {
        Label {
            continuations: Vec::new(),
            ty: ty
        }
    }

    fn terminate(&self, opcodes: &mut [wasm_core::opcode::Opcode]) {
        let target = match self.ty {
            LabelType::Block | LabelType::If(_) | LabelType::Else => opcodes.len(),
            LabelType::Loop(begin) => begin
        };
        for cont in &self.continuations {
            cont.write(target, opcodes);
        }
    }
}

pub fn translate_opcodes(ops: &[elements::Opcode]) -> Vec<wasm_core::opcode::Opcode> {
    use self::elements::Opcode as PwOp;
    use self::wasm_core::opcode::Opcode as WcOp;

    let mut result: Vec<wasm_core::opcode::Opcode> = Vec::new();
    let mut labels: Vec<Label> = Vec::new();
    let mut expecting_seq_end = false;

    //dprintln!("{:?}", ops);

    for op in ops {
        if expecting_seq_end {
            panic!("Expecting end of opcode sequence");
        }
        match *op {
            PwOp::Drop => result.push(WcOp::Drop),
            PwOp::Select => result.push(WcOp::Select),

            PwOp::GetLocal(id) => result.push(WcOp::GetLocal(id)),
            PwOp::SetLocal(id) => result.push(WcOp::SetLocal(id)),
            PwOp::TeeLocal(id) => result.push(WcOp::TeeLocal(id)),
            PwOp::GetGlobal(id) => result.push(WcOp::GetGlobal(id)),
            PwOp::SetGlobal(id) => result.push(WcOp::SetGlobal(id)),

            PwOp::CurrentMemory(_) => result.push(WcOp::CurrentMemory),
            PwOp::GrowMemory(_) => result.push(WcOp::GrowMemory),

            PwOp::Nop => result.push(WcOp::Nop),
            PwOp::Unreachable => result.push(WcOp::Unreachable),
            PwOp::Return => result.push(WcOp::Return),
            PwOp::Call(id) => result.push(WcOp::Call(id)),
            PwOp::CallIndirect(id, _) => result.push(WcOp::CallIndirect(id)),

            PwOp::I32Const(v) => result.push(WcOp::I32Const(v)),
            
            PwOp::I32Clz => result.push(WcOp::I32Clz),
            PwOp::I32Ctz => result.push(WcOp::I32Ctz),
            PwOp::I32Popcnt => result.push(WcOp::I32Popcnt),

            PwOp::I32Add => result.push(WcOp::I32Add),
            PwOp::I32Sub => result.push(WcOp::I32Sub),
            PwOp::I32Mul => result.push(WcOp::I32Mul),
            PwOp::I32DivU => result.push(WcOp::I32DivU),
            PwOp::I32DivS => result.push(WcOp::I32DivS),
            PwOp::I32RemU => result.push(WcOp::I32RemU),
            PwOp::I32RemS => result.push(WcOp::I32RemS),
            PwOp::I32And => result.push(WcOp::I32And),
            PwOp::I32Or => result.push(WcOp::I32Or),
            PwOp::I32Xor => result.push(WcOp::I32Xor),
            PwOp::I32Shl => result.push(WcOp::I32Shl),
            PwOp::I32ShrU => result.push(WcOp::I32ShrU),
            PwOp::I32ShrS => result.push(WcOp::I32ShrS),
            PwOp::I32Rotl => result.push(WcOp::I32Rotl),
            PwOp::I32Rotr => result.push(WcOp::I32Rotr),

            PwOp::I32Eqz => result.push(WcOp::I32Eqz),

            PwOp::I32Eq => result.push(WcOp::I32Eq),
            PwOp::I32Ne => result.push(WcOp::I32Ne),
            PwOp::I32LtU => result.push(WcOp::I32LtU),
            PwOp::I32LtS => result.push(WcOp::I32LtS),
            PwOp::I32LeU => result.push(WcOp::I32LeU),
            PwOp::I32LeS => result.push(WcOp::I32LeS),
            PwOp::I32GtU => result.push(WcOp::I32GtU),
            PwOp::I32GtS => result.push(WcOp::I32GtS),
            PwOp::I32GeU => result.push(WcOp::I32GeU),
            PwOp::I32GeS => result.push(WcOp::I32GeS),

            PwOp::I32WrapI64 => result.push(WcOp::I32WrapI64),

            PwOp::I32Load(align, offset) => result.push(WcOp::I32Load(Memarg { offset: offset, align: align })),
            PwOp::I32Store(align, offset) => result.push(WcOp::I32Store(Memarg { offset: offset, align: align })),
            PwOp::I32Load8U(align, offset) => result.push(WcOp::I32Load8U(Memarg { offset: offset, align: align })),
            PwOp::I32Load8S(align, offset) => result.push(WcOp::I32Load8S(Memarg { offset: offset, align: align })),
            PwOp::I32Load16U(align, offset) => result.push(WcOp::I32Load16U(Memarg { offset: offset, align: align })),
            PwOp::I32Load16S(align, offset) => result.push(WcOp::I32Load16S(Memarg { offset: offset, align: align })),
            PwOp::I32Store8(align, offset) => result.push(WcOp::I32Store8(Memarg { offset: offset, align: align })),
            PwOp::I32Store16(align, offset) => result.push(WcOp::I32Store16(Memarg { offset: offset, align: align })),

            PwOp::I64Const(v) => result.push(WcOp::I64Const(v)),
            
            PwOp::I64Clz => result.push(WcOp::I64Clz),
            PwOp::I64Ctz => result.push(WcOp::I64Ctz),
            PwOp::I64Popcnt => result.push(WcOp::I64Popcnt),

            PwOp::I64Add => result.push(WcOp::I64Add),
            PwOp::I64Sub => result.push(WcOp::I64Sub),
            PwOp::I64Mul => result.push(WcOp::I64Mul),
            PwOp::I64DivU => result.push(WcOp::I64DivU),
            PwOp::I64DivS => result.push(WcOp::I64DivS),
            PwOp::I64RemU => result.push(WcOp::I64RemU),
            PwOp::I64RemS => result.push(WcOp::I64RemS),
            PwOp::I64And => result.push(WcOp::I64And),
            PwOp::I64Or => result.push(WcOp::I64Or),
            PwOp::I64Xor => result.push(WcOp::I64Xor),
            PwOp::I64Shl => result.push(WcOp::I64Shl),
            PwOp::I64ShrU => result.push(WcOp::I64ShrU),
            PwOp::I64ShrS => result.push(WcOp::I64ShrS),
            PwOp::I64Rotl => result.push(WcOp::I64Rotl),
            PwOp::I64Rotr => result.push(WcOp::I64Rotr),

            PwOp::I64Eqz => result.push(WcOp::I64Eqz),

            PwOp::I64Eq => result.push(WcOp::I64Eq),
            PwOp::I64Ne => result.push(WcOp::I64Ne),
            PwOp::I64LtU => result.push(WcOp::I64LtU),
            PwOp::I64LtS => result.push(WcOp::I64LtS),
            PwOp::I64LeU => result.push(WcOp::I64LeU),
            PwOp::I64LeS => result.push(WcOp::I64LeS),
            PwOp::I64GtU => result.push(WcOp::I64GtU),
            PwOp::I64GtS => result.push(WcOp::I64GtS),
            PwOp::I64GeU => result.push(WcOp::I64GeU),
            PwOp::I64GeS => result.push(WcOp::I64GeS),

            PwOp::I64ExtendUI32 => result.push(WcOp::I64ExtendI32U),
            PwOp::I64ExtendSI32 => result.push(WcOp::I64ExtendI32S),

            PwOp::I64Load(align, offset) => result.push(WcOp::I64Load(Memarg { offset: offset, align: align })),
            PwOp::I64Store(align, offset) => result.push(WcOp::I64Store(Memarg { offset: offset, align: align })),
            PwOp::I64Load8U(align, offset) => result.push(WcOp::I64Load8U(Memarg { offset: offset, align: align })),
            PwOp::I64Load8S(align, offset) => result.push(WcOp::I64Load8S(Memarg { offset: offset, align: align })),
            PwOp::I64Load16U(align, offset) => result.push(WcOp::I64Load16U(Memarg { offset: offset, align: align })),
            PwOp::I64Load16S(align, offset) => result.push(WcOp::I64Load16S(Memarg { offset: offset, align: align })),
            PwOp::I64Load32U(align, offset) => result.push(WcOp::I64Load32U(Memarg { offset: offset, align: align })),
            PwOp::I64Load32S(align, offset) => result.push(WcOp::I64Load32S(Memarg { offset: offset, align: align })),
            PwOp::I64Store8(align, offset) => result.push(WcOp::I64Store8(Memarg { offset: offset, align: align })),
            PwOp::I64Store16(align, offset) => result.push(WcOp::I64Store16(Memarg { offset: offset, align: align })),
            PwOp::I64Store32(align, offset) => result.push(WcOp::I64Store32(Memarg { offset: offset, align: align })),

            PwOp::F32Const(v) => {
                result.push(WcOp::F32Const(v));
            },
            PwOp::F64Const(v) => {
                result.push(WcOp::F64Const(v));
            },
            PwOp::F32Load(align, offset) => {
                result.push(WcOp::I32Load(Memarg { offset: offset, align: align }));
                result.push(WcOp::I32ReinterpretF32);
            },
            PwOp::F64Load(align, offset) => {
                result.push(WcOp::I64Load(Memarg { offset: offset, align: align }));
                result.push(WcOp::I64ReinterpretF64);
            },
            PwOp::F32ReinterpretI32 => result.push(WcOp::F32ReinterpretI32),
            PwOp::F64ReinterpretI64 => result.push(WcOp::F64ReinterpretI64),
            PwOp::I32ReinterpretF32 => result.push(WcOp::I32ReinterpretF32),
            PwOp::I64ReinterpretF64 => result.push(WcOp::I64ReinterpretF64),
            PwOp::I32TruncSF32 => result.push(WcOp::I32TruncSF32),
            PwOp::I32TruncUF32 => result.push(WcOp::I32TruncUF32),
            PwOp::I32TruncSF64 => result.push(WcOp::I32TruncSF64),
            PwOp::I32TruncUF64 => result.push(WcOp::I32TruncUF64),
            PwOp::I64TruncSF32 => result.push(WcOp::I64TruncSF32),
            PwOp::I64TruncUF32 => result.push(WcOp::I64TruncUF32),
            PwOp::I64TruncSF64 => result.push(WcOp::I64TruncSF64),
            PwOp::I64TruncUF64 => result.push(WcOp::I64TruncUF64),
            PwOp::F32ConvertSI32 => result.push(WcOp::F32ConvertSI32),
            PwOp::F32ConvertUI32 => result.push(WcOp::F32ConvertUI32),
            PwOp::F32ConvertSI64 => result.push(WcOp::F32ConvertSI64),
            PwOp::F32ConvertUI64 => result.push(WcOp::F32ConvertUI64),
            PwOp::F64ConvertSI32 => result.push(WcOp::F64ConvertSI32),
            PwOp::F64ConvertUI32 => result.push(WcOp::F64ConvertUI32),
            PwOp::F64ConvertSI64 => result.push(WcOp::F64ConvertSI64),
            PwOp::F64ConvertUI64 => result.push(WcOp::F64ConvertUI64),
            PwOp::F32DemoteF64 => result.push(WcOp::F32DemoteF64),
            PwOp::F64PromoteF32 => result.push(WcOp::F64PromoteF32),
            PwOp::F32Abs => result.push(WcOp::F32Abs),
            PwOp::F32Neg => result.push(WcOp::F32Neg),
            PwOp::F32Ceil => result.push(WcOp::F32Ceil),
            PwOp::F32Floor => result.push(WcOp::F32Floor),
            PwOp::F32Trunc => result.push(WcOp::F32Trunc),
            PwOp::F32Nearest => result.push(WcOp::F32Nearest),
            PwOp::F32Sqrt => result.push(WcOp::F32Sqrt),
            PwOp::F32Add => result.push(WcOp::F32Add),
            PwOp::F32Sub => result.push(WcOp::F32Sub),
            PwOp::F32Mul => result.push(WcOp::F32Mul),
            PwOp::F32Div => result.push(WcOp::F32Div),
            PwOp::F32Min => result.push(WcOp::F32Min),
            PwOp::F32Max => result.push(WcOp::F32Max),
            PwOp::F32Copysign => result.push(WcOp::F32Copysign),
            PwOp::F32Eq => result.push(WcOp::F32Eq),
            PwOp::F32Ne => result.push(WcOp::F32Ne),
            PwOp::F32Lt => result.push(WcOp::F32Lt),
            PwOp::F32Gt => result.push(WcOp::F32Gt),
            PwOp::F32Le => result.push(WcOp::F32Le),
            PwOp::F32Ge => result.push(WcOp::F32Ge),
            PwOp::F64Abs => result.push(WcOp::F64Abs),
            PwOp::F64Neg => result.push(WcOp::F64Neg),
            PwOp::F64Ceil => result.push(WcOp::F64Ceil),
            PwOp::F64Floor => result.push(WcOp::F64Floor),
            PwOp::F64Trunc => result.push(WcOp::F64Trunc),
            PwOp::F64Nearest => result.push(WcOp::F64Nearest),
            PwOp::F64Sqrt => result.push(WcOp::F64Sqrt),
            PwOp::F64Add => result.push(WcOp::F64Add),
            PwOp::F64Sub => result.push(WcOp::F64Sub),
            PwOp::F64Mul => result.push(WcOp::F64Mul),
            PwOp::F64Div => result.push(WcOp::F64Div),
            PwOp::F64Min => result.push(WcOp::F64Min),
            PwOp::F64Max => result.push(WcOp::F64Max),
            PwOp::F64Copysign => result.push(WcOp::F64Copysign),
            PwOp::F64Eq => result.push(WcOp::F64Eq),
            PwOp::F64Ne => result.push(WcOp::F64Ne),
            PwOp::F64Lt => result.push(WcOp::F64Lt),
            PwOp::F64Gt => result.push(WcOp::F64Gt),
            PwOp::F64Le => result.push(WcOp::F64Le),
            PwOp::F64Ge => result.push(WcOp::F64Ge),

            PwOp::End => {
                if let Some(label) = labels.pop() {
                    if let LabelType::If(instr_id) = label.ty {
                        let result_len = result.len() as u32;
                        if let WcOp::Jmp(ref mut t) = result[instr_id] {
                            *t = result_len;
                        } else {
                            panic!("Expecting Jmp");
                        }
                    }
                    // Make emscripten happy
                    /*
                    if label.ty == LabelType::If {
                        panic!("Expecting Else, not End");
                    }
                    */
                    label.terminate(result.as_mut_slice());
                } else {
                    expecting_seq_end = true;
                }
            },
            PwOp::If(_) => {
                let len = result.len();
                result.push(WcOp::JmpIf((len + 2) as u32));

                let mut new_label = Label::new(LabelType::If(result.len()));
                result.push(WcOp::Jmp(0xffffffff));

                labels.push(new_label);
            },
            PwOp::Else => {
                let label = labels.pop().expect("Got End outside of a block");

                {
                    match label.ty {
                        LabelType::If(instr_id) => {
                            result.push(WcOp::Jmp(0xffffffff));

                            let result_len = result.len() as u32;
                            if let WcOp::Jmp(ref mut t) = result[instr_id] {
                                *t = result_len as u32;
                            } else {
                                panic!("Expecting Jmp");
                            }
                        },
                        _ => panic!("Else must follow an If")
                    }
                }

                // defer out-branches to else blk
                let mut new_label = Label::new(LabelType::Else);
                new_label.continuations = label.continuations;
                new_label.continuations.push(Continuation::with_opcode_index(result.len() - 1)); // Jmp of the `if` branch
                labels.push(new_label);
            },
            PwOp::Block(_) => {
                labels.push(Label::new(LabelType::Block));
            },
            PwOp::Loop(_) => {
                labels.push(Label::new(LabelType::Loop(result.len())));
            },
            PwOp::Br(lb) => {
                let target = labels.iter_mut().rev().nth(lb as usize).expect("Branch target out of bound");
                target.continuations.push(Continuation::with_opcode_index(result.len()));
                result.push(WcOp::Jmp(0xffffffff));
            },
            PwOp::BrIf(lb) => {
                let target = labels.iter_mut().rev().nth(lb as usize).expect("Branch target out of bound");
                target.continuations.push(Continuation::with_opcode_index(result.len()));
                result.push(WcOp::JmpIf(0xffffffff));
            },
            PwOp::BrTable(ref targets, otherwise) => {
                let mut jmp_targets: Vec<u32> = Vec::new();

                for (i, target) in targets.iter().enumerate() {
                    let label = labels.iter_mut().rev().nth(*target as usize).expect("Branch target out of bound");
                    label.continuations.push(Continuation::brtable(result.len(), i as usize));
                    jmp_targets.push(0xffffffff);
                }

                let label = labels.iter_mut().rev().nth(otherwise as usize).expect("Branch target out of bound");
                label.continuations.push(Continuation::brtable(result.len(), targets.len()));
                result.push(WcOp::JmpTable(jmp_targets, 0xffffffff));
            },
            _ => {
                dprintln!("Warning: Generating trap for unimplemented opcode: {:?}", op);
                result.push(WcOp::NotImplemented(format!("{:?}", op)));
            }
        }
    }

    result
}
