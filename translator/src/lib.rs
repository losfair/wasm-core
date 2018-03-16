pub extern crate wasm_core;
extern crate parity_wasm;

use std::io::Write;
use std::collections::BTreeMap;

use parity_wasm::elements;
use wasm_core::opcode::Memarg;
use wasm_core::executor::RuntimeConfig;

pub fn translate_value_type(v: &elements::ValueType) -> wasm_core::module::ValType {
    match *v {
        elements::ValueType::I32 => wasm_core::module::ValType::I32,
        elements::ValueType::I64 => wasm_core::module::ValType::I64,
        elements::ValueType::F32 => wasm_core::module::ValType::F32,
        elements::ValueType::F64 => wasm_core::module::ValType::F64
    }
}

struct Continuation {
    opcode_index: usize
}

impl Continuation {
    fn write(&self, target: usize, opcodes: &mut [wasm_core::opcode::Opcode]) {
        use self::wasm_core::opcode::Opcode;

        let op_index = self.opcode_index;

        let new_op = match opcodes[op_index] {
            Opcode::Jmp(_) => Opcode::Jmp(target as u32),
            Opcode::JmpIf(_) => Opcode::JmpIf(target as u32),
            Opcode::JmpTable(_, _) => panic!("JmpTable is not implemented yet"),
            _ => panic!("Expecting Jmp*")
        };
        opcodes[op_index] = new_op;
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum LabelType {
    Block,
    Loop(usize), // begin
    If,
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
            LabelType::Block | LabelType::If | LabelType::Else => opcodes.len(),
            LabelType::Loop(begin) => begin
        };
        for cont in &self.continuations {
            cont.write(target, opcodes);
        }
    }
}

pub fn eval_init_expr(expr: &elements::InitExpr) -> i32 {
    let mut code = translate_opcodes(expr.code());
    code.push(wasm_core::opcode::Opcode::Return);

    let module = wasm_core::module::Module {
        types: vec! [
            wasm_core::module::Type::Func(Vec::new(), vec! [ wasm_core::module::ValType::I32 ])
        ],
        functions: vec! [
            wasm_core::module::Function {
                typeidx: 0,
                locals: Vec::new(),
                body: wasm_core::module::FunctionBody {
                    opcodes: code
                }
            }
        ],
        data_segments: vec! [],
        exports: BTreeMap::new()
    };
    let val = module.execute(RuntimeConfig {
        mem_default_size_pages: 1,
        mem_max_size_pages: Some(1)
    }, 0).unwrap().unwrap();
    val.get_i32().unwrap()
}

pub fn translate_opcodes(ops: &[elements::Opcode]) -> Vec<wasm_core::opcode::Opcode> {
    use self::elements::Opcode as PwOp;
    use self::wasm_core::opcode::Opcode as WcOp;

    let mut result: Vec<wasm_core::opcode::Opcode> = Vec::new();
    let mut labels: Vec<Label> = Vec::new();
    let mut expecting_seq_end = false;

    //eprintln!("{:?}", ops);

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

            PwOp::I32Load(offset, align) => result.push(WcOp::I32Load(Memarg { offset: offset, align: align })),
            PwOp::I32Store(offset, align) => result.push(WcOp::I32Store(Memarg { offset: offset, align: align })),
            PwOp::I32Load8U(offset, align) => result.push(WcOp::I32Load8U(Memarg { offset: offset, align: align })),
            PwOp::I32Load8S(offset, align) => result.push(WcOp::I32Load8S(Memarg { offset: offset, align: align })),
            PwOp::I32Load16U(offset, align) => result.push(WcOp::I32Load16U(Memarg { offset: offset, align: align })),
            PwOp::I32Load16S(offset, align) => result.push(WcOp::I32Load16S(Memarg { offset: offset, align: align })),
            PwOp::I32Store8(offset, align) => result.push(WcOp::I32Store8(Memarg { offset: offset, align: align })),
            PwOp::I32Store16(offset, align) => result.push(WcOp::I32Store16(Memarg { offset: offset, align: align })),

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

            PwOp::I64Load(offset, align) => result.push(WcOp::I64Load(Memarg { offset: offset, align: align })),
            PwOp::I64Store(offset, align) => result.push(WcOp::I64Store(Memarg { offset: offset, align: align })),
            PwOp::I64Load8U(offset, align) => result.push(WcOp::I64Load8U(Memarg { offset: offset, align: align })),
            PwOp::I64Load8S(offset, align) => result.push(WcOp::I64Load8S(Memarg { offset: offset, align: align })),
            PwOp::I64Load16U(offset, align) => result.push(WcOp::I64Load16U(Memarg { offset: offset, align: align })),
            PwOp::I64Load16S(offset, align) => result.push(WcOp::I64Load16S(Memarg { offset: offset, align: align })),
            PwOp::I64Load32U(offset, align) => result.push(WcOp::I64Load32U(Memarg { offset: offset, align: align })),
            PwOp::I64Load32S(offset, align) => result.push(WcOp::I64Load32S(Memarg { offset: offset, align: align })),
            PwOp::I64Store8(offset, align) => result.push(WcOp::I64Store8(Memarg { offset: offset, align: align })),
            PwOp::I64Store16(offset, align) => result.push(WcOp::I64Store16(Memarg { offset: offset, align: align })),
            PwOp::I64Store32(offset, align) => result.push(WcOp::I64Store32(Memarg { offset: offset, align: align })),

            PwOp::End => {
                if let Some(label) = labels.pop() {
                    if label.ty == LabelType::If {
                        panic!("Expecting Else, not End");
                    }
                    label.terminate(result.as_mut_slice());
                } else {
                    expecting_seq_end = true;
                }
            },
            PwOp::If(_) => {
                labels.push(Label::new(LabelType::If));
            },
            PwOp::Else => {
                let label = labels.pop().expect("Got End outside of a block");
                if label.ty != LabelType::If {
                    panic!("Else must follows an If");
                }
                // defer out-branches to else blk
                let mut new_label = Label::new(LabelType::Else);
                new_label.continuations = label.continuations;
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
                target.continuations.push(Continuation { opcode_index: result.len() });
                result.push(WcOp::Jmp(0xffffffff));
            },
            PwOp::BrIf(lb) => {
                let target = labels.iter_mut().rev().nth(lb as usize).expect("Branch target out of bound");
                target.continuations.push(Continuation { opcode_index: result.len() });
                result.push(WcOp::JmpIf(0xffffffff));
            },
            _ => {
                eprintln!("Warning: Generating trap for unimplemented opcode: {:?}", op);
                result.push(WcOp::Unreachable);
            }
        }
    }

    result
}

pub fn translate_module(code: &[u8]) {
    let module: elements::Module = parity_wasm::deserialize_buffer(code).unwrap();

    let types: Vec<wasm_core::module::Type> = if let Some(s) = module.type_section() {
        s.types().iter().map(|t| {
            let elements::Type::Function(ref ft) = *t;
            wasm_core::module::Type::Func(
                ft.params().iter().map(|v| translate_value_type(v)).collect(),
                if let Some(ret_type) = ft.return_type() {
                    vec! [ translate_value_type(&ret_type) ]
                } else {
                    vec! []
                }
            )
        }).collect()
    } else {
        Vec::new()
    };

    let code_section: &elements::CodeSection = module.code_section().unwrap();
    let function_section: &elements::FunctionSection = module.function_section().unwrap();

    let bodies: &[elements::FuncBody] = code_section.bodies();
    let fdefs: &[elements::Func] = function_section.entries();

    assert_eq!(bodies.len(), fdefs.len());

    let functions: Vec<wasm_core::module::Function> = (0..bodies.len()).map(|i| {
        let typeidx = fdefs[i].type_ref() as usize;
        let mut locals: Vec<wasm_core::module::ValType> = Vec::new();
        for lc in bodies[i].locals() {
            let t = translate_value_type(&lc.value_type());
            for _ in 0..lc.count() {
                locals.push(t);
            }
        }
        let mut opcodes = translate_opcodes(bodies[i].code().elements());
        opcodes.push(wasm_core::opcode::Opcode::Return);

        wasm_core::module::Function {
            typeidx: typeidx,
            locals: locals,
            body: wasm_core::module::FunctionBody {
                opcodes: opcodes
            }
        }
    }).collect();

    let mut data_segs: Vec<wasm_core::module::DataSegment> = Vec::new();
    if let Some(ds) = module.data_section() {
        for seg in ds.entries() {
            let offset = eval_init_expr(seg.offset()) as u32;
            eprintln!("Offset resolved: {}", offset);
            data_segs.push(wasm_core::module::DataSegment {
                offset: offset,
                data: seg.value().to_vec()
            });
        }
    } else {
        eprintln!("Warning: Data section not found");
    }

    let mut export_map: BTreeMap<String, wasm_core::module::Export> = BTreeMap::new();
    if let Some(exports) = module.export_section() {
        for entry in exports.entries() {
            use self::elements::Internal;
            eprintln!("Export: {} -> {:?}", entry.field(), entry.internal());

            let field: &str = entry.field();
            let internal: &Internal = entry.internal();

            match *internal {
                Internal::Function(id) => {
                    export_map.insert(
                        field.to_string(),
                        wasm_core::module::Export::Function(id as usize)
                    );
                },
                _ => {
                    eprintln!("Warning: Import type not supported ({:?})", internal);
                }
            }
        }
    } else {
        eprintln!("Warning: Export section not found");
    }

    let target_module = wasm_core::module::Module {
        types: types,
        functions: functions,
        data_segments: data_segs,
        exports: export_map
    };
    let serialized = target_module.std_serialize().unwrap();

    ::std::io::stdout().write(serialized.as_slice()).unwrap();
}
