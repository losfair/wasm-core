pub extern crate wasm_core;
extern crate parity_wasm;

use std::collections::BTreeMap;

use parity_wasm::elements;
use wasm_core::opcode::Memarg;
use wasm_core::executor::RuntimeConfig;
use wasm_core::resolver::NullResolver;

pub fn translate_value_type(v: &elements::ValueType) -> wasm_core::module::ValType {
    match *v {
        elements::ValueType::I32 => wasm_core::module::ValType::I32,
        elements::ValueType::I64 => wasm_core::module::ValType::I64,
        elements::ValueType::F32 => wasm_core::module::ValType::F32,
        elements::ValueType::F64 => wasm_core::module::ValType::F64
    }
}

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

pub fn eval_init_expr(
    expr: &elements::InitExpr,
    globals: &mut Vec<wasm_core::module::Global>
) -> wasm_core::value::Value {
    let mut code = translate_opcodes(expr.code());
    code.push(wasm_core::opcode::Opcode::Return);

    let module = wasm_core::module::Module {
        types: vec! [
            wasm_core::module::Type::Func(Vec::new(), vec! [ wasm_core::module::ValType::I32 ])
        ],
        functions: vec! [
            wasm_core::module::Function {
                name: None,
                typeidx: 0,
                locals: Vec::new(),
                body: wasm_core::module::FunctionBody {
                    opcodes: code
                }
            }
        ],
        data_segments: vec! [],
        exports: BTreeMap::new(),
        tables: Vec::new(),
        globals: globals.clone(),
        natives: Vec::new()
    };
    let val = module.execute(RuntimeConfig {
        mem_default_size_pages: 1,
        mem_max_size_pages: Some(1),
        resolver: Box::new(NullResolver::new())
    }, 0).unwrap().unwrap();
    val
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
            PwOp::F32Const(v) => {
                result.push(WcOp::F32Const(v));
            },
            PwOp::F64Const(v) => {
                result.push(WcOp::F64Const(v));
            },
            _ => {
                eprintln!("Warning: Generating trap for unimplemented opcode: {:?}", op);
                result.push(WcOp::NotImplemented(format!("{:?}", op)));
            }
        }
    }

    result
}

pub fn translate_module(code: &[u8]) -> Vec<u8> {
    let mut module: elements::Module = parity_wasm::deserialize_buffer(code).unwrap();
    module = match module.parse_names() {
        Ok(v) => v,
        Err((_, m)) => {
            eprintln!("Warning: Failed to parse names");
            m
        }
    };

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

    let mut emscripten_patch: bool = false;
    let mut functions: Vec<wasm_core::module::Function> = Vec::new();
    let mut natives: Vec<wasm_core::module::Native> = Vec::new();
    let mut globals: Vec<wasm_core::module::Global> = Vec::new();
    let mut tables: Vec<wasm_core::module::Table> = Vec::new();

    module.import_section().and_then(|isec| {
        for entry in isec.entries() {
            use self::elements::External;
            match *entry.external() {
                External::Function(typeidx) => {
                    let typeidx = typeidx as usize;

                    use self::wasm_core::opcode::Opcode;
                    use self::wasm_core::module::Native;

                    let native_id = natives.len();
                    natives.push(Native {
                        module: entry.module().to_string(),
                        field: entry.field().to_string(),
                        typeidx: typeidx as u32
                    });

                    let mut opcodes: Vec<Opcode> = vec! [];
                    let wasm_core::module::Type::Func(ref ty, _) = types[typeidx];

                    for i in 0..ty.len() {
                        opcodes.push(Opcode::GetLocal(i as u32));
                    }

                    opcodes.push(Opcode::NativeInvoke(native_id as u32));
                    opcodes.push(Opcode::Return);

                    functions.push(wasm_core::module::Function {
                        name: None,
                        typeidx: typeidx as u32,
                        locals: Vec::new(),
                        body: wasm_core::module::FunctionBody {
                            opcodes: opcodes
                        }
                    });
                },
                External::Global(ref gt) => {
                    let v = try_patch_emscripten_global(entry.field());
                    if let Some(v) = v {
                        eprintln!("Global {:?} patched as an Emscripten import", entry);
                        globals.push(wasm_core::module::Global {
                            value: v
                        });
                        emscripten_patch = true;
                    } else {
                        eprintln!("Warning: Generating undef for Global import: {:?}", entry);
                        globals.push(wasm_core::module::Global {
                            value: wasm_core::value::Value::default()
                        });
                    }
                },
                External::Table(ref tt) => {
                    eprintln!("Warning: Generating undef for Table import: {:?}", entry);
                    let limits: &elements::ResizableLimits = tt.limits();
                    let (min, max) = (limits.initial(), limits.maximum());

                    // Hard limit.
                    if min > 1048576 {
                        panic!("Hard limit for table size (min) exceeded");
                    }

                    let elements: Vec<Option<u32>> = vec! [ None; min as usize ];
                    tables.push(wasm_core::module::Table {
                        min: min,
                        max: max,
                        elements: elements
                    });
                },
                _ => {
                    eprintln!("Warning: Import ignored: {:?}", entry);
                    continue;
                }
            }
        }
        Some(())
    });

    {
        let to_extend = module.global_section().and_then(|gs| {
            Some(gs.entries().iter().map(|entry| {
                eprintln!("Global {:?} -> {:?}", entry, entry.init_expr());
                wasm_core::module::Global {
                    value: eval_init_expr(
                        entry.init_expr(),
                        &mut globals
                    )
                }
            }).collect())
        }).or_else(|| Some(Vec::new())).unwrap();
        globals.extend(to_extend.into_iter());
    }

    tables.extend(
        module.table_section().and_then(|ts| {
            Some(ts.entries().iter().map(|entry| {
                let limits: &elements::ResizableLimits = entry.limits();
                let (min, max) = (limits.initial(), limits.maximum());

                // Hard limit.
                if min > 1048576 {
                    panic!("Hard limit for table size (min) exceeded");
                }

                let elements: Vec<Option<u32>> = vec! [ None; min as usize ];
                wasm_core::module::Table {
                    min: min,
                    max: max,
                    elements: elements
                }
            }).collect())
        }).or_else(|| Some(Vec::new())).unwrap().into_iter()
    );

    let code_section: &elements::CodeSection = module.code_section().unwrap();
    let function_section: &elements::FunctionSection = module.function_section().unwrap();

    let bodies: &[elements::FuncBody] = code_section.bodies();
    let fdefs: &[elements::Func] = function_section.entries();

    assert_eq!(bodies.len(), fdefs.len());

    functions.extend((0..bodies.len()).map(|i| {
        //eprintln!("Function {}: {:?} {:?}", i, fdefs[i], bodies[i]);
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
            name: None,
            typeidx: typeidx as u32,
            locals: locals,
            body: wasm_core::module::FunctionBody {
                opcodes: opcodes
            }
        }
    }));

    for sec in module.sections() {
        let ns = if let elements::Section::Name(ref ns) = *sec {
            ns
        } else {
            continue;
        };
        if let elements::NameSection::Function(ref fns) = *ns {
            eprintln!("Found function name section");
            for (i, name) in fns.names() {
                functions[i as usize].name = Some(name.to_string());
            }
            break;
        }
    }

    let mut data_segs: Vec<wasm_core::module::DataSegment> = Vec::new();
    if let Some(ds) = module.data_section() {
        for seg in ds.entries() {
            let offset = eval_init_expr(seg.offset(), &mut globals).get_i32().unwrap() as u32;
            //eprintln!("Offset resolved: {} {:?}", offset, seg.value());
            data_segs.push(wasm_core::module::DataSegment {
                offset: offset,
                data: seg.value().to_vec()
            });
        }
    } else {
        eprintln!("Warning: Data section not found");
    }

    if emscripten_patch {
        eprintln!("Writing DYNAMICTOP_PTR");
        let mem_end = unsafe {
            ::std::mem::transmute::<i32, [u8; 4]>(524288)
        };
        data_segs.push(wasm_core::module::DataSegment {
            offset: 16,
            data: mem_end.to_vec()
        });
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
                        wasm_core::module::Export::Function(id as u32)
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

    if let Some(elems) = module.elements_section() {
        for entry in elems.entries() {
            let offset = eval_init_expr(entry.offset(), &mut globals).get_i32().unwrap() as u32 as usize;
            let members = entry.members();
            let end = offset + members.len();
            let tt = &mut tables[entry.index() as usize];

            if end > tt.elements.len() {
                if let Some(max) = tt.max {
                    if end > max as usize {
                        panic!("Max table length exceeded");
                    }
                }
                // Hard limit.
                if end > 1048576 {
                    panic!("Hard limit for table size (max) exceeded");
                }
                while end > tt.elements.len() {
                    tt.elements.push(None);
                }
            }

            for i in 0..members.len() {
                tt.elements[offset + i] = Some(members[i]);
            }

            eprintln!("Elements written to table: {}, {}", offset, members.len());
        }
        eprintln!("{} elements added to table", elems.entries().len());
    } else {
        eprintln!("Warning: Elements section not found");
    }

    let target_module = wasm_core::module::Module {
        types: types,
        functions: functions,
        data_segments: data_segs,
        exports: export_map,
        tables: tables,
        globals: globals,
        natives: natives
    };
    let serialized = target_module.std_serialize().unwrap();

    serialized
}

fn try_patch_emscripten_global(field_name: &str) -> Option<wasm_core::value::Value> {
    use self::wasm_core::value::Value;

    match field_name {
        "memoryBase" => Some(Value::I32(0)),
        "tableBase" => Some(Value::I32(0)),
        "DYNAMICTOP_PTR" => Some(Value::I32(16)), // needs mem init
        "tempDoublePtr" => Some(Value::I32(0)),
        "STACK_MAX" => Some(Value::I32(65536)),
        _ => None
    }
}
