pub extern crate wasm_core;
extern crate parity_wasm;
extern crate serde;
#[macro_use]
extern crate serde_derive;

#[macro_use]
mod debug_print;

pub mod config;
pub mod optrans;

use std::collections::BTreeMap;

use parity_wasm::elements;
use wasm_core::executor::RuntimeConfig;
use wasm_core::resolver::NullResolver;
use config::ModuleConfig;

pub fn translate_value_type(v: &elements::ValueType) -> wasm_core::module::ValType {
    match *v {
        elements::ValueType::I32 => wasm_core::module::ValType::I32,
        elements::ValueType::I64 => wasm_core::module::ValType::I64,
        elements::ValueType::F32 => wasm_core::module::ValType::F32,
        elements::ValueType::F64 => wasm_core::module::ValType::F64
    }
}

pub fn eval_init_expr(
    expr: &elements::InitExpr,
    globals: &mut Vec<wasm_core::module::Global>
) -> wasm_core::value::Value {
    let mut code = optrans::translate_opcodes(expr.code());
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
        natives: Vec::new(),
        start_function: None
    };
    let val = module.execute(RuntimeConfig {
        mem_default_size_pages: 1,
        mem_max_size_pages: Some(1),
        resolver: Box::new(NullResolver::new())
    }, 0).unwrap().unwrap();
    val
}



pub fn translate_module_raw(
    code: &[u8],
    config: ModuleConfig
) -> wasm_core::module::Module {
    let mut module: elements::Module = parity_wasm::deserialize_buffer(code).unwrap();
    module = match module.parse_names() {
        Ok(v) => v,
        Err((_, m)) => {
            dprintln!("Warning: Failed to parse names");
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

    let mut export_map: BTreeMap<String, wasm_core::module::Export> = BTreeMap::new();
    if let Some(exports) = module.export_section() {
        for entry in exports.entries() {
            use self::elements::Internal;
            dprintln!("Export: {} -> {:?}", entry.field(), entry.internal());

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
                    dprintln!("Warning: Internal type not supported ({:?})", internal);
                }
            }
        }
    } else {
        dprintln!("Warning: Export section not found");
    }

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

                    dprintln!("Importing function: {:?} type: {:?}", entry, types[typeidx]);

                    let patched = if config.emscripten.unwrap_or(false) && entry.module() == "env" {
                        let f: Option<wasm_core::module::Function> = try_patch_emscripten_func_import(
                            entry.field(),
                            typeidx,
                            &types[typeidx],
                            &export_map
                        );
                        if let Some(f) = f {
                            functions.push(f);
                            dprintln!("Patch applied");
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    };

                    if !patched {
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
                    }
                },
                External::Global(ref gt) => {
                    let patched = if config.emscripten.unwrap_or(false) {
                        let v = try_patch_emscripten_global(entry.field());
                        if let Some(v) = v {
                            dprintln!("Global {:?} patched as an Emscripten import", entry);
                            globals.push(wasm_core::module::Global {
                                value: v
                            });
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    };
                    if !patched {
                        dprintln!("Warning: Generating undef for Global import: {:?}", entry);
                        globals.push(wasm_core::module::Global {
                            value: wasm_core::value::Value::default()
                        });
                    }
                },
                External::Table(ref tt) => {
                    dprintln!("Warning: Generating undef for Table import: {:?}", entry);
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
                    dprintln!("Warning: Import ignored: {:?}", entry);
                    continue;
                }
            }
        }
        Some(())
    });

    {
        let to_extend = module.global_section().and_then(|gs| {
            Some(gs.entries().iter().map(|entry| {
                dprintln!("Global {:?} -> {:?}", entry, entry.init_expr());
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
        //dprintln!("Function {}: {:?} {:?}", i, fdefs[i], bodies[i]);
        let typeidx = fdefs[i].type_ref() as usize;
        let mut locals: Vec<wasm_core::module::ValType> = Vec::new();
        for lc in bodies[i].locals() {
            let t = translate_value_type(&lc.value_type());
            for _ in 0..lc.count() {
                locals.push(t);
            }
        }
        let mut opcodes = optrans::translate_opcodes(bodies[i].code().elements());
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

    let start_func_id = module.start_section();

    for sec in module.sections() {
        let ns = if let elements::Section::Name(ref ns) = *sec {
            ns
        } else {
            continue;
        };
        if let elements::NameSection::Function(ref fns) = *ns {
            dprintln!("Found function name section");
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
            //dprintln!("Offset resolved: {} {:?}", offset, seg.value());
            data_segs.push(wasm_core::module::DataSegment {
                offset: offset,
                data: seg.value().to_vec()
            });
        }
    } else {
        dprintln!("Warning: Data section not found");
    }

    if config.emscripten.unwrap_or(false) {
        dprintln!("Writing DYNAMICTOP_PTR");
        let mem_end = unsafe {
            ::std::mem::transmute::<i32, [u8; 4]>(524288)
        };
        data_segs.push(wasm_core::module::DataSegment {
            offset: 16,
            data: mem_end.to_vec()
        });
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

            dprintln!("Elements written to table: {}, {}", offset, members.len());
        }
        dprintln!("{} elements added to table", elems.entries().len());
    } else {
        dprintln!("Warning: Elements section not found");
    }

    dprintln!("Start: {:?}", start_func_id);

    let target_module = wasm_core::module::Module {
        types: types,
        functions: functions,
        data_segments: data_segs,
        exports: export_map,
        tables: tables,
        globals: globals,
        natives: natives,
        start_function: start_func_id
    };

    target_module
}

pub fn translate_module(
    code: &[u8],
    config: ModuleConfig
) -> Vec<u8> {
    let serialized = translate_module_raw(code, config)
        .std_serialize().unwrap();
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

fn try_patch_emscripten_func_import(
    field_name: &str,
    typeidx: usize,
    ty: &wasm_core::module::Type,
    export_map: &BTreeMap<String, wasm_core::module::Export>
) -> Option<wasm_core::module::Function> {
    use self::wasm_core::module::{Function, FunctionBody};
    use self::wasm_core::opcode::Opcode;

    if field_name.starts_with("invoke_") {
        let parts: Vec<&str> = field_name.split("_").collect();
        if parts.len() == 2 {
            return Some(
                gen_invoke_n_to_dyncall_n(
                    parts[1],
                    typeidx,
                    ty,
                    export_map
                )
            );
        }
    }

    match field_name {
        "_emscripten_memcpy_big" => Some(
            Function {
                name: Some("_emscripten_memcpy_big".into()),
                typeidx: typeidx as u32,
                locals: Vec::new(),
                body: FunctionBody {
                    opcodes: vec! [
                        Opcode::GetLocal(0), // dest
                        Opcode::GetLocal(1), // src
                        Opcode::GetLocal(2), // n_bytes
                        Opcode::Memcpy,
                        Opcode::GetLocal(0),
                        Opcode::Return
                    ]
                }
            }
        ),
        _ => None
    }
}

fn gen_invoke_n_to_dyncall_n(
    suffix: &str,
    typeidx: usize,
    ty: &wasm_core::module::Type,
    export_map: &BTreeMap<String, wasm_core::module::Export>
) -> wasm_core::module::Function {
    use self::wasm_core::module::{Function, FunctionBody, Type, Export};
    use self::wasm_core::opcode::Opcode;

    let invoke_name = format!("invoke_{}", suffix);
    let dyncall_name = format!("dynCall_{}", suffix);

    let Export::Function(fn_idx) = *export_map.get(dyncall_name.as_str()).unwrap();

    let mut opcodes: Vec<Opcode> = Vec::new();

    let Type::Func(ref ft_args, ref ft_ret) = *ty;

    for i in 0..ft_args.len() {
        opcodes.push(Opcode::GetLocal(i as u32));
    }
    opcodes.push(Opcode::Call(fn_idx));
    opcodes.push(Opcode::Return);

    Function {
        name: Some(invoke_name.clone()),
        typeidx: typeidx as u32,
        locals: Vec::new(),
        body: FunctionBody {
            opcodes: opcodes
        }
    }
}
