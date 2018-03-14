extern crate wasm_core;
extern crate parity_wasm;

use parity_wasm::elements;

pub fn translate_value_type(v: &elements::ValueType) -> wasm_core::module::ValType {
    match *v {
        elements::ValueType::I32 => wasm_core::module::ValType::I32,
        elements::ValueType::I64 => wasm_core::module::ValType::I64,
        elements::ValueType::F32 => wasm_core::module::ValType::F32,
        elements::ValueType::F64 => wasm_core::module::ValType::F64
    }
}

pub fn translate_opcodes(op: &[elements::Opcode]) -> Vec<wasm_core::opcode::Opcode> {
    unimplemented!()
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
        let opcodes = translate_opcodes(bodies[i].code().elements());

        wasm_core::module::Function {
            typeidx: typeidx,
            locals: locals,
            body: wasm_core::module::FunctionBody {
                opcodes: opcodes
            }
        }
    }).collect();
}
