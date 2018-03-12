use alloc::Vec;

use opcode::Opcode;

pub struct Module {
    pub(crate) types: Vec<Type>,
    pub(crate) functions: Vec<Function>
}

pub enum Type {
    Func(Vec<ValType>, Vec<ValType>) // (args...) -> (ret)
}

pub struct Function {
    pub(crate) typeidx: usize,
    pub(crate) locals: Vec<ValType>,
    pub(crate) body: FunctionBody
}

pub struct FunctionBody {
    pub(crate) opcodes: Vec<Opcode>
}

pub enum ValType {
    I32,
    I64,
    F32,
    F64
}
