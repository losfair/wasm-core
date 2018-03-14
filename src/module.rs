use alloc::Vec;

use opcode::Opcode;

pub struct Module {
    pub types: Vec<Type>,
    pub functions: Vec<Function>
}

pub enum Type {
    Func(Vec<ValType>, Vec<ValType>) // (args...) -> (ret)
}

pub struct Function {
    pub typeidx: usize,
    pub locals: Vec<ValType>,
    pub body: FunctionBody
}

pub struct FunctionBody {
    pub opcodes: Vec<Opcode>
}

#[derive(Copy, Clone, Debug)]
pub enum ValType {
    I32,
    I64,
    F32,
    F64
}
