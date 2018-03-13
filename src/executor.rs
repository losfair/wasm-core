use module::{Module, Function};
use core::result::Result;
use alloc::{Vec, String};
use opcode::Opcode;
use int_ops;
use value::Value;

#[derive(Debug)]
pub enum ExecuteError {
    Custom(String),
    OperandStackUnderflow,
    NotImplemented,
    FunctionIndexOutOfBound,
    OpcodeIndexOutOfBound,
    FrameIndexOutOfBound,
    LocalIndexOutOfBound,
    UnreachableExecuted,
    TypeMismatch
}

impl ::core::fmt::Display for ExecuteError {
    fn fmt(&self, f: &mut ::core::fmt::Formatter) -> Result<(), ::core::fmt::Error> {
        <Self as ::core::fmt::Debug>::fmt(self, f)
    }
}

pub type ExecuteResult<T> = Result<T, ExecuteError>;

pub struct Frame {
    func_id: usize,
    ip: Option<usize>,
    operands: Vec<Value>,
    locals: Vec<Value>
}

impl Frame {
    pub fn setup(func_id: usize, func: &Function) -> Frame {
        Frame {
            func_id: func_id,
            ip: None,
            operands: Vec::new(),
            locals: vec![Value::default(); func.locals.len()]
        }
    }

    pub fn top_operand(&self) -> ExecuteResult<Value> {
        match self.operands.last() {
            Some(v) => Ok(*v),
            None => Err(ExecuteError::OperandStackUnderflow)
        }
    }

    pub fn pop_operand(&mut self) -> ExecuteResult<Value> {
        match self.operands.pop() {
            Some(v) => Ok(v),
            None => Err(ExecuteError::OperandStackUnderflow)
        }
    }

    pub fn push_operand(&mut self, operand: Value) {
        self.operands.push(operand);
    }

    pub fn set_local(&mut self, idx: u32, val: Value) -> ExecuteResult<()> {
        let idx = idx as usize;

        if idx >= self.locals.len() {
            Err(ExecuteError::LocalIndexOutOfBound)
        } else {
            self.locals[idx] = val;
            Ok(())
        }
    }

    pub fn get_local(&mut self, idx: u32) -> ExecuteResult<Value> {
        let idx = idx as usize;

        if idx >= self.locals.len() {
            Err(ExecuteError::LocalIndexOutOfBound)
        } else {
            Ok(self.locals[idx])
        }
    }
}

impl Module {
    pub fn execute(&self, initial_func: usize) -> ExecuteResult<()> {
        let mut frames: Vec<Frame> = Vec::new();
        let mut ip: usize = 0;

        let mut current_func: &Function = &self.functions[initial_func];
        frames.push(Frame::setup(initial_func, current_func));

        loop {
            let frame: &mut Frame = match frames.last_mut() {
                Some(v) => v,
                None => return Err(ExecuteError::FrameIndexOutOfBound)
            };

            // Fetch the current instruction and move to the next one.
            if ip >= current_func.body.opcodes.len() {
                return Err(ExecuteError::OpcodeIndexOutOfBound);
            }
            let op = &current_func.body.opcodes[ip];
            ip += 1;

            match *op {
                Opcode::Drop => {
                    frame.pop_operand()?;
                },
                Opcode::Call(idx) => {
                    // "Push" IP so that we can restore it after the call is done.
                    frame.ip = Some(ip);

                    // Reset IP.
                    ip = 0;

                    let idx = idx as usize;
                    if idx >= self.functions.len() {
                        return Err(ExecuteError::FunctionIndexOutOfBound);
                    }
                    current_func = &self.functions[idx];

                    // NLL is required for this to work.
                    frames.push(Frame::setup(idx, current_func));
                },
                Opcode::Return => {
                    // Pop the current frame.
                    frames.pop().unwrap();

                    // Restore IP.
                    let frame: &mut Frame = match frames.last_mut() {
                        Some(v) => v,
                        None => return Ok(()) // We've reached the end of the entry function
                    };
                    ip = frame.ip.take().unwrap();

                    current_func = &self.functions[frame.func_id];
                },
                Opcode::Nop => {},
                Opcode::Jmp(target) => {
                    ip = target as usize;
                },
                Opcode::JmpIf(target) => {
                    let v = frame.pop_operand()?.get_i32()?;
                    if v != 0 {
                        ip = target as usize;
                    }
                },
                Opcode::JmpTable(ref table, otherwise) => {
                    let v = frame.pop_operand()?.get_i32()? as usize;
                    if v < table.len() {
                        ip = table[v] as usize;
                    } else {
                        ip = otherwise as usize;
                    }
                },
                Opcode::SetLocal(idx) => {
                    let v = frame.pop_operand()?;
                    frame.set_local(idx, v)?;
                },
                Opcode::GetLocal(idx) => {
                    let v = frame.get_local(idx)?;
                    frame.push_operand(v);
                },
                Opcode::TeeLocal(idx) => {
                    let v = frame.top_operand()?;
                    frame.set_local(idx, v)?;
                },
                Opcode::Unreachable => {
                    return Err(ExecuteError::UnreachableExecuted);
                },
                Opcode::I32Const(v) => {
                    frame.push_operand(Value::I32(v));
                },
                Opcode::I32Clz => {
                    let v = frame.pop_operand()?;
                    frame.push_operand(int_ops::i32_clz(v.get_i32()?));
                },
                Opcode::I32Ctz => {
                    let v = frame.pop_operand()?;
                    frame.push_operand(int_ops::i32_ctz(v.get_i32()?));
                },
                Opcode::I32Popcnt => {
                    let v = frame.pop_operand()?;
                    frame.push_operand(int_ops::i32_popcnt(v.get_i32()?));
                },
                Opcode::I32Add => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i32_add(c1.get_i32()?, c2.get_i32()?));
                },
                Opcode::I32Sub => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i32_sub(c1.get_i32()?, c2.get_i32()?));
                },
                Opcode::I32Mul => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i32_mul(c1.get_i32()?, c2.get_i32()?));
                },
                Opcode::I32DivU => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i32_div_u(c1.get_i32()?, c2.get_i32()?));
                },
                Opcode::I32DivS => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i32_div_s(c1.get_i32()?, c2.get_i32()?));
                },
                Opcode::I32RemU => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i32_rem_u(c1.get_i32()?, c2.get_i32()?));
                },
                Opcode::I32RemS => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i32_rem_s(c1.get_i32()?, c2.get_i32()?));
                },
                Opcode::I32And => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i32_and(c1.get_i32()?, c2.get_i32()?));
                },
                Opcode::I32Or => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i32_or(c1.get_i32()?, c2.get_i32()?));
                },
                Opcode::I32Shl => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i32_shl(c1.get_i32()?, c2.get_i32()?));
                },
                Opcode::I32ShrU => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i32_shr_u(c1.get_i32()?, c2.get_i32()?));
                },
                Opcode::I32ShrS => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i32_shr_s(c1.get_i32()?, c2.get_i32()?));
                },
                Opcode::I32Rotl => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i32_rotl(c1.get_i32()?, c2.get_i32()?));
                },
                Opcode::I32Rotr => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i32_rotr(c1.get_i32()?, c2.get_i32()?));
                },
                Opcode::I32Eqz => {
                    let v = frame.pop_operand()?;
                    frame.push_operand(int_ops::i32_eqz(v.get_i32()?));
                },
                Opcode::I32Eq => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i32_eq(c1.get_i32()?, c2.get_i32()?));
                },
                Opcode::I32Ne => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i32_ne(c1.get_i32()?, c2.get_i32()?));
                },
                Opcode::I32LtU => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i32_lt_u(c1.get_i32()?, c2.get_i32()?));
                },
                Opcode::I32LtS => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i32_lt_s(c1.get_i32()?, c2.get_i32()?));
                },
                Opcode::I32LeU => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i32_le_u(c1.get_i32()?, c2.get_i32()?));
                },
                Opcode::I32LeS => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i32_le_s(c1.get_i32()?, c2.get_i32()?));
                },
                Opcode::I32GtU => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i32_gt_u(c1.get_i32()?, c2.get_i32()?));
                },
                Opcode::I32GtS => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i32_gt_s(c1.get_i32()?, c2.get_i32()?));
                },
                Opcode::I32GeU => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i32_ge_u(c1.get_i32()?, c2.get_i32()?));
                },
                Opcode::I32GeS => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i32_ge_s(c1.get_i32()?, c2.get_i32()?));
                },
                Opcode::I32WrapI64 => {
                    let v = frame.pop_operand()?;
                    frame.push_operand(int_ops::i32_wrap_i64(v.get_i32()?));
                },
                Opcode::I64Const(v) => {
                    frame.push_operand(Value::I64(v));
                },
                Opcode::I64Clz => {
                    let v = frame.pop_operand()?;
                    frame.push_operand(int_ops::i64_clz(v.get_i64()?));
                },
                Opcode::I64Ctz => {
                    let v = frame.pop_operand()?;
                    frame.push_operand(int_ops::i64_ctz(v.get_i64()?));
                },
                Opcode::I64Popcnt => {
                    let v = frame.pop_operand()?;
                    frame.push_operand(int_ops::i64_popcnt(v.get_i64()?));
                },
                Opcode::I64Add => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i64_add(c1.get_i64()?, c2.get_i64()?));
                },
                Opcode::I64Sub => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i64_sub(c1.get_i64()?, c2.get_i64()?));
                },
                Opcode::I64Mul => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i64_mul(c1.get_i64()?, c2.get_i64()?));
                },
                Opcode::I64DivU => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i64_div_u(c1.get_i64()?, c2.get_i64()?));
                },
                Opcode::I64DivS => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i64_div_s(c1.get_i64()?, c2.get_i64()?));
                },
                Opcode::I64RemU => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i64_rem_u(c1.get_i64()?, c2.get_i64()?));
                },
                Opcode::I64RemS => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i64_rem_s(c1.get_i64()?, c2.get_i64()?));
                },
                Opcode::I64And => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i64_and(c1.get_i64()?, c2.get_i64()?));
                },
                Opcode::I64Or => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i64_or(c1.get_i64()?, c2.get_i64()?));
                },
                Opcode::I64Shl => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i64_shl(c1.get_i64()?, c2.get_i64()?));
                },
                Opcode::I64ShrU => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i64_shr_u(c1.get_i64()?, c2.get_i64()?));
                },
                Opcode::I64ShrS => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i64_shr_s(c1.get_i64()?, c2.get_i64()?));
                },
                Opcode::I64Rotl => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i64_rotl(c1.get_i64()?, c2.get_i64()?));
                },
                Opcode::I64Rotr => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i64_rotr(c1.get_i64()?, c2.get_i64()?));
                },
                Opcode::I64Eqz => {
                    let v = frame.pop_operand()?;
                    frame.push_operand(int_ops::i64_eqz(v.get_i64()?));
                },
                Opcode::I64Eq => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i64_eq(c1.get_i64()?, c2.get_i64()?));
                },
                Opcode::I64Ne => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i64_ne(c1.get_i64()?, c2.get_i64()?));
                },
                Opcode::I64LtU => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i64_lt_u(c1.get_i64()?, c2.get_i64()?));
                },
                Opcode::I64LtS => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i64_lt_s(c1.get_i64()?, c2.get_i64()?));
                },
                Opcode::I64LeU => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i64_le_u(c1.get_i64()?, c2.get_i64()?));
                },
                Opcode::I64LeS => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i64_le_s(c1.get_i64()?, c2.get_i64()?));
                },
                Opcode::I64GtU => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i64_gt_u(c1.get_i64()?, c2.get_i64()?));
                },
                Opcode::I64GtS => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i64_gt_s(c1.get_i64()?, c2.get_i64()?));
                },
                Opcode::I64GeU => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i64_ge_u(c1.get_i64()?, c2.get_i64()?));
                },
                Opcode::I64GeS => {
                    let c2 = frame.pop_operand()?;
                    let c1 = frame.pop_operand()?;
                    frame.push_operand(int_ops::i64_ge_s(c1.get_i64()?, c2.get_i64()?));
                },
                Opcode::I64ExtendI32U => {
                    let v = frame.pop_operand()?;
                    frame.push_operand(int_ops::i64_extend_i32_u(v.get_i64()?));
                },
                Opcode::I64ExtendI32S => {
                    let v = frame.pop_operand()?;
                    frame.push_operand(int_ops::i64_extend_i32_s(v.get_i64()?));
                },
                _ => return Err(ExecuteError::NotImplemented)
            }
        }

        Ok(())
    }
}
