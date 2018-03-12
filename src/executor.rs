use module::{Module, Function};
use core::result::Result;
use alloc::{Vec, String};
use opcode::Opcode;

#[derive(Debug)]
pub enum ExecuteError {
    Custom(String),
    OperandStackUnderflow,
    NotImplemented,
    FunctionIndexOutOfBound,
    OpcodeIndexOutOfBound,
    FrameIndexOutOfBound,
    LocalIndexOutOfBound
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
    operands: Vec<i64>,
    locals: Vec<i64>
}

impl Frame {
    pub fn setup(func_id: usize, func: &Function) -> Frame {
        Frame {
            func_id: func_id,
            ip: None,
            operands: Vec::new(),
            locals: vec![0; func.locals.len()]
        }
    }

    pub fn top_operand(&self) -> ExecuteResult<i64> {
        match self.operands.last() {
            Some(v) => Ok(*v),
            None => Err(ExecuteError::OperandStackUnderflow)
        }
    }

    pub fn pop_operand(&mut self) -> ExecuteResult<i64> {
        match self.operands.pop() {
            Some(v) => Ok(v),
            None => Err(ExecuteError::OperandStackUnderflow)
        }
    }

    pub fn push_operand(&mut self, operand: i64) {
        self.operands.push(operand);
    }

    pub fn set_local(&mut self, idx: u32, val: i64) -> ExecuteResult<()> {
        let idx = idx as usize;

        if idx >= self.locals.len() {
            Err(ExecuteError::LocalIndexOutOfBound)
        } else {
            self.locals[idx] = val;
            Ok(())
        }
    }

    pub fn get_local(&mut self, idx: u32) -> ExecuteResult<i64> {
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
                    let v = frame.pop_operand()?;
                    if v != 0 {
                        ip = target as usize;
                    }
                },
                Opcode::JmpTable(ref table, otherwise) => {
                    let v = frame.pop_operand()? as usize;
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
                _ => return Err(ExecuteError::NotImplemented)
            }
        }

        Ok(())
    }
}
