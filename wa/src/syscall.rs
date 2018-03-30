use std::cell::RefCell;

use translator::wasm_core::executor::{
    NativeEntry,
    NativeResolver,
    GlobalStateProvider,
    ExecuteResult,
    ExecuteError
};
use translator::wasm_core::value::Value;

use stream::StreamManager;
use utils;

thread_local! {
    static STREAM_MANAGER: RefCell<StreamManager> = RefCell::new(
        StreamManager::new()
    );
}

pub enum Syscall {
    Ioctl,
    Writev,
    Readv
}

impl Syscall {
    pub fn from_id(id: u32) -> Option<Syscall> {
        match id {
            54 => Some(Syscall::Ioctl),
            145 => Some(Syscall::Readv),
            146 => Some(Syscall::Writev),
            _ => None
        }
    }
}

pub struct SyscallResolver {

}

impl SyscallResolver {
    pub fn new() -> SyscallResolver {
        SyscallResolver {}
    }
}

impl NativeResolver for SyscallResolver {
    fn resolve(&self, module: &str, field: &str) -> Option<NativeEntry> {
        if module != "env" {
            return None;
        }

        if !field.starts_with("___syscall") {
            return None;
        }

        let id_str = match field.split("syscall").nth(1) {
            Some(v) => v,
            _ => return None
        };
        let id: u32 = match id_str.parse() {
            Ok(v) => v,
            Err(_) => return None
        };

        let sc = match Syscall::from_id(id) {
            Some(v) => v,
            None => return None
        };

        Some(match sc {
            Syscall::Ioctl => wrap_sc(|rt, varargs| {
                Ok(0)
            }),
            Syscall::Readv => wrap_sc(|rt, varargs| {
                let fd = varargs.next(rt.get_memory())? as u32;

                let iov = varargs.next(rt.get_memory())? as u32;
                let iovcnt = varargs.next(rt.get_memory())? as usize;

                let mut read_cnt: i32 = 0;

                for i in 0..iovcnt {
                    let mem = rt.get_memory_mut();

                    let base = iov + (i * 8) as u32;
                    let iov_base = utils::read_mem_i32(mem, base)? as usize;
                    let iov_len = utils::read_mem_i32(mem, base + 4)? as usize;

                    let data = &mut mem[iov_base..iov_base + iov_len];
                    let ret = STREAM_MANAGER.with(|sm| {
                        sm.borrow_mut().read_stream(fd, data)
                    });

                    if ret < 0 {
                        return Ok(ret);
                    }
                    read_cnt += ret;
                }
                Ok(read_cnt)
            }),
            Syscall::Writev => wrap_sc(|rt, varargs| {
                let fd = varargs.next(rt.get_memory())? as u32;

                let iov = varargs.next(rt.get_memory())? as u32;
                let iovcnt = varargs.next(rt.get_memory())? as usize;

                let mut write_cnt: i32 = 0;

                for i in 0..iovcnt {
                    let mem = rt.get_memory();

                    let base = iov + (i * 8) as u32;
                    let iov_base = utils::read_mem_i32(mem, base)? as usize;
                    let iov_len = utils::read_mem_i32(mem, base + 4)? as usize;

                    let data = &mem[iov_base..iov_base + iov_len];
                    let ret = STREAM_MANAGER.with(|sm| {
                        sm.borrow_mut().write_stream(fd, data)
                    });

                    if ret < 0 {
                        return Ok(ret);
                    }
                    write_cnt += ret;
                }
                Ok(write_cnt)
            })
        })
    }
}

fn wrap_sc<F: Fn(&mut GlobalStateProvider, &mut VarargInfo) -> ExecuteResult<i32> + 'static>(
    f: F
) -> NativeEntry {
    Box::new(move |rt, args| {
        if args.len() != 2 {
            return Err(ExecuteError::TypeMismatch);
        }

        let varargs_ptr = args[1].get_i32()?;
        let mut va_info = VarargInfo {
            current_ptr: varargs_ptr as u32
        };

        f(rt, &mut va_info).map(|ret| Some(Value::I32(ret)))
    })
}

pub struct VarargInfo {
    current_ptr: u32
}

impl VarargInfo {
    pub fn next(&mut self, mem: &[u8]) -> ExecuteResult<i32> {
        let v = utils::read_mem_i32(mem, self.current_ptr)?;
        self.current_ptr += 4;
        Ok(v)
    }
}
