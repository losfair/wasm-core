use std::cell::UnsafeCell;
use executor::NativeResolver;

pub struct Runtime {
    mem: UnsafeCell<Vec<u8>>,
    mem_max: usize,
    jit_info: Box<UnsafeCell<JitInfo>>
}

#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    mem_default: usize,
    mem_max: usize
}

#[repr(C)]
pub struct JitInfo {
    pub mem_begin: *mut u8,
    pub mem_len: usize
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        RuntimeConfig {
            mem_default: 4096 * 1024,
            mem_max: 16384 * 1024
        }
    }
}

impl Runtime {
    pub fn new(cfg: RuntimeConfig) -> Runtime {
        if cfg.mem_max < cfg.mem_default {
            panic!("mem_max < mem_default");
        }

        if cfg.mem_default == 0 {
            panic!("mem_default == 0");
        }

        let mut mem_vec: Vec<u8> = vec! [ 0; cfg.mem_default ];
        let jit_info = JitInfo {
            mem_begin: &mut mem_vec[0],
            mem_len: cfg.mem_default
        };

        Runtime {
            mem: UnsafeCell::new(mem_vec),
            mem_max: cfg.mem_max,
            jit_info: Box::new(UnsafeCell::new(jit_info))
        }
    }

    pub fn grow_memory(&self, len_inc: usize) {
        unsafe {
            let mem: &mut Vec<u8> = &mut *self.mem.get();
            if mem.len().checked_add(len_inc).unwrap() > self.mem_max {
                panic!("Memory limit exceeded");
            }
            mem.extend((0..len_inc).map(|_| 0));

            let jit_info = &mut *self.jit_info.get();
            jit_info.mem_begin = &mut mem[0];
            jit_info.mem_len = mem.len();
        }
    }

    pub fn get_jit_info(&self) -> *mut JitInfo {
        self.jit_info.get()
    }
}
