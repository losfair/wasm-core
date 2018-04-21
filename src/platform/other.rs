use std::ptr::null_mut;
use std::cell::{Cell, RefCell};
use super::generic::*;

pub struct NativeMemoryManager {
    min: usize,
    max: usize,
    mem: Vec<u8>,
    mem_start: *mut u8,
    mem_len: usize
}

impl NativeMemoryManager {
    pub fn new(opts: MemInitOptions) -> NativeMemoryManager {
        let mut mem: Vec<u8> = vec! [ 0; opts.min ];
        let mem_start = &mut mem[0] as *mut u8;

        NativeMemoryManager {
            min: opts.min,
            max: opts.max,
            mem: mem,
            mem_start: mem_start,
            mem_len: opts.min
        }
    }

    pub fn protected_call<T, F: FnOnce(&mut Self) -> T>(&mut self, f: F) -> T {
        f(self)
    }
}

unsafe impl MemoryManager for NativeMemoryManager {
    fn grow(&mut self, len_inc: usize) {
        if self.mem.len().checked_add(len_inc).unwrap() > self.max {
            panic!("Memory limit exceeded");
        }
        self.mem.extend((0..len_inc).map(|_| 0));

        let mem_start = &mut self.mem[0] as *mut u8;
        self.mem_start = mem_start;
        self.mem_len = self.mem.len();
    }

    fn len(&self) -> usize {
        self.mem_len
    }

    fn get_ref(&self) -> &[u8] {
        &self.mem
    }

    fn get_ref_mut(&mut self) -> &mut [u8] {
        &mut self.mem
    }

    fn hints(&self) -> MemCodegenHints {
        MemCodegenHints {
            needs_bounds_check: true,
            address_mask: self.max.next_power_of_two() - 1,
            indirect_len_ptr: &self.mem_len,
            indirect_start_address_ptr: &self.mem_start,
            static_start_address: None
        }
    }

    fn start_address(&self) -> *mut u8 {
        self.mem_start
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_native_mm() {
        use std::panic::{catch_unwind, AssertUnwindSafe};

        let mut mm = NativeMemoryManager::new(MemInitOptions {
            min: 100000,
            max: 3221225472
        });

        let err = catch_unwind(AssertUnwindSafe(|| {
            let mem = mm.get_ref_mut();

            mem[0] = 1;
            mem[100] = 2;
            mem[99999] = 3;
        }));
        assert!(err.is_ok());

        let err = catch_unwind(AssertUnwindSafe(|| {
            let mem = mm.get_ref_mut();
            mem[100000] = 3;
        }));
        assert!(err.is_err());
    }

    #[test]
    fn test_native_mm_concurrent() {
        use std::panic::{catch_unwind, AssertUnwindSafe};

        let mut handles = Vec::new();

        for _ in 0..10000 {
            handles.push(::std::thread::spawn(|| {
                let mut mm = NativeMemoryManager::new(MemInitOptions {
                    min: 100000,
                    max: 3221225472
                });
                let err = catch_unwind(AssertUnwindSafe(|| {
                    let mem = mm.get_ref_mut();
                    mem[100000] = 42;
                }));
                assert!(err.is_err());
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }
}
