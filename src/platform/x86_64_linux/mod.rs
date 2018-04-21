mod sigsegv;

use libc;
use std::ptr::null_mut;
use std::cell::{Cell, RefCell};
use super::generic::*;

extern "C" {
    fn __sigsetjmp(env: *mut u8, savesigs: i32) -> i32;
    fn __libc_siglongjmp(env: *mut u8, val: i32) -> !;
}

lazy_static! {
    pub static ref PAGE_SIZE: usize = {
        unsafe {
            //let mut _si: libc::siginfo_t = ::std::mem::uninitialized();
            //_si.si_addr = 1;
            libc::sysconf(libc::_SC_PAGESIZE) as usize
        }
    };

    static ref ENSURE_SIGSEGV_HANDLER: () = {
        extern "C" fn handle_sigsegv(signo: i32, siginfo: &sigsegv::SigsegvInfo, _: *mut libc::c_void) {
            MEM_FAULT_JMP_BUF.with(|buf| {
                let jmpbuf_addr: *mut u8 = {
                    let mut buf = buf.borrow_mut();
                    if buf.0 == false {
                        ::std::process::abort();
                    }
                    &mut buf.1[0] as *mut u8
                };
                MEM_FAULT_ADDR.with(|addr| {
                    addr.set(Some(siginfo.si_addr));
                });
                unsafe {
                    __libc_siglongjmp(jmpbuf_addr, 1);
                }
            });
        }

        unsafe {
            let mut sa: libc::sigaction = ::std::mem::zeroed();
            sa.sa_flags = libc::SA_SIGINFO;
            libc::sigemptyset(&mut sa.sa_mask);
            sa.sa_sigaction = handle_sigsegv as usize;

            libc::sigaction(libc::SIGSEGV, &sa, null_mut());
        }
    };
}

thread_local! {
    static MEM_FAULT_JMP_BUF: RefCell<(bool, [u8; 512])> = RefCell::new((false, [0; 512]));
    static MEM_FAULT_ADDR: Cell<Option<*mut libc::c_void>> = Cell::new(None);
}

pub struct NativeMemoryManager {
    min: usize,
    max: usize,
    mapped_len: usize,
    mem_ptr: *mut u8,
    mem_len: usize
}

impl NativeMemoryManager {
    pub fn new(opts: MemInitOptions) -> NativeMemoryManager {
        let mapped_len = opts.max.next_power_of_two();

        let mem = unsafe {
            libc::mmap(
                null_mut(),
                mapped_len,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0
            ) as *mut u8
        };
        if mem.is_null() {
            panic!("mmap failed");
        }

        let mut mm = NativeMemoryManager {
            min: opts.min,
            max: opts.max,
            mapped_len: mapped_len,
            mem_ptr: mem,
            mem_len: 0
        };

        mm.set_len(opts.min);
        mm
    }

    fn set_len(&mut self, len: usize) {
        if len > self.max {
            panic!("len > max");
        }

        let rounded_len = round_up_to_page_size(len);

        let ret = unsafe {
            libc::mprotect(
                self.mem_ptr as *mut libc::c_void,
                rounded_len,
                libc::PROT_READ | libc::PROT_WRITE
            )
        };
        if ret != 0 {
            panic!("mprotect failed");
        }

        self.mem_len = len;
    }

    pub fn protected_call<T, F: FnOnce(&mut Self) -> T>(&mut self, f: F) -> T {
        struct JmpBufGuard;

        impl Drop for JmpBufGuard {
            fn drop(&mut self) {
                MEM_FAULT_JMP_BUF.with(|buf| {
                    let mut buf = buf.borrow_mut();
                    assert_eq!(buf.0, true);
                    buf.0 = false;
                });
            }
        }

        let _ = *ENSURE_SIGSEGV_HANDLER;

        MEM_FAULT_ADDR.with(|_| {}); // Just initialize it to prevent initialization in signal handler.

        MEM_FAULT_JMP_BUF.with(move |buf| {
            let jmpbuf_addr: *mut u8 = {
                let mut buf = buf.borrow_mut();
                if buf.0 == true {
                    panic!("protected_call is not re-entrant");
                }
                buf.0 = true;

                &mut buf.1[0] as *mut u8
            };

            let _guard = JmpBufGuard;

            // Set the jmp buf.
            // The call to the target function should be immediately made after this.
            let sig = unsafe {
                __sigsetjmp(jmpbuf_addr, 1)
            };

            if sig == 0 {
                // The normal execution flow.
                f(self)
            } else {
                // sigsegv?
                let fault_addr = MEM_FAULT_ADDR.with(|addr| {
                    addr.get().unwrap_or_else(|| {
                        eprintln!("BUG: longjmp caught without fault address set");
                        ::std::process::abort();
                    })
                }) as usize;
                let expected_start = self.mem_ptr as usize;
                let expected_end = expected_start + self.mapped_len;

                if fault_addr >= expected_start && fault_addr < expected_end {
                    panic!("Memory access out of bounds");
                } else {
                    eprintln!(
                        "Fault out of protected memory: {:x}; Expecting {:x}-{:x}",
                        fault_addr,
                        expected_start,
                        expected_end
                    );
                    ::std::process::abort();
                }
            }
        })
    }
}

unsafe impl MemoryManager for NativeMemoryManager {
    fn grow(&mut self, inc_len: usize) {
        let new_len = self.mem_len + inc_len;
        self.set_len(new_len);
    }

    fn len(&self) -> usize {
        self.mem_len
    }

    fn get_ref(&self) -> &[u8] {
        unsafe { ::std::slice::from_raw_parts(
            self.mem_ptr,
            self.mem_len
        ) }
    }

    fn get_ref_mut(&mut self) -> &mut [u8] {
        unsafe { ::std::slice::from_raw_parts_mut(
            self.mem_ptr,
            self.mem_len
        ) }
    }

    fn hints(&self) -> MemCodegenHints {
        MemCodegenHints {
            needs_bounds_check: false,
            address_mask: self.mapped_len - 1
        }
    }

    fn start_address(&self) -> *mut u8 {
        self.mem_ptr
    }
}

impl Drop for NativeMemoryManager {
    fn drop(&mut self) {
        let ret = unsafe {
            libc::munmap(
                self.mem_ptr as *mut libc::c_void,
                self.mapped_len
            )
        };
        if ret != 0 {
            panic!("munmap failed");
        }
    }
}

#[inline]
fn round_up_to_page_size(size: usize) -> usize {
    let page_size: usize = *PAGE_SIZE;

    let rem = size % page_size;
    if rem > 0 {
        size - rem + page_size
    } else {
        size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mem_size_should_be_rounded_up() {
        let page_size: usize = *PAGE_SIZE;
        assert_eq!(round_up_to_page_size(123), page_size);
        assert_eq!(round_up_to_page_size(page_size), page_size);
        assert_eq!(round_up_to_page_size(page_size + 1), page_size * 2);
        assert_eq!(round_up_to_page_size(page_size * 2 - 1), page_size * 2);
        assert_eq!(round_up_to_page_size(page_size * 2), page_size * 2);
        assert_eq!(round_up_to_page_size(page_size * 2 + 1), page_size * 3);
    }

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

        let err = catch_unwind(AssertUnwindSafe(|| {
            mm.protected_call(|mm| {
                unsafe {
                    *mm.start_address().offset(1048576) = 42;
                }
            });
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
                    mm.protected_call(|mm| {
                        unsafe {
                            *mm.start_address().offset(1048576) = 42;
                        }
                    });
                }));
                assert!(err.is_err());
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }
}
