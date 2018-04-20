use libc;
use std::ptr::null_mut;
use super::generic::*;

lazy_static! {
    pub static ref PAGE_SIZE: usize = {
        unsafe {
            libc::sysconf(libc::_SC_PAGESIZE) as usize
        }
    };
}

pub struct NativeMemoryManager {
    min: usize,
    max: usize,
    mem_ptr: *mut u8,
    mem_len: usize
}

impl NativeMemoryManager {
    pub fn new(opts: MemInitOptions) -> NativeMemoryManager {
        let mem = unsafe {
            libc::mmap(
                null_mut(),
                round_up_to_page_size(opts.max),
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
            address_mask: calc_address_mask_from_len(self.max)
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
                round_up_to_page_size(self.max)
            )
        };
        if ret != 0 {
            panic!("munmap failed");
        }
    }
}

#[inline]
fn calc_address_mask_from_len(len: usize) -> usize {
    round_up_to_page_size(len) - 1
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
    fn address_mask_should_be_calculated_correctly() {
        assert_eq!(calc_address_mask_from_len(1), *PAGE_SIZE - 1);
    }
}
