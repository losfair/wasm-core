pub unsafe trait MemoryManager {
    fn grow(&mut self, inc_len: usize);
    fn len(&self) -> usize;
    fn get_ref(&self) -> &[u8];
    fn get_ref_mut(&mut self) -> &mut [u8];
    fn hints(&self) -> MemCodegenHints;
    fn start_address(&self) -> *mut u8;
}

#[derive(Copy, Clone, Debug)]
pub struct MemInitOptions {
    pub min: usize,
    pub max: usize
}

#[derive(Copy, Clone, Debug)]
pub struct MemCodegenHints {
    pub needs_bounds_check: bool,
    pub address_mask: usize,
    pub indirect_len_ptr: *const usize,
    pub indirect_start_address_ptr: *const *mut u8,
    pub static_start_address: Option<*mut u8>
}
