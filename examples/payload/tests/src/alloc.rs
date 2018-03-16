#[no_mangle]
pub extern "C" fn alloc_blocks(n: i32) -> *mut *mut [u8] {
    let mut blocks: Vec<*mut [u8]> = Vec::new();
    for _ in 0..n {
        let v = vec! [0; 4096];
        blocks.push(Box::into_raw(v.into_boxed_slice()));
    }

    let ret: *mut *mut [u8] = &mut blocks[0];
    ::std::mem::forget(blocks);
    ret
}
