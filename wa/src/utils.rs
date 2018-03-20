use translator::wasm_core::executor::{
    ExecuteResult,
    ExecuteError
};

pub fn read_mem_i32(mem: &[u8], ptr: u32) -> ExecuteResult<i32> {
    let ptr = ptr as usize;
    if ptr + 4 > mem.len() {
        return Err(ExecuteError::AddrOutOfBound(
            ptr as u32,
            4
        ));
    }

    let v: [u8; 4] = [
        mem[ptr],
        mem[ptr + 1],
        mem[ptr + 2],
        mem[ptr + 3]
    ];
    Ok(unsafe {
        ::std::mem::transmute::<[u8; 4], i32>(v)
    })
}
