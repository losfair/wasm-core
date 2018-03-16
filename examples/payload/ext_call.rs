extern "C" {
    fn __wcore_ping(v: i32) -> i32;
}

#[no_mangle]
pub extern "C" fn call() -> i32 {
    unsafe {
        __wcore_ping(42)
    }
}
