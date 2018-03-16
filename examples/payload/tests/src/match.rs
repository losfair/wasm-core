#[no_mangle]
//#[inline(never)]
pub extern "C" fn do_match(v: i32) -> i32 {
     match v {
        3 => do_match(5),
        6 => do_match(10),
        10 => 42,
        5 => 99,
        _ => unreachable!()
    }
}

#[no_mangle]
pub extern "C" fn run() {
    assert_eq!(do_match(111), 0);
    assert_eq!(do_match(6), 42);
    assert_eq!(do_match(3), 99);
    assert_eq!(do_match(10), 42);
    assert_eq!(do_match(5), 99);
}
