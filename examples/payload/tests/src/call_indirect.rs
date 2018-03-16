type Callable = extern "C" fn () -> i32;

static mut CLOSURE: Option<*mut Fn(i64) -> i32> = None;

#[no_mangle]
pub extern "C" fn call(target: Callable) -> i32 {
    target()
}

#[no_mangle]
pub extern "C" fn produce_value() -> i32 {
    42
}

#[no_mangle]
pub extern "C" fn get_addr() -> Callable {
    produce_value
}

#[inline(never)]
fn produce_closure() -> Box<Fn(i64) -> i32> {
    let v: i32 = 0;
    Box::new(|a| {
        //panic!();
        produce_value() + a as i32
    })
}

#[no_mangle]
pub extern "C" fn run() {
    let f: &Fn(i64) -> i32 = unsafe {
        if CLOSURE.is_none() {
            CLOSURE = Some(Box::into_raw(
                produce_closure()
            ));
        }
        &*CLOSURE.unwrap()
    };
    let result = f(99);
    assert_eq!(result, 42 + 99);
}
