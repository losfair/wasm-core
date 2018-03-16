#[no_mangle]
pub extern "C" fn fib(n: i32) -> i32 {
	if n == 1 || n == 2 {
		1
	} else {
		fib(n - 1) + fib(n - 2)
	}
}
