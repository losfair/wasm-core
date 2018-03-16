#[no_mangle]
pub extern "C" fn run() {
    let vec: Vec<i32> = vec! [42; 10000];
    let mut result: i32 = 0;
    for v in vec {
        result += v;
    }
    assert_eq!(result, 42000);
}

fn main() {
    run();
}
