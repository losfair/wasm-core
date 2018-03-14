extern crate translator;

use std::fs::File;
use std::env;
use std::io::Read;

fn main() {
    let mut f = File::open(env::args().nth(1).unwrap()).unwrap();
    let mut code: Vec<u8> = Vec::new();

    f.read_to_end(&mut code).unwrap();

    translator::translate_module(code.as_slice());
}
