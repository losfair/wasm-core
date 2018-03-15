extern crate translator;

use std::fs::File;
use std::env;
use std::io::Read;
use translator::wasm_core;
use translator::wasm_core::executor::RuntimeConfig;

fn main() {
    let mut f = File::open(env::args().nth(1).unwrap()).unwrap();
    let mut code: Vec<u8> = Vec::new();

    f.read_to_end(&mut code).unwrap();

    let module = wasm_core::module::Module::std_deserialize(code.as_slice()).unwrap();
    module.execute(RuntimeConfig {
        mem_default_size_pages: 8,
        mem_max_size_pages: Some(32)
    }, 158).unwrap();
}
