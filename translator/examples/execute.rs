extern crate translator;

use std::fs::File;
use std::env;
use std::io::Read;
use translator::wasm_core;
use translator::wasm_core::value::Value;
use translator::wasm_core::executor::RuntimeConfig;
use translator::wasm_core::executor::VirtualMachine;

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut f = File::open(args[1].as_str()).unwrap();
    let entry_fn: &str = &args[2];

    let mut code: Vec<u8> = Vec::new();

    f.read_to_end(&mut code).unwrap();

    let module = wasm_core::module::Module::std_deserialize(code.as_slice()).unwrap();
    let mut vm: VirtualMachine = VirtualMachine::new(&module, RuntimeConfig {
        mem_default_size_pages: 64,
        mem_max_size_pages: Some(128)
    }).unwrap();

    let entry = vm.lookup_exported_func(entry_fn).unwrap();
    println!("{:?}", module.functions[entry]);

    let mut call_args: Vec<Value> = Vec::new();
    for i in 3..args.len() {
        let v: i32 = args[i].parse().unwrap();
        call_args.push(Value::I32(v));
    }

    let result = vm.execute(entry, &call_args).unwrap();
    println!("{:?}", result);
}
