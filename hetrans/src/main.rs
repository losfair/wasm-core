extern crate wasm_core;

use std::fs::File;
use std::env;
use std::io::Read;
use std::io::Write;

use wasm_core::value::Value;
use wasm_core::module::*;
use wasm_core::trans::config::ModuleConfig;
use wasm_core::hetrans::translate_module;
use wasm_core::hetrans::NullMapNativeInvoke;

fn main() {
    let mut args = env::args();
    args.next().unwrap();

    let path = args.next().expect("Path required");
    let entry_fn_name = args.next().expect("Entry function required");
    let mut f = File::open(&path).unwrap();
    let mut code: Vec<u8> = Vec::new();

    let cfg: ModuleConfig = ModuleConfig::default();

    f.read_to_end(&mut code).unwrap();
    let module = wasm_core::trans::translate_module_raw(code.as_slice(), cfg);
    let entry_fn = module.lookup_exported_func(&entry_fn_name).expect("Entry function not found");

    let result = translate_module(&module, entry_fn, &mut NullMapNativeInvoke);
    eprintln!("{:?}", module.functions[entry_fn].body.opcodes);
    ::std::io::stdout().write_all(&result).unwrap();
}
