extern crate translator;
extern crate serde_json;

use std::fs::File;
use std::env;
use std::io::Read;
use std::io::Write;

use translator::wasm_core::value::Value;
use translator::wasm_core::module::Module;
use translator::wasm_core::executor::{VirtualMachine, RuntimeConfig, NativeResolver, NativeEntry, ExecuteError};

struct DefaultResolver {

}

impl NativeResolver for DefaultResolver {
    fn resolve(&self, module: &str, field: &str) -> Option<NativeEntry> {
        println!("Resolve: {} {}", module, field);
        if module != "env" {
            return None;
        }

        match field {
            "__wcore_ping" => Some(Box::new(|_, args| {
                if args.len() == 0 {
                    Err(ExecuteError::TypeMismatch)
                } else {
                    Ok(Some(args[0]))
                }
            })),
            _ => None
        }
    }
}

fn main() {
    let mut args = env::args();
    args.next().unwrap();

    let mode: String = args.next().expect("Mode required");
    let mut f = File::open(args.next().expect("Path required")).unwrap();
    let mut code: Vec<u8> = Vec::new();

    f.read_to_end(&mut code).unwrap();

    let serialized = translator::translate_module(code.as_slice());

    if mode == "build" {
        ::std::io::stdout().write(serialized.as_slice()).unwrap();
    } else if mode == "build_json" {
        let module = Module::std_deserialize(serialized.as_slice()).unwrap();
        println!("{}", serde_json::to_string(&module).unwrap());
    } else if mode == "exec" {
        let entry = args.next().expect("Entry function required");

        let mut call_args: Vec<Value> = Vec::new();
        for arg in args {
            let v: i32 = arg.parse().unwrap();
            call_args.push(Value::I32(v));
        }

        let module = Module::std_deserialize(serialized.as_slice()).unwrap();
        let mut vm: VirtualMachine = VirtualMachine::new(&module, RuntimeConfig {
            mem_default_size_pages: 128,
            mem_max_size_pages: Some(256),
            resolver: Box::new(DefaultResolver {})
        }).unwrap();

        let entry = vm.lookup_exported_func(entry.as_str()).unwrap();
        eprintln!("{:?}", module.functions[entry]);

        let result = vm.execute(entry, &call_args);
        let result = match result {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Error: {:?}", e);
                eprintln!("Backtrace:");
                let bt = vm.backtrace();
                eprintln!("{:?}", bt);
                eprintln!("{:?}", vm.last_function());
                None
            }
        };
        println!("{:?}", result);
    } else {
        eprintln!("Unrecognized mode: {}", mode);
    }
}
