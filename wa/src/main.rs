extern crate wasm_translator as translator;

mod syscall;
mod stream;
mod utils;

use std::fs::File;
use std::env;
use std::io::Read;
use std::io::Write;

use translator::wasm_core::value::Value;
use translator::wasm_core::module::{Module, Export};
use translator::wasm_core::executor::{VirtualMachine, RuntimeConfig, NativeResolver, NativeEntry, ExecuteError};
use translator::wasm_core::resolver::EmscriptenResolver;
use translator::config::ModuleConfig;

use syscall::SyscallResolver;

struct PrivilegedResolver {
    syscall_resolver: SyscallResolver
}

impl NativeResolver for PrivilegedResolver {
    fn resolve(&self, module: &str, field: &str) -> Option<NativeEntry> {
        //eprintln!("Resolve: {} {}", module, field);
        if module != "env" {
            return None;
        }

        match field {
            _ => self.syscall_resolver.resolve(module, field)
        }
    }
}

fn main() {
    let mut args = env::args();
    args.next().unwrap();

    let path = args.next().expect("Path required");
    let mut f = File::open(&path).unwrap();
    let mut code: Vec<u8> = Vec::new();

    let cfg: ModuleConfig = ModuleConfig::default()
        .with_emscripten();

    f.read_to_end(&mut code).unwrap();

    let module = translator::translate_module_raw(code.as_slice(), cfg);

    let mut call_args: Vec<String> = Vec::new();
    for arg in args {
        call_args.push(arg);
    }

    let mut vm: VirtualMachine = VirtualMachine::new(&module, RuntimeConfig {
        mem_default_size_pages: 128,
        mem_max_size_pages: Some(256),
        resolver: Box::new(EmscriptenResolver::new(PrivilegedResolver {
            syscall_resolver: SyscallResolver::new()
        }))
    }).unwrap();

    let argv_addr = write_main_args_emscripten(&mut vm, call_args.as_slice());

    for (k, v) in &module.exports {
        if k.starts_with("__GLOBAL_") {
            let Export::Function(id) = *v;
            vm.execute(id as usize, &[]).unwrap();
        }
    }

    if let Some(start_fn) = module.start_function {
        eprintln!("Start function found");
        vm.execute(start_fn as usize, &[]).unwrap();
    }

    let entry = vm.lookup_exported_func("_main").unwrap();

    let result = vm.execute(entry, &[
        Value::I32(call_args.len() as i32),
        Value::I32(argv_addr)
    ]);
    match result {
        Ok(_) => {},
        Err(e) => {
            eprintln!("Error: {:?}", e);
            eprintln!("Backtrace:");
            let bt = vm.backtrace();
            eprintln!("{:?}", bt);
            eprintln!("{:?}", vm.last_function());
        }
    }
}

fn write_main_args_emscripten(vm: &mut VirtualMachine, args: &[String]) -> i32 {
    let malloc = vm.lookup_exported_func("_malloc").unwrap();
    let list_addr = vm.execute(
        malloc,
        &[ Value::I32((args.len() * 4) as i32)]
    ).unwrap().unwrap().get_i32().unwrap() as usize;
    if list_addr == 0 {
        panic!("List malloc failed");
    }

    for i in 0..args.len() {
        let s: &[u8] = args[i].as_bytes();
        let s_addr = vm.execute(
            malloc,
            &[ Value::I32((s.len() + 1) as i32) ]
        ).unwrap().unwrap().get_i32().unwrap();
        if s_addr == 0 {
            panic!("String malloc failed");
        }
        let s_addr_bytes = unsafe {
            ::std::mem::transmute::<i32, [u8; 4]>(s_addr)
        };
        let mem = vm.get_runtime_info_mut().get_memory_mut();

        let begin = list_addr + i * 4;
        mem[begin .. begin + 4].copy_from_slice(&s_addr_bytes);

        mem[(s_addr as usize) .. (s_addr as usize) + s.len()]
            .copy_from_slice(s);
        mem[(s_addr as usize) + s.len()] = 0;
    }

    list_addr as i32
}
