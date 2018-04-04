extern crate wasm_core;

mod syscall;
mod stream;
mod utils;
mod jit;
mod resolver;

use std::fs::File;
use std::env;
use std::io::Read;
use std::io::Write;

use wasm_core::value::Value;
use wasm_core::module::{Module, Export, Type};
use wasm_core::executor::{VirtualMachine, RuntimeConfig, NativeResolver, NativeEntry, ExecuteError};
use wasm_core::resolver::EmscriptenResolver;
use wasm_core::optimizers::RemoveDeadBasicBlocks;
use wasm_core::cfgraph::CFGraph;
use wasm_core::trans::config::ModuleConfig;

use syscall::SyscallResolver;
use resolver::PrivilegedResolver;

fn main() {
    let mut args = env::args();
    args.next().unwrap();

    let path = args.next().expect("Path required");
    let mut f = File::open(&path).unwrap();
    let mut code: Vec<u8> = Vec::new();

    let cfg: ModuleConfig = ModuleConfig::default()
        .with_emscripten();

    f.read_to_end(&mut code).unwrap();

    let mut module = wasm_core::trans::translate_module_raw(code.as_slice(), cfg);
    for f in &mut module.functions {
        let mut cfg = CFGraph::from_function(f.body.opcodes.as_slice()).unwrap();
        cfg.validate().unwrap();
        cfg.optimize(RemoveDeadBasicBlocks).unwrap();
        cfg.validate().unwrap();
        f.body.opcodes = cfg.gen_opcodes();
    }

    jit::run_with_jit(&module);

    /*
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

    let result = {
        let typeidx = module.functions[entry].typeidx;
        let Type::Func(ref ty_args, ref ty_rets) = module.types[typeidx as usize];
        if ty_args.len() == 0 {
            vm.execute(entry, &[])
        } else if ty_args.len() == 2 {
            let argv_addr = write_main_args_emscripten(&mut vm, call_args.as_slice());
            vm.execute(entry, &[
                Value::I32(call_args.len() as i32),
                Value::I32(argv_addr)
            ])
        } else {
            panic!("Invalid signature for main function");
        }
    };
    
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
    */
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
