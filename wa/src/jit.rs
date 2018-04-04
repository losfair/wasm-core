use wasm_core::jit::compiler::Compiler;
use wasm_core::jit::runtime::RuntimeConfig;
use wasm_core::resolver::EmscriptenResolver;
use wasm_core::module::{Module, Export, Type};
use wasm_core::value::Value;
use resolver::PrivilegedResolver;
use syscall::SyscallResolver;

pub fn run_with_jit(module: &Module) {
    let mut rt_config = RuntimeConfig::default();
    rt_config.mem_default = 128 * 65536;
    rt_config.mem_max = 256 * 65536;

    let comp = Compiler::with_runtime_config(module, rt_config).unwrap();
    comp.set_native_resolver(EmscriptenResolver::new(PrivilegedResolver {
        syscall_resolver: SyscallResolver::new()
    }));

    let vm = comp.compile().unwrap().into_execution_context();

    for (k, v) in &module.exports {
        if k.starts_with("__GLOBAL_") {
            let Export::Function(id) = *v;
            vm.execute(id as usize, &[]);
        }
    }

    if let Some(start_fn) = module.start_function {
        eprintln!("Start function found");
        vm.execute(start_fn as usize, &[]);
    }

    let entry = module.lookup_exported_func("_main").unwrap();

    let typeidx = module.functions[entry].typeidx;
    let Type::Func(ref ty_args, ref ty_rets) = module.types[typeidx as usize];

    if ty_args.len() == 0 {
        vm.execute(entry, &[]);
    } else if ty_args.len() == 2 {
        vm.execute(entry, &[Value::I32(0), Value::I32(0)]);
    } else {
        panic!("Unknown main function signature");
    }
}
