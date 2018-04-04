use wasm_core::value::Value;
use wasm_core::executor::{RuntimeConfig, NativeResolver, NativeEntry, ExecuteError};
use wasm_core::resolver::EmscriptenResolver;

use syscall::SyscallResolver;

pub struct PrivilegedResolver {
    pub syscall_resolver: SyscallResolver
}

impl NativeResolver for PrivilegedResolver {
    fn resolve(&self, module: &str, field: &str) -> Option<NativeEntry> {
        eprintln!("Resolve: {} {}", module, field);
        if module != "env" {
            return None;
        }

        match field {
            _ => self.syscall_resolver.resolve(module, field)
        }
    }
}
