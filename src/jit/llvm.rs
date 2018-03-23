use llvm_sys;
use llvm_sys::prelude::*;
use llvm_sys::core::*;
use llvm_sys::execution_engine::*;
use llvm_sys::target::*;
use std::rc::Rc;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

lazy_static! {
    static ref LLVM_EXEC: bool = {
        unsafe {
            LLVMLinkInMCJIT();
            assert_eq!(
                LLVM_InitializeNativeTarget(),
                0
            );
            assert_eq!(
                LLVM_InitializeNativeAsmPrinter(),
                0
            );
        }
        true
    };
}

#[derive(Clone)]
pub struct Context {
    inner: Rc<ContextImpl>
}

pub struct ContextImpl {
    _ref: LLVMContextRef
}

impl Context {
    pub fn new() -> Context {
        Context {
            inner: Rc::new(ContextImpl {
                _ref: unsafe { LLVMContextCreate() }
            })
        }
    }
}

impl Drop for ContextImpl {
    fn drop(&mut self) {
        unsafe { LLVMContextDispose(self._ref); }
    }
}

pub struct Module {
    _context: Context,
    _ref: LLVMModuleRef
}

impl Module {
    pub fn new(ctx: &Context, name: String) -> Module {
        let name = CString::new(name).unwrap();

        Module {
            _context: ctx.clone(),
            _ref: unsafe { LLVMModuleCreateWithNameInContext(
                name.as_ptr(),
                ctx.inner._ref
            ) }
        }
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        unsafe {
            if !self._ref.is_null() {
                LLVMDisposeModule(self._ref);
            }
        }
    }
}

pub struct ExecutionEngine {
    _context: Context,
    _ref: LLVMExecutionEngineRef
}

impl ExecutionEngine {
    pub fn new(mut m: Module) -> ExecutionEngine {
        // Ensure that LLVM JIT has been initialized
        assert_eq!(*LLVM_EXEC, true);

        unsafe {
            let mut ee: LLVMExecutionEngineRef = ::std::mem::uninitialized();
            let mut err: *mut c_char = ::std::ptr::null_mut();

            let ret = LLVMCreateExecutionEngineForModule(&mut ee, m._ref, &mut err);
            if ret != 0 {
                let e = CStr::from_ptr(err).to_str().unwrap_or_else(|_| {
                    eprintln!("Fatal error: Unable to read error string");
                    ::std::process::abort();
                });
                eprintln!("Fatal error: Unable to create execution engine from module: {}", e);
                ::std::process::abort();
            }

            m._ref = ::std::ptr::null_mut();

            ExecutionEngine {
                _context: m._context.clone(),
                _ref: ee
            }
        }
    }
}

impl Drop for ExecutionEngine {
    fn drop(&mut self) {
        unsafe {
            LLVMDisposeExecutionEngine(self._ref);
        }
    }
}
