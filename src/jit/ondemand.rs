use super::llvm;
use super::runtime::Runtime;
use std::cell::RefCell;
use std::rc::Rc;
use std::os::raw::c_void;
use super::compiler::Compiler;

pub struct Ondemand {
    rt: Rc<Runtime>,
    context: llvm::Context,
    functions: Vec<RefCell<OndemandFunction>>
}

pub enum OndemandFunction {
    Processing,
    Uncompiled(llvm::Module),
    Compiled(llvm::ExecutionEngine, *const c_void)
}

impl Ondemand {
    pub fn new(rt: Rc<Runtime>, ctx: llvm::Context, fn_modules: Vec<llvm::Module>) -> Ondemand {
        Ondemand {
            rt: rt,
            context: ctx,
            functions: fn_modules.into_iter()
                .map(|f| RefCell::new(OndemandFunction::Uncompiled(f)))
                .collect()
        }
    }

    pub fn get_function_addr(&self, id: usize) -> *const c_void {
        if let OndemandFunction::Compiled(_, addr) = *self.functions[id].borrow() {
            return addr;
        }

        let mut f_handle = self.functions[id].borrow_mut();
        let f = ::std::mem::replace(&mut *f_handle, OndemandFunction::Processing);

        let ee = if let OndemandFunction::Uncompiled(m) = f {
            m.optimize();
            llvm::ExecutionEngine::new(m)
        } else {
            panic!("Unexpected OndemandFunction state");
        };

        let addr = ee.get_function_address("entry").unwrap();
        ::std::mem::replace(&mut *f_handle, OndemandFunction::Compiled(
            ee,
            addr
        ));

        addr
    }
}
