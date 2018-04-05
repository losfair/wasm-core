use super::llvm;
use super::runtime::Runtime;
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::os::raw::c_void;
use super::compiler;
use super::compiler::Compiler;

pub struct Ondemand {
    rt: Rc<Runtime>,
    context: llvm::Context,
    orc: llvm::Orc,
    functions: Vec<RefCell<Option<*const c_void>>>
}

const OPT_THRESHOLD: usize = 50;

impl Ondemand {
    pub fn new(rt: Rc<Runtime>, ctx: llvm::Context, m: llvm::Module) -> Ondemand {
        let orc = llvm::Orc::new();

        m.inline_with_threshold(100);
        m.optimize();

        orc.add_lazily_compiled_ir(m);

        let functions: Vec<RefCell<Option<*const c_void>>> = (0..rt.source_module.functions.len())
            .map(|_| RefCell::new(None))
            .collect();

        Ondemand {
            rt: rt,
            context: ctx,
            orc: orc,
            functions: functions
        }
    }

    pub fn get_function_addr(&self, id: usize) -> *const c_void {
        let mut f = self.functions[id].borrow_mut();
        match *f {
            Some(v) => v,
            None => {
                let name = compiler::generate_function_name(id);
                let addr = self.orc.resolve(&name);
                *f = Some(addr);
                addr
            }
        }
    }
}
