use super::llvm;
use super::runtime::Runtime;
use std::cell::{Cell, RefCell};
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
    Compiled(
        llvm::ExecutionEngine /* original */,
        Option<llvm::ExecutionEngine> /* optimized */,
        usize /* exec_count */,
        *const c_void
    )
}

const OPT_THRESHOLD: usize = 50;

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
        let mut f = self.functions[id].borrow_mut();
        match *f {
            OndemandFunction::Compiled(
                ref mut ee, ref mut optimized,
                ref mut exec_count, ref mut addr
            ) => {
                let ec = *exec_count;
                if ec < OPT_THRESHOLD || optimized.is_some() {
                    *exec_count += 1;
                    *addr
                } else {
                    eprintln!("JIT Level II");
                    let m = ee.deep_clone_module();
                    m.optimize();
                    let new_ee = llvm::ExecutionEngine::with_opt_level(m, 1);
                    let new_addr = new_ee.get_function_address("entry").unwrap();

                    *optimized = Some(new_ee);
                    *addr = new_addr;
                    *addr
                }
            },
            OndemandFunction::Uncompiled(_) => {
                if let OndemandFunction::Uncompiled(m) = ::std::mem::replace(
                    &mut *f,
                    OndemandFunction::Processing
                ) {
                    let ee = llvm::ExecutionEngine::with_opt_level(m, 0);
                    let addr = ee.get_function_address("entry").unwrap();

                    ::std::mem::replace(&mut *f, OndemandFunction::Compiled(
                        ee,
                        None,
                        0,
                        addr
                    ));

                    addr
                } else {
                    unreachable!()
                }
            },
            OndemandFunction::Processing => unreachable!()
        }
    }
}
