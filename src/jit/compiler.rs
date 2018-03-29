use std::rc::Rc;
use std::ops::Deref;
use std::os::raw::c_void;
use module::*;
use cfgraph::*;
use super::llvm;
use opcode::{Opcode, Memarg};
use super::runtime::{Runtime, RuntimeConfig};

fn generate_function_name(id: usize) -> String {
    format!("Wfunc_{}", id)
}

pub struct Compiler<'a> {
    _context: llvm::Context,
    source_module: &'a Module,
    rt: Rc<Runtime>
}

struct CompilerIntrinsics {
    check_stack: llvm::Function,
    select: llvm::Function,
    translate_pointer: llvm::Function,
    indirect_get_function_addr: llvm::Function,
    enforce_indirect_fn_typeck: llvm::Function,
    grow_memory: llvm::Function,
    current_memory: llvm::Function
}

impl CompilerIntrinsics {
    pub fn new(ctx: &llvm::Context, m: &llvm::Module, rt: &Runtime) -> CompilerIntrinsics {
        CompilerIntrinsics {
            check_stack: Self::build_check_stack(ctx, m),
            select: Self::build_select(ctx, m),
            translate_pointer: Self::build_translate_pointer(ctx, m, rt),
            indirect_get_function_addr: Self::build_indirect_get_function_addr(ctx, m, rt),
            enforce_indirect_fn_typeck: Self::build_enforce_indirect_fn_typeck(ctx, m, rt),
            grow_memory: Self::build_grow_memory(ctx, m, rt),
            current_memory: Self::build_current_memory(ctx, m, rt)
        }
    }

    extern "C" fn stack_check_failed() {
        panic!("Stack check failed");
    }

    extern "C" fn mem_bounds_check_failed() {
        panic!("Memory bounds check failed");
    }

    extern "C" fn do_enforce_indirect_fn_typeck(rt: &Runtime, expected_typeidx: usize, indirect_index: usize) {
        let ty1 = &rt.source_module.types[expected_typeidx];

        let got_typeidx = rt.source_module.tables[0].elements[indirect_index].unwrap() as usize;
        let ty2 = &rt.source_module.types[got_typeidx];

        if expected_typeidx == got_typeidx {
            return;
        }

        if ty1 != ty2 {
            panic!("enforce_typeck: Type mismatch: Expected {:?}, got {:?}", ty1, ty2);
        }
    }

    fn build_current_memory(ctx: &llvm::Context, m: &llvm::Module, rt: &Runtime) -> llvm::Function {
        let f: llvm::Function = llvm::Function::new(
            ctx,
            m,
            "current_memory",
            llvm::Type::function(
                ctx,
                llvm::Type::int64(ctx),
                &[]
            )
        );
        let initial_bb = llvm::BasicBlock::new(&f);

        unsafe {
            let builder = initial_bb.builder();
            let jit_info = &mut *rt.get_jit_info();

            let ret = builder.build_cast(
                llvm::LLVMOpcode::LLVMZExt,
                builder.build_load(
                    builder.build_cast(
                        llvm::LLVMOpcode::LLVMIntToPtr,
                        builder.build_const_int(
                            llvm::Type::int64(ctx),
                            (&jit_info.mem_len as *const usize as usize) as u64,
                            false
                        ),
                        llvm::Type::pointer(llvm::Type::int_native(ctx))
                    )
                ),
                llvm::Type::int64(ctx)
            );
            builder.build_ret(
                builder.build_udiv(
                    ret,
                    builder.build_const_int(
                        llvm::Type::int64(ctx),
                        65536,
                        false
                    )
                )
            );
        }

        f.verify();
        f
    }

    fn build_grow_memory(ctx: &llvm::Context, m: &llvm::Module, rt: &Runtime) -> llvm::Function {
        let f: llvm::Function = llvm::Function::new(
            ctx,
            m,
            "grow_memory",
            llvm::Type::function(
                ctx,
                llvm::Type::int64(ctx), // prev_pages
                &[
                    llvm::Type::int64(ctx) // n_pages
                ]
            )
        );
        let initial_bb = llvm::BasicBlock::new(&f);

        unsafe {
            let builder = initial_bb.builder();
            let ret = builder.build_call_raw(
                builder.build_cast(
                    llvm::LLVMOpcode::LLVMIntToPtr,
                    builder.build_const_int(
                        llvm::Type::int64(ctx),
                        Runtime::_jit_grow_memory as usize as _,
                        false
                    ),
                    llvm::Type::pointer(
                        llvm::Type::function(
                            ctx,
                            llvm::Type::int_native(ctx),
                            &[
                                llvm::Type::int_native(ctx), // rt
                                llvm::Type::int_native(ctx) // len_inc
                            ]
                        )
                    )
                ),
                &[
                    builder.build_const_int(
                        llvm::Type::int_native(ctx),
                        (rt as *const Runtime as usize) as u64,
                        false
                    ),
                    builder.build_cast(
                        llvm::LLVMOpcode::LLVMTrunc,
                        builder.build_mul(
                            f.get_param(0),
                            builder.build_const_int(
                                llvm::Type::int64(ctx),
                                65536,
                                false
                            )
                        ),
                        llvm::Type::int_native(ctx)
                    )
                ]
            );
            builder.build_ret(
                builder.build_udiv(
                    builder.build_cast(
                        llvm::LLVMOpcode::LLVMZExt,
                        ret,
                        llvm::Type::int64(ctx)
                    ),
                    builder.build_const_int(
                        llvm::Type::int64(ctx),
                        65536,
                        false
                    )
                )
            );
        }

        f.verify();
        f
    }

    fn build_enforce_indirect_fn_typeck(ctx: &llvm::Context, m: &llvm::Module, rt: &Runtime) -> llvm::Function {
        let f: llvm::Function = llvm::Function::new(
            ctx,
            m,
            "enforce_indirect_fn_typeck",
            llvm::Type::function(
                ctx,
                llvm::Type::void(ctx),
                &[
                    llvm::Type::int64(ctx), // expected
                    llvm::Type::int64(ctx) // got
                ]
            )
        );
        let initial_bb = llvm::BasicBlock::new(&f);

        unsafe {
            let builder = initial_bb.builder();
            builder.build_call_raw(
                builder.build_cast(
                    llvm::LLVMOpcode::LLVMIntToPtr,
                    builder.build_const_int(
                        llvm::Type::int64(ctx),
                        (Self::do_enforce_indirect_fn_typeck as usize) as u64,
                        false
                    ),
                    llvm::Type::pointer(
                        llvm::Type::function(
                            ctx,
                            llvm::Type::void(ctx),
                            &[
                                llvm::Type::int_native(ctx),
                                llvm::Type::int_native(ctx),
                                llvm::Type::int_native(ctx)
                            ]
                        )
                    )
                ),
                &[
                    builder.build_const_int(
                        llvm::Type::int_native(ctx),
                        (rt as *const Runtime as usize) as u64,
                        false
                    ),
                    builder.build_cast(
                        llvm::LLVMOpcode::LLVMTrunc,
                        f.get_param(0),
                        llvm::Type::int_native(ctx)
                    ),
                    builder.build_cast(
                        llvm::LLVMOpcode::LLVMTrunc,
                        f.get_param(1),
                        llvm::Type::int_native(ctx)
                    )
                ]
            );
            builder.build_ret_void();
        }

        f.verify();
        f
    }

    fn build_select(ctx: &llvm::Context, m: &llvm::Module) -> llvm::Function {
        let f: llvm::Function = llvm::Function::new(
            ctx,
            m,
            "select",
            llvm::Type::function(
                ctx,
                llvm::Type::int64(ctx),
                &[
                    llvm::Type::int64(ctx), // condition
                    llvm::Type::int64(ctx), // if true
                    llvm::Type::int64(ctx) // if false
                ]
            )
        );

        let initial_bb = llvm::BasicBlock::new(&f);
        let if_true_bb = llvm::BasicBlock::new(&f);
        let if_false_bb = llvm::BasicBlock::new(&f);

        unsafe {
            let builder = initial_bb.builder();
            let cond = builder.build_icmp(
                llvm::LLVMIntNE,
                f.get_param(0),
                builder.build_const_int(
                    llvm::Type::int64(ctx),
                    0,
                    false
                )
            );
            builder.build_cond_br(
                cond,
                &if_true_bb,
                &if_false_bb
            );
        }

        unsafe {
            let builder = if_true_bb.builder();
            builder.build_ret(f.get_param(1));
        }

        unsafe {
            let builder = if_false_bb.builder();
            builder.build_ret(f.get_param(2));
        }

        f.verify();
        f
    }

    fn build_translate_pointer(ctx: &llvm::Context, m: &llvm::Module, rt: &Runtime) -> llvm::Function {
        let f: llvm::Function = llvm::Function::new(
            ctx,
            m,
            "translate_pointer",
            llvm::Type::function(
                ctx,
                llvm::Type::pointer(llvm::Type::void(ctx)),
                &[
                    llvm::Type::int64(ctx), // virtual address
                    llvm::Type::int64(ctx) // read / write length (trusted)
                ]
            )
        );

        let jit_info = unsafe {
            &mut *rt.get_jit_info()
        };
        let mem_begin_ptr = &mut jit_info.mem_begin as *mut *mut u8;
        let mem_len_ptr = &mut jit_info.mem_len as *mut usize;

        let initial_bb = llvm::BasicBlock::new(&f);
        let ok_bb = llvm::BasicBlock::new(&f);
        let failed_bb = llvm::BasicBlock::new(&f);

        let mut vaddr = unsafe { f.get_param(0) };
        let mut access_len = unsafe { f.get_param(1) };

        unsafe {
            let builder = initial_bb.builder();

            // If we are on a 32-bit system...
            if llvm::NativePointerWidth::detect() == llvm::NativePointerWidth::W32 {
                vaddr = builder.build_cast(
                    llvm::LLVMOpcode::LLVMTrunc,
                    vaddr,
                    llvm::Type::int32(ctx)
                );
                access_len = builder.build_cast(
                    llvm::LLVMOpcode::LLVMTrunc,
                    access_len,
                    llvm::Type::int32(ctx)
                );
            }

            let mem_len = builder.build_load(
                builder.build_cast(
                    llvm::LLVMOpcode::LLVMIntToPtr,
                    builder.build_const_int(
                        llvm::Type::int64(ctx),
                        mem_len_ptr as usize as _,
                        false
                    ),
                    llvm::Type::pointer(llvm::Type::int_native(ctx))
                )
            );

            let access_end = builder.build_add(
                vaddr,
                access_len
            );

            // Double check for overflow (?)
            let cmp_ok_1 = builder.build_icmp(
                llvm::LLVMIntULT,
                vaddr,
                mem_len
            );
            let cmp_ok_2 = builder.build_icmp(
                llvm::LLVMIntULE,
                access_end,
                mem_len
            );
            let cmp_ok = builder.build_and(cmp_ok_1, cmp_ok_2);

            builder.build_cond_br(cmp_ok, &ok_bb, &failed_bb);
        }

        unsafe {
            let builder = ok_bb.builder();
            let mem_begin = builder.build_load(
                builder.build_cast(
                    llvm::LLVMOpcode::LLVMIntToPtr,
                    builder.build_const_int(
                        llvm::Type::int64(ctx),
                        mem_begin_ptr as usize as _,
                        false
                    ),
                    llvm::Type::pointer(llvm::Type::int_native(ctx))
                )
            );
            let real_addr = builder.build_add(
                mem_begin,
                vaddr
            );
            builder.build_ret(
                builder.build_cast(
                    llvm::LLVMOpcode::LLVMIntToPtr,
                    real_addr,
                    llvm::Type::pointer(llvm::Type::void(ctx))
                )
            );
        }

        unsafe {
            let builder = failed_bb.builder();
            let call_target = builder.build_cast(
                llvm::LLVMOpcode::LLVMIntToPtr,
                builder.build_const_int(
                    llvm::Type::int64(ctx),
                    (Self::mem_bounds_check_failed as usize) as u64,
                    false
                ),
                llvm::Type::pointer(
                    llvm::Type::function(
                        ctx,
                        llvm::Type::void(ctx),
                        &[]
                    )
                )
            );
            builder.build_call_raw(call_target, &[]);
            builder.build_unreachable();
        }

        f.verify();
        f
    }

    fn build_check_stack(ctx: &llvm::Context, m: &llvm::Module) -> llvm::Function {
        let f: llvm::Function = llvm::Function::new(
            ctx,
            m,
            "check_stack",
            llvm::Type::function(
                ctx,
                llvm::Type::void(ctx),
                &[
                    llvm::Type::int32(ctx), // current
                    llvm::Type::int32(ctx), // lower (inclusive)
                    llvm::Type::int32(ctx) // upper (exclusive)
                ]
            )
        );
        let initial_bb = llvm::BasicBlock::new(&f);
        let check_ok_bb = llvm::BasicBlock::new(&f);
        let check_failed_bb = llvm::BasicBlock::new(&f);

        unsafe {
            let builder = initial_bb.builder();
            let upper_cmp = builder.build_icmp(
                llvm::LLVMIntSLT,
                f.get_param(0),
                f.get_param(1)
            );
            let lower_cmp = builder.build_icmp(
                llvm::LLVMIntSGE,
                f.get_param(0),
                f.get_param(2)
            );
            let cmp_result = builder.build_or(upper_cmp, lower_cmp);
            builder.build_cond_br(cmp_result, &check_failed_bb, &check_ok_bb);
        }

        unsafe {
            let builder = check_ok_bb.builder();
            builder.build_ret_void();
        }

        unsafe {
            let builder = check_failed_bb.builder();
            let call_target = builder.build_cast(
                llvm::LLVMOpcode::LLVMIntToPtr,
                builder.build_const_int(
                    llvm::Type::int64(ctx),
                    (Self::stack_check_failed as usize) as u64,
                    false
                ),
                llvm::Type::pointer(
                    llvm::Type::function(
                        ctx,
                        llvm::Type::void(ctx),
                        &[]
                    )
                )
            );
            builder.build_call_raw(call_target, &[]);
            builder.build_ret_void();
        }

        f.verify();

        f
    }

     fn build_indirect_get_function_addr(ctx: &llvm::Context, m: &llvm::Module, rt: &Runtime) -> llvm::Function {
        let f: llvm::Function = llvm::Function::new(
            ctx,
            m,
            "indirect_get_function_addr",
            llvm::Type::function(
                ctx,
                llvm::Type::pointer(llvm::Type::void(ctx)),
                &[
                    llvm::Type::int64(ctx) // function id
                ]
            )
        );
        let initial_bb = llvm::BasicBlock::new(&f);

        unsafe {
            let builder = initial_bb.builder();
            let ret = builder.build_call_raw(
                builder.build_cast(
                    llvm::LLVMOpcode::LLVMIntToPtr,
                    builder.build_const_int(
                        llvm::Type::int64(ctx),
                        (Runtime::_jit_indirect_get_function_addr as usize) as u64,
                        false
                    ),
                    llvm::Type::pointer(
                        llvm::Type::function(
                            ctx,
                            llvm::Type::pointer(llvm::Type::void(ctx)),
                            &[
                                llvm::Type::int_native(ctx),
                                llvm::Type::int_native(ctx)
                            ]
                        )
                    )
                ),
                &[
                    builder.build_const_int(
                        llvm::Type::int_native(ctx),
                        (rt as *const Runtime as usize) as u64,
                        false
                    ),
                    builder.build_cast(
                        llvm::LLVMOpcode::LLVMTrunc,
                        f.get_param(0),
                        llvm::Type::int_native(ctx)
                    )
                ]
            );
            builder.build_ret(ret);
        }

        f.verify();
        f
     }
}

struct FunctionId(usize);

pub struct CompiledModule {
    rt: Rc<Runtime>,
    source_module: Module,
    module: llvm::Module
}

pub struct ExecutionContext {
    rt: Rc<Runtime>,
    source_module: Module,
    ee: llvm::ExecutionEngine
}

impl CompiledModule {
    pub fn into_execution_context(self) -> ExecutionContext {
        ExecutionContext::from_compiled_module(self)
    }
}

impl ExecutionContext {
    pub fn from_compiled_module(m: CompiledModule) -> ExecutionContext {
        let rt = m.rt.clone();
        let ee = llvm::ExecutionEngine::new(m.module);

        let function_addrs: Vec<*const c_void> = (0..m.source_module.functions.len())
            .map(|i| ee.get_function_address(generate_function_name(i).as_str()).unwrap())
            .collect();

        rt.set_function_addrs(function_addrs);

        ExecutionContext {
            rt: rt,
            source_module: m.source_module,
            ee: ee
        }
    }

    pub fn get_function_address(&self, id: usize) -> *const c_void {
        self.rt.get_function_addr(id)
    }
}

impl<'a> Compiler<'a> {
    pub fn new(m: &'a Module, ctx: llvm::Context) -> OptimizeResult<Compiler<'a>> {
        Self::with_runtime(m, ctx, Rc::new(Runtime::new(RuntimeConfig::default(), m.clone())))
    }

    fn with_runtime(m: &'a Module, ctx: llvm::Context, rt: Rc<Runtime>) -> OptimizeResult<Compiler<'a>> {
        Ok(Compiler {
            _context: ctx.clone(),
            source_module: m,
            rt: rt
        })
    }

    pub fn compile(&self) -> OptimizeResult<CompiledModule> {
        let target_module = llvm::Module::new(&self._context, "".into());
        let target_runtime = self.rt.clone();

        let intrinsics = CompilerIntrinsics::new(&self._context, &target_module, &*self.rt);

        let target_functions: Vec<llvm::Function> = Self::gen_function_defs(
            &self._context,
            self.source_module,
            &target_module
        )?;

        for i in 0..target_functions.len() {
            Self::gen_function_body(
                &self._context,
                &self.rt,
                &intrinsics,
                self.source_module,
                &target_module,
                &target_functions,
                FunctionId(i)
            )?;
            //println!("{}", target_functions[i].to_string());
            target_functions[i].verify();
        }

        Ok(CompiledModule {
            rt: target_runtime,
            source_module: self.source_module.clone(),
            module: target_module
        })
    }

    fn gen_function_defs(
        ctx: &llvm::Context,
        source_module: &Module,
        target_module: &llvm::Module
    ) -> OptimizeResult<Vec<llvm::Function>> {
        let mut result: Vec<llvm::Function> = Vec::with_capacity(source_module.functions.len());

        for (i, f) in source_module.functions.iter().enumerate() {
            let ty = &source_module.types[f.typeidx as usize];

            let target_f: llvm::Function = llvm::Function::new(
                ctx,
                target_module,
                generate_function_name(i).as_str(),
                ty.to_llvm_function_type(ctx)
            );

            result.push(target_f);
        }

        Ok(result)
    }

    fn gen_function_body(
        ctx: &llvm::Context,
        rt: &Rc<Runtime>,
        intrinsics: &CompilerIntrinsics,
        source_module: &Module,
        target_module: &llvm::Module,
        target_functions: &[llvm::Function],
        this_function: FunctionId
    ) -> OptimizeResult<()> {
        extern "C" fn _grow_memory(rt: &Runtime, len_inc: usize) {
            rt.grow_memory(len_inc);
        }

        const STACK_SIZE: usize = 32;
        const TAG_I32: i32 = 0x01;
        const TAG_I64: i32 = 0x02;

        let source_func = &source_module.functions[this_function.0];
        let Type::Func(ref source_func_args_ty, ref source_func_ret_ty) =
            source_module.types[source_func.typeidx as usize];
        let target_func = &target_functions[this_function.0];

        let source_cfg = CFGraph::from_function(&source_func.body.opcodes)?;
        source_cfg.validate()?;

        let get_stack_elem_type = || llvm::Type::int64(ctx);
        let get_stack_array_type = || llvm::Type::array(
            ctx,
            get_stack_elem_type(),
            STACK_SIZE
        );

        let initializer_bb = llvm::BasicBlock::new(target_func);
        let stack_base;
        let stack_index;
        let locals_base;
        let n_locals = source_func_args_ty.len() + source_func.locals.len();

        unsafe {
            let builder = initializer_bb.builder();
            stack_base = builder.build_alloca(
                get_stack_array_type()
            );
            stack_index = builder.build_alloca(
                llvm::Type::int32(ctx)
            );
            builder.build_store(
                builder.build_const_int(
                    llvm::Type::int32(ctx),
                    0,
                    false
                ),
                stack_index
            );
            let mut locals_ty_info: Vec<llvm::Type> = source_func_args_ty.iter()
                .map(|v| v.to_llvm_type(ctx))
                .collect();
            locals_ty_info.extend(
                source_func.locals.iter()
                    .map(|v| v.to_llvm_type(ctx))
            );
            let locals_ty = llvm::Type::struct_type(
                ctx,
                &locals_ty_info,
                false
            );
            locals_base = builder.build_alloca(locals_ty);

            for i in 0..source_func_args_ty.len() {
                builder.build_store(
                    target_func.get_param(i),
                    builder.build_gep(
                        locals_base,
                        &[
                            builder.build_const_int(
                                llvm::Type::int32(ctx),
                                0,
                                false
                            ),
                            builder.build_const_int(
                                llvm::Type::int32(ctx),
                                i as _,
                                false
                            )
                        ]
                    )
                );
            }
        }

        let build_stack_pop = |builder: &llvm::Builder| -> llvm::LLVMValueRef {
            unsafe {
                let cur_stack_index = builder.build_load(
                    stack_index
                );
                builder.build_call(
                    &intrinsics.check_stack,
                    &[
                        cur_stack_index,
                        builder.build_const_int(
                            llvm::Type::int32(ctx),
                            1,
                            false
                        ),
                        builder.build_const_int(
                            llvm::Type::int32(ctx),
                            (STACK_SIZE + 1) as _,
                            false
                        )
                    ]
                );
                let new_stack_index = builder.build_sub(
                    cur_stack_index,
                    builder.build_const_int(
                        llvm::Type::int32(ctx),
                        1,
                        false
                    )
                );
                builder.build_store(
                    new_stack_index,
                    stack_index
                );
                let ret = builder.build_load(
                    builder.build_gep(
                        stack_base,
                        &[
                            builder.build_const_int(
                                llvm::Type::int32(ctx),
                                0,
                                false
                            ),
                            new_stack_index
                        ]
                    )
                );
                ret
            }
        };

        let build_stack_pop_i32 = |builder: &llvm::Builder| -> llvm::LLVMValueRef {
            unsafe {
                builder.build_cast(
                    llvm::LLVMOpcode::LLVMTrunc,
                    build_stack_pop(builder),
                    llvm::Type::int32(ctx)
                )
            }
        };

        let build_stack_push = |builder: &llvm::Builder, v: llvm::LLVMValueRef| {
            unsafe {
                let cur_stack_index = builder.build_load(
                    stack_index
                );
                builder.build_call(
                    &intrinsics.check_stack,
                    &[
                        cur_stack_index,
                        builder.build_const_int(
                            llvm::Type::int32(ctx),
                            0,
                            false
                        ),
                        builder.build_const_int(
                            llvm::Type::int32(ctx),
                            STACK_SIZE as _,
                            false
                        )
                    ]
                );
                builder.build_store(
                    v,
                    builder.build_gep(
                        stack_base,
                        &[
                            builder.build_const_int(
                                llvm::Type::int32(ctx),
                                0,
                                false
                            ),
                            cur_stack_index
                        ]
                    )
                );
                builder.build_store(
                    builder.build_add(
                        cur_stack_index,
                        builder.build_const_int(
                            llvm::Type::int32(ctx),
                            1,
                            false
                        )
                    ),
                    stack_index
                );
            }
        };

        let build_stack_push_i32 = |builder: &llvm::Builder, v: llvm::LLVMValueRef| {
            unsafe {
                build_stack_push(
                    builder,
                    builder.build_cast(
                        llvm::LLVMOpcode::LLVMZExt,
                        v,
                        llvm::Type::int64(ctx)
                    )
                )
            }
        };

        let build_get_local = |builder: &llvm::Builder, id: usize| -> llvm::LLVMValueRef {
            if id >= n_locals {
                panic!("Local index out of bound");
            }
            unsafe {
                builder.build_load(
                    builder.build_gep(
                        locals_base,
                        &[
                            builder.build_const_int(
                                llvm::Type::int32(ctx),
                                0,
                                false
                            ),
                            builder.build_const_int(
                                llvm::Type::int32(ctx),
                                id as _,
                                false
                            )
                        ]
                    )
                )
            }
        };

        let build_set_local = |builder: &llvm::Builder, id: usize, v: llvm::LLVMValueRef| {
            if id >= n_locals {
                panic!("Local index out of bound");
            }
            unsafe {
                builder.build_store(
                    v,
                    builder.build_gep(
                        locals_base,
                        &[
                            builder.build_const_int(
                                llvm::Type::int32(ctx),
                                0,
                                false
                            ),
                            builder.build_const_int(
                                llvm::Type::int32(ctx),
                                id as _,
                                false
                            )
                        ]
                    )
                )
            }
        };

        let build_i32_binop = |
            builder: &llvm::Builder,
            op: &Fn(&llvm::Builder, llvm::LLVMValueRef, llvm::LLVMValueRef) -> llvm::LLVMValueRef
        | {
            unsafe {
                let b = build_stack_pop_i32(&builder);
                let a = build_stack_pop_i32(&builder);
                build_stack_push_i32(
                    &builder,
                    op(builder, a, b)
                );
            }
        };

        let build_i32_relop = |
            builder: &llvm::Builder,
            op: &Fn(&llvm::Builder, llvm::LLVMValueRef, llvm::LLVMValueRef) -> llvm::LLVMValueRef
        | {
            unsafe {
                let b = build_stack_pop_i32(&builder);
                let a = build_stack_pop_i32(&builder);
                build_stack_push(
                    &builder,
                    builder.build_cast(
                        llvm::LLVMOpcode::LLVMZExt,
                        op(builder, a, b),
                        llvm::Type::int64(ctx)
                    )
                );
            }
        };

        let build_i64_binop = |
            builder: &llvm::Builder,
            op: &Fn(&llvm::Builder, llvm::LLVMValueRef, llvm::LLVMValueRef) -> llvm::LLVMValueRef
        | {
            unsafe {
                let b = build_stack_pop(&builder);
                let a = build_stack_pop(&builder);
                build_stack_push(
                    &builder,
                    op(builder, a, b)
                );
            }
        };

        let build_i64_relop = |
            builder: &llvm::Builder,
            op: &Fn(&llvm::Builder, llvm::LLVMValueRef, llvm::LLVMValueRef) -> llvm::LLVMValueRef
        | {
            unsafe {
                let b = build_stack_pop(&builder);
                let a = build_stack_pop(&builder);
                build_stack_push(
                    &builder,
                    builder.build_cast(
                        llvm::LLVMOpcode::LLVMZExt,
                        op(builder, a, b),
                        llvm::Type::int64(ctx)
                    )
                );
            }
        };

        let build_std_zext = |
            builder: &llvm::Builder,
            v: llvm::LLVMValueRef
        | -> llvm::LLVMValueRef {
            unsafe {
                builder.build_cast(
                    llvm::LLVMOpcode::LLVMZExt,
                    v,
                    llvm::Type::int64(ctx)
                )
            }
        };

        let build_std_sext = |
            builder: &llvm::Builder,
            v: llvm::LLVMValueRef
        | -> llvm::LLVMValueRef {
            unsafe {
                builder.build_cast(
                    llvm::LLVMOpcode::LLVMSExt,
                    v,
                    llvm::Type::int64(ctx)
                )
            }
        };

        let build_std_load = |
            builder: &llvm::Builder,
            trusted_len: usize,
            signed: bool,
            ptr: llvm::LLVMValueRef
        | -> llvm::LLVMValueRef {
            unsafe {
                let real_addr = builder.build_call(
                    &intrinsics.translate_pointer,
                    &[
                        ptr,
                        builder.build_const_int(
                            llvm::Type::int64(ctx),
                            trusted_len as _,
                            false
                        )
                    ]
                );
                let build_ext: &Fn(&llvm::Builder, llvm::LLVMValueRef) -> llvm::LLVMValueRef =
                    if signed {
                        &build_std_sext
                    } else {
                        &build_std_zext
                    };

                if trusted_len == 8 {
                    build_ext(
                        builder,
                        builder.build_load(
                            builder.build_bitcast(
                                real_addr,
                                llvm::Type::pointer(llvm::Type::int64(ctx))
                            )
                        )
                    )
                } else if trusted_len == 4 {
                    build_ext(
                        builder,
                        builder.build_load(
                            builder.build_bitcast(
                                real_addr,
                                llvm::Type::pointer(llvm::Type::int32(ctx))
                            )
                        )
                    )
                } else if trusted_len == 2 {
                    build_ext(
                        builder,
                        builder.build_load(
                            builder.build_bitcast(
                                real_addr,
                                llvm::Type::pointer(llvm::Type::int16(ctx))
                            )
                        )
                    )
                } else if trusted_len == 1 {
                    build_ext(
                        builder,
                        builder.build_load(
                            builder.build_bitcast(
                                real_addr,
                                llvm::Type::pointer(llvm::Type::int8(ctx))
                            )
                        )
                    )
                } else {
                    panic!("Unknown trusted length: {}", trusted_len);
                }
            }
        };

        let build_std_store = |
            builder: &llvm::Builder,
            trusted_len: usize,
            val: llvm::LLVMValueRef,
            ptr: llvm::LLVMValueRef
        | {
            unsafe {
                let real_addr = builder.build_call(
                    &intrinsics.translate_pointer,
                    &[
                        ptr,
                        builder.build_const_int(
                            llvm::Type::int64(ctx),
                            trusted_len as _,
                            false
                        )
                    ]
                );

                if trusted_len == 8 {
                    builder.build_store(
                        val,
                        builder.build_bitcast(
                            real_addr,
                            llvm::Type::pointer(llvm::Type::int64(ctx))
                        )
                    );
                } else if trusted_len == 4 {
                    builder.build_store(
                        builder.build_cast(
                            llvm::LLVMOpcode::LLVMTrunc,
                            val,
                            llvm::Type::int32(ctx)
                        ),
                        builder.build_bitcast(
                            real_addr,
                            llvm::Type::pointer(llvm::Type::int32(ctx))
                        )
                    );
                } else if trusted_len == 2 {
                    builder.build_store(
                        builder.build_cast(
                            llvm::LLVMOpcode::LLVMTrunc,
                            val,
                            llvm::Type::int16(ctx)
                        ),
                        builder.build_bitcast(
                            real_addr,
                            llvm::Type::pointer(llvm::Type::int16(ctx))
                        )
                    );
                } else if trusted_len == 1 {
                    builder.build_store(
                        builder.build_cast(
                            llvm::LLVMOpcode::LLVMTrunc,
                            val,
                            llvm::Type::int8(ctx)
                        ),
                        builder.build_bitcast(
                            real_addr,
                            llvm::Type::pointer(llvm::Type::int8(ctx))
                        )
                    );
                } else {
                    panic!("Unknown trusted length: {}", trusted_len);
                }
            }
        };

        let build_indirect_get_function_addr = |
            builder: &llvm::Builder,
            id: llvm::LLVMValueRef
        | -> llvm::LLVMValueRef {
            unsafe {
                builder.build_call(
                    &intrinsics.indirect_get_function_addr,
                    &[
                        id
                    ]
                )
            }
        };

        let build_indirect_fn_typeck = |
            builder: &llvm::Builder,
            expected: u64,
            indirect_index: llvm::LLVMValueRef
        | {
            unsafe {
                builder.build_call(
                    &intrinsics.enforce_indirect_fn_typeck,
                    &[
                        builder.build_const_int(
                            llvm::Type::int64(ctx),
                            expected,
                            false
                        ),
                        indirect_index
                    ]
                );
            }
        };

        let build_grow_memory = |
            builder: &llvm::Builder,
            n_pages: llvm::LLVMValueRef
        | -> llvm::LLVMValueRef {
            unsafe {
                builder.build_call(
                    &intrinsics.grow_memory,
                    &[
                        n_pages
                    ]
                )
            }
        };

        let build_current_memory = |
            builder: &llvm::Builder
        | -> llvm::LLVMValueRef {
            unsafe {
                builder.build_call(
                    &intrinsics.current_memory,
                    &[]
                )
            }
        };

        let target_basic_blocks: Vec<llvm::BasicBlock<'_>> = (0..source_cfg.blocks.len())
            .map(|_| llvm::BasicBlock::new(target_func))
            .collect();

        for (i, bb) in source_cfg.blocks.iter().enumerate() {
            let target_bb = &target_basic_blocks[i];

            for op in &bb.opcodes {
                unsafe {
                    match *op {
                        Opcode::Nop => {},
                        Opcode::Drop => {
                            build_stack_pop(&target_bb.builder());
                        },
                        Opcode::Select => {
                            let builder = target_bb.builder();
                            let c = build_stack_pop(&builder);
                            let val2 = build_stack_pop(&builder);
                            let val1 = build_stack_pop(&builder);
                            let v = builder.build_call(
                                &intrinsics.select,
                                &[
                                    c,
                                    val1,
                                    val2
                                ]
                            );
                            build_stack_push(&builder, v);
                        },
                        Opcode::Call(id) => {
                            let builder = target_bb.builder();

                            let f = &source_module.functions[id as usize];
                            let Type::Func(ref ft_args, ref ft_ret) = source_module.types[f.typeidx as usize];
                            let target_f = &target_functions[id as usize];

                            let mut call_args: Vec<llvm::LLVMValueRef> = ft_args.iter().rev()
                                .map(|t| {
                                    builder.build_bitcast(
                                        build_stack_pop(&builder),
                                        t.to_llvm_type(ctx)
                                    )
                                })
                                .collect();

                            call_args.reverse();

                            let ret = builder.build_call(
                                target_f,
                                &call_args
                            );

                            if ft_ret.len() > 0 {
                                build_stack_push(&builder, ret);
                            }
                        },
                        Opcode::CallIndirect(typeidx) => {
                            let builder = target_bb.builder();

                            let indirect_index = build_stack_pop(&builder);

                            let ft = &source_module.types[typeidx as usize];
                            let Type::Func(ref ft_args, ref ft_ret) = *ft;

                            let mut call_args: Vec<llvm::LLVMValueRef> = ft_args.iter().rev()
                                .map(|t| {
                                    builder.build_bitcast(
                                        build_stack_pop(&builder),
                                        t.to_llvm_type(ctx)
                                    )
                                })
                                .collect();

                            call_args.reverse();

                            build_indirect_fn_typeck(
                                &builder,
                                typeidx as u64,
                                indirect_index
                            );
                            let real_addr = build_indirect_get_function_addr(&builder, indirect_index);
                            let ret = builder.build_call_raw(
                                builder.build_bitcast(
                                    real_addr,
                                    llvm::Type::pointer(ft.to_llvm_function_type(ctx))
                                ),
                                &call_args
                            );

                            if ft_ret.len() > 0 {
                                build_stack_push(&builder, ret);
                            }
                        },
                        Opcode::NativeInvoke(id) => {
                            let builder = target_bb.builder();

                            let native = &source_module.natives[id as usize];

                            let ft = &source_module.types[native.typeidx as usize];
                            let Type::Func(ref ft_args, ref ft_ret) = *ft;

                            let mut call_args: Vec<llvm::LLVMValueRef> = ft_args.iter().rev()
                                .map(|_| build_stack_pop(&builder))
                                .collect();

                            call_args.reverse();

                            let req = builder.build_call_raw(
                                builder.build_cast(
                                    llvm::LLVMOpcode::LLVMIntToPtr,
                                    builder.build_const_int(
                                        llvm::Type::int64(ctx),
                                        (Runtime::_jit_native_invoke_request as usize) as u64,
                                        false
                                    ),
                                    llvm::Type::pointer(llvm::Type::function(
                                        ctx,
                                        llvm::Type::int_native(ctx), // req ptr
                                        &[
                                            llvm::Type::int_native(ctx) // n_args
                                        ]
                                    ))
                                ),
                                &[
                                    builder.build_const_int(
                                        llvm::Type::int_native(ctx),
                                        call_args.len() as _,
                                        false
                                    )
                                ]
                            );
                            for arg in &call_args {
                                builder.build_call_raw(
                                    builder.build_cast(
                                        llvm::LLVMOpcode::LLVMIntToPtr,
                                        builder.build_const_int(
                                            llvm::Type::int64(ctx),
                                            (Runtime::_jit_native_invoke_push_arg as usize) as u64,
                                            false
                                        ),
                                        llvm::Type::pointer(llvm::Type::function(
                                            ctx,
                                            llvm::Type::void(ctx),
                                            &[
                                                llvm::Type::int_native(ctx), // req ptr
                                                llvm::Type::int64(ctx) // arg
                                            ]
                                        ))
                                    ),
                                    &[
                                        req,
                                        *arg
                                    ]
                                );
                            }
                            let rt: &Runtime = &*rt;
                            let ret = builder.build_call_raw(
                                builder.build_cast(
                                    llvm::LLVMOpcode::LLVMIntToPtr,
                                    builder.build_const_int(
                                        llvm::Type::int64(ctx),
                                        (Runtime::_jit_native_invoke_complete as usize) as u64,
                                        false
                                    ),
                                    llvm::Type::pointer(llvm::Type::function(
                                        ctx,
                                        llvm::Type::int64(ctx),
                                        &[
                                            llvm::Type::int_native(ctx), // rt
                                            llvm::Type::int_native(ctx), // id
                                            llvm::Type::int_native(ctx) // req ptr
                                        ]
                                    ))
                                ),
                                &[
                                    builder.build_const_int(
                                        llvm::Type::int_native(ctx),
                                        rt as *const Runtime as usize as _,
                                        false
                                    ),
                                    builder.build_const_int(
                                        llvm::Type::int_native(ctx),
                                        id as _,
                                        false
                                    ),
                                    req
                                ]
                            );
                            if ft_ret.len() > 0 {
                                build_stack_push(&builder, ret);
                            }
                        },
                        Opcode::CurrentMemory => {
                            let builder = target_bb.builder();
                            build_stack_push(
                                &builder,
                                build_current_memory(&builder)
                            );
                        },
                        Opcode::GrowMemory => {
                            let builder = target_bb.builder();
                            let n_pages = build_stack_pop(&builder);
                            build_stack_push(
                                &builder,
                                build_grow_memory(&builder, n_pages)
                            );
                        },
                        Opcode::GetLocal(id) => {
                            let builder = target_bb.builder();
                            let v = build_get_local(&builder, id as _);
                            build_stack_push(&builder, v);
                        },
                        Opcode::SetLocal(id) => {
                            let builder = target_bb.builder();
                            let v = build_stack_pop(&builder);
                            build_set_local(&builder, id as _, v);
                        },
                        Opcode::TeeLocal(id) => {
                            let builder = target_bb.builder();
                            let v = build_stack_pop(&builder);
                            build_stack_push(&builder, v);
                            build_set_local(&builder, id as _, v);
                        },
                        Opcode::I32Const(v) => {
                            let builder = target_bb.builder();
                            let v = builder.build_cast(
                                llvm::LLVMOpcode::LLVMZExt,
                                builder.build_const_int(
                                    llvm::Type::int32(ctx),
                                    v as _,
                                    false
                                ),
                                llvm::Type::int64(ctx)
                            );
                            build_stack_push(&builder, v);
                        },
                        Opcode::I32Add => {
                            build_i32_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_add(a, b)
                            );
                        },
                        Opcode::I32Sub => {
                            build_i32_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_sub(a, b)
                            );
                        },
                        Opcode::I32Mul => {
                            build_i32_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_mul(a, b)
                            );
                        },
                        Opcode::I32DivU => {
                            build_i32_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_udiv(a, b)
                            );
                        },
                        Opcode::I32DivS => {
                            build_i32_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_sdiv(a, b)
                            );
                        },
                        Opcode::I32RemU => {
                            build_i32_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_urem(a, b)
                            );
                        },
                        Opcode::I32RemS => {
                            build_i32_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_srem(a, b)
                            );
                        },
                        Opcode::I32Shl => {
                            build_i32_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_shl(a, b)
                            );
                        },
                        Opcode::I32ShrU => {
                            build_i32_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_lshr(a, b)
                            );
                        },
                        Opcode::I32ShrS => {
                            build_i32_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_ashr(a, b)
                            );
                        },
                        Opcode::I32Eq => {
                            build_i32_relop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_icmp(
                                    llvm::LLVMIntEQ,
                                    a,
                                    b
                                )
                            );
                        },
                        Opcode::I32Ne => {
                            build_i32_relop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_icmp(
                                    llvm::LLVMIntNE,
                                    a,
                                    b
                                )
                            );
                        },
                        Opcode::I32LtU => {
                            build_i32_relop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_icmp(
                                    llvm::LLVMIntULT,
                                    a,
                                    b
                                )
                            );
                        },
                        Opcode::I32LtS => {
                            build_i32_relop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_icmp(
                                    llvm::LLVMIntSLT,
                                    a,
                                    b
                                )
                            );
                        },
                        Opcode::I32LeU => {
                            build_i32_relop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_icmp(
                                    llvm::LLVMIntULE,
                                    a,
                                    b
                                )
                            );
                        },
                        Opcode::I32LeS => {
                            build_i32_relop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_icmp(
                                    llvm::LLVMIntSLE,
                                    a,
                                    b
                                )
                            );
                        },
                        Opcode::I32GtU => {
                            build_i32_relop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_icmp(
                                    llvm::LLVMIntUGT,
                                    a,
                                    b
                                )
                            );
                        },
                        Opcode::I32GtS => {
                            build_i32_relop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_icmp(
                                    llvm::LLVMIntSGT,
                                    a,
                                    b
                                )
                            );
                        },
                        Opcode::I32GeU => {
                            build_i32_relop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_icmp(
                                    llvm::LLVMIntUGE,
                                    a,
                                    b
                                )
                            );
                        },
                        Opcode::I32GeS => {
                            build_i32_relop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_icmp(
                                    llvm::LLVMIntSGE,
                                    a,
                                    b
                                )
                            );
                        },
                        Opcode::I32WrapI64 => {
                            let builder = target_bb.builder();
                            build_stack_push_i32(
                                &builder,
                                builder.build_cast(
                                    llvm::LLVMOpcode::LLVMTrunc,
                                    build_stack_pop(&builder),
                                    llvm::Type::int32(ctx)
                                )
                            );
                        },
                        Opcode::I32Load(ref m) => {
                            let builder = target_bb.builder();
                            let addr = build_stack_pop(&builder);
                            build_stack_push(&builder, build_std_load(
                                &builder,
                                4,
                                false, // signed
                                builder.build_add(
                                    addr,
                                    builder.build_const_int(
                                        llvm::Type::int64(ctx),
                                        m.offset as _,
                                        false
                                    )
                                )
                            ));
                        },
                        Opcode::I32Load8U(ref m) => {
                            let builder = target_bb.builder();
                            let addr = build_stack_pop(&builder);
                            build_stack_push(&builder, build_std_load(
                                &builder,
                                1,
                                false, // signed
                                builder.build_add(
                                    addr,
                                    builder.build_const_int(
                                        llvm::Type::int64(ctx),
                                        m.offset as _,
                                        false
                                    )
                                )
                            ));
                        },
                        Opcode::I32Load8S(ref m) => {
                            let builder = target_bb.builder();
                            let addr = build_stack_pop(&builder);
                            build_stack_push(&builder, build_std_load(
                                &builder,
                                1,
                                true, // signed
                                builder.build_add(
                                    addr,
                                    builder.build_const_int(
                                        llvm::Type::int64(ctx),
                                        m.offset as _,
                                        false
                                    )
                                )
                            ));
                        },
                        Opcode::I32Load16U(ref m) => {
                            let builder = target_bb.builder();
                            let addr = build_stack_pop(&builder);
                            build_stack_push(&builder, build_std_load(
                                &builder,
                                2,
                                false, // signed
                                builder.build_add(
                                    addr,
                                    builder.build_const_int(
                                        llvm::Type::int64(ctx),
                                        m.offset as _,
                                        false
                                    )
                                )
                            ));
                        },
                        Opcode::I32Load16S(ref m) => {
                            let builder = target_bb.builder();
                            let addr = build_stack_pop(&builder);
                            build_stack_push(&builder, build_std_load(
                                &builder,
                                2,
                                true, // signed
                                builder.build_add(
                                    addr,
                                    builder.build_const_int(
                                        llvm::Type::int64(ctx),
                                        m.offset as _,
                                        false
                                    )
                                )
                            ));
                        },
                        Opcode::I32Store(ref m) => {
                            let builder = target_bb.builder();
                            let val = build_stack_pop_i32(&builder);
                            let addr = build_stack_pop(&builder);
                            build_std_store(
                                &builder,
                                4,
                                val,
                                builder.build_add(
                                    addr,
                                    builder.build_const_int(
                                        llvm::Type::int64(ctx),
                                        m.offset as _,
                                        false
                                    )
                                )
                            );
                        },
                        Opcode::I32Store8(ref m) => {
                            let builder = target_bb.builder();
                            let val = build_stack_pop_i32(&builder);
                            let addr = build_stack_pop(&builder);
                            build_std_store(
                                &builder,
                                1,
                                val,
                                builder.build_add(
                                    addr,
                                    builder.build_const_int(
                                        llvm::Type::int64(ctx),
                                        m.offset as _,
                                        false
                                    )
                                )
                            );
                        },
                        Opcode::I32Store16(ref m) => {
                            let builder = target_bb.builder();
                            let val = build_stack_pop_i32(&builder);
                            let addr = build_stack_pop(&builder);
                            build_std_store(
                                &builder,
                                2,
                                val,
                                builder.build_add(
                                    addr,
                                    builder.build_const_int(
                                        llvm::Type::int64(ctx),
                                        m.offset as _,
                                        false
                                    )
                                )
                            );
                        },
                        Opcode::I64Const(v) => {
                            let builder = target_bb.builder();
                            let v = builder.build_const_int(
                                llvm::Type::int64(ctx),
                                v as _,
                                false
                            );
                            build_stack_push(&builder, v);
                        },
                        _ => panic!("Opcode not implemented: {:?}", op)
                    }
                }
            }
        }

        for (i, bb) in source_cfg.blocks.iter().enumerate() {
            let target_bb = &target_basic_blocks[i];
            let builder = target_bb.builder();

            unsafe { 
                match *bb.br.as_ref().unwrap() {
                    Branch::Jmp(id) => {
                        builder.build_br(&target_basic_blocks[id.0]);
                    },
                    Branch::JmpEither(a, b) => {
                        let v = build_stack_pop(&builder);
                        let cond = builder.build_icmp(
                            llvm::LLVMIntNE,
                            v,
                            builder.build_const_int(
                                llvm::Type::int64(ctx),
                                0,
                                false
                            )
                        );
                        builder.build_cond_br(
                            cond,
                            &target_basic_blocks[a.0],
                            &target_basic_blocks[b.0]
                        );
                    },
                    Branch::JmpTable(ref targets, otherwise) => {
                        let v = build_stack_pop(&builder);

                        let targets: Vec<(llvm::LLVMValueRef, &llvm::BasicBlock)> = targets.iter()
                            .enumerate()
                            .map(|(i, t)| {
                                (
                                    builder.build_const_int(
                                        llvm::Type::int64(ctx),
                                        i as _,
                                        false
                                    ),
                                    &target_basic_blocks[t.0]
                                )
                            })
                            .collect();

                        builder.build_switch(
                            v,
                            &targets,
                            &target_basic_blocks[otherwise.0]
                        );
                    },
                    Branch::Return => {
                        if source_func_ret_ty.len() == 0 {
                            builder.build_ret_void();
                        } else {
                            let v = build_stack_pop(&builder);
                            builder.build_ret(v);
                        }
                    }
                }
            }
        }

        unsafe {
            if target_basic_blocks.len() == 0 {
                initializer_bb.builder().build_ret_void();
            } else {
                initializer_bb.builder().build_br(&target_basic_blocks[0]);
            }
        }

        Ok(())
    }
}

impl ValType {
    fn to_llvm_type(&self, ctx: &llvm::Context) -> llvm::Type {
        match *self {
            ValType::I32 | ValType::I64 => llvm::Type::int64(ctx),
            //ValType::I64 => llvm::Type::int64(ctx),
            ValType::F32 => llvm::Type::float32(ctx),
            ValType::F64 => llvm::Type::float64(ctx)
        }
    }
}

impl Type {
    fn to_llvm_function_type(&self, ctx: &llvm::Context) -> llvm::Type {
        let Type::Func(ref ft_args, ref ft_ret) = *self;

        let target_ft_args: Vec<llvm::Type> = ft_args
            .iter()
            .map(|t| t.to_llvm_type(ctx))
            .collect();

        llvm::Type::function(
            ctx,
            if ft_ret.len() == 0 {
                llvm::Type::void(ctx)
            } else if ft_ret.len() == 1 {
                ft_ret[0].to_llvm_type(ctx)
            } else {
                panic!("Invalid number of return values");
            },
            target_ft_args.as_slice()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use test::Bencher;

    fn prepare_module_from_fn_bodies(
        mut m: Module,
        fns: Vec<(Type, Vec<ValType>, Vec<Opcode>)>
    ) -> CompiledModule {
        for (i, f) in fns.into_iter().enumerate() {
            m.types.push(f.0);
            m.functions.push(Function {
                name: None,
                typeidx: i as _,
                locals: f.1,
                body: FunctionBody {
                    opcodes: f.2
                }
            });
        }
        let compiler = Compiler::new(&m, llvm::Context::new()).unwrap();
        let target_module = compiler.compile().unwrap();

        target_module.module.optimize();
        target_module
    }

    fn build_module_from_fn_bodies(
        fns: Vec<(Type, Vec<ValType>, Vec<Opcode>)>
    ) -> CompiledModule {
        prepare_module_from_fn_bodies(Module::default(), fns)
    }

    fn build_module_from_fn_body(ty: Type, locals: Vec<ValType>, body: Vec<Opcode>) -> CompiledModule {
        build_module_from_fn_bodies(
            vec! [ (ty, locals, body) ]
        )
    }

    fn build_ee_from_fn_bodies(
        fns: Vec<(Type, Vec<ValType>, Vec<Opcode>)>
    ) -> ExecutionContext {
        let target_module = build_module_from_fn_bodies(fns);
        target_module.into_execution_context()
    }

    fn build_ee_from_fn_body(ty: Type, locals: Vec<ValType>, body: Vec<Opcode>) -> ExecutionContext {
        build_ee_from_fn_bodies(
            vec! [ (ty, locals, body) ]
        )
    }

    #[bench]
    fn bench_simple_jit_compile(b: &mut Bencher) {
        b.iter(|| {
            let _ = build_ee_from_fn_body(
                Type::Func(vec! [], vec! [ ValType::I32 ]),
                vec! [],
                vec! [
                    Opcode::I32Const(42), // 0
                    Opcode::Jmp(3), // 1
                    Opcode::I32Const(21), // 2
                    Opcode::Return // 3
                ]
            );
        });
    }

    #[test]
    fn test_concurrent_jit_compile() {
        let threads: Vec<::std::thread::JoinHandle<()>> = (0..8).map(|_| {
            ::std::thread::spawn(|| {
                for _ in 0..300 {
                    let _ = build_ee_from_fn_body(
                        Type::Func(vec! [], vec! [ ValType::I32 ]),
                        vec! [],
                        vec! [
                            Opcode::I32Const(42), // 0
                            Opcode::Jmp(3), // 1
                            Opcode::I32Const(21), // 2
                            Opcode::Return // 3
                        ]
                    );
                }
            })
        }).collect();
        for t in threads {
            t.join().unwrap();
        }
    }

    #[test]
    fn test_simple_jit() {
        let ee = build_ee_from_fn_body(
            Type::Func(vec! [], vec! [ ValType::I32 ]),
            vec! [],
            vec! [
                Opcode::I32Const(42), // 0
                Opcode::Jmp(3), // 1
                Opcode::I32Const(21), // 2
                Opcode::Return // 3
            ]
        );

        println!("{}", ee.ee.to_string());

        let f: extern "C" fn () -> i64 = unsafe {
            ::std::mem::transmute(ee.get_function_address(0))
        };
        let ret = f();
        assert_eq!(ret, 42);
    }

    #[test]
    fn test_operand_stack_overflow() {
        let ee = build_ee_from_fn_body(
            Type::Func(vec! [], vec! [ ValType::I32 ]),
            vec! [],
            vec! [
                Opcode::I32Const(42), // 0
                Opcode::Jmp(0), // 1
                Opcode::Return // 2
            ]
        );

        let f: extern "C" fn () -> i64 = unsafe {
            ::std::mem::transmute(ee.get_function_address(0))
        };
        match catch_unwind(AssertUnwindSafe(|| f())) {
            Ok(_) => panic!("Expecting panic"),
            Err(_) => {}
        }
    }

    #[test]
    fn test_operand_stack_underflow() {
        let ee = build_ee_from_fn_body(
            Type::Func(vec! [], vec! [ ValType::I32 ]),
            vec! [],
            vec! [
                Opcode::Drop, // 0
                Opcode::Return // 1
            ]
        );

        let f: extern "C" fn () -> i64 = unsafe {
            ::std::mem::transmute(ee.get_function_address(0))
        };
        match catch_unwind(AssertUnwindSafe(|| f())) {
            Ok(_) => panic!("Expecting panic"),
            Err(_) => {}
        }
    }

    #[test]
    fn test_select_true() {
        let ee = build_ee_from_fn_body(
            Type::Func(vec! [], vec! [ ValType::I32 ]),
            vec! [],
            vec! [
                Opcode::I32Const(1),
                Opcode::I32Const(2),
                Opcode::I32Const(1),
                Opcode::Select,
                Opcode::Return
            ]
        );

        let f: extern "C" fn () -> i64 = unsafe {
            ::std::mem::transmute(ee.get_function_address(0))
        };
        let ret = f();
        assert_eq!(ret, 1);
    }

    #[test]
    fn test_select_false() {
        let ee = build_ee_from_fn_body(
            Type::Func(vec! [], vec! [ ValType::I32 ]),
            vec! [],
            vec! [
                Opcode::I32Const(1),
                Opcode::I32Const(2),
                Opcode::I32Const(0),
                Opcode::Select,
                Opcode::Return
            ]
        );

        let f: extern "C" fn () -> i64 = unsafe {
            ::std::mem::transmute(ee.get_function_address(0))
        };
        let ret = f();
        assert_eq!(ret, 2);
    }

    #[test]
    fn test_get_local() {
        let ee = build_ee_from_fn_body(
            Type::Func(vec! [ ValType::I32 ], vec! [ ValType::I32 ]),
            vec! [],
            vec! [
                Opcode::GetLocal(0),
                Opcode::I32Const(1),
                Opcode::I32Add,
                Opcode::Return
            ]
        );

        //println!("{}", ee.to_string());

        let f: extern "C" fn (v: i64) -> i64 = unsafe {
            ::std::mem::transmute(ee.get_function_address(0))
        };
        let ret = f(22);
        assert_eq!(ret, 23);
    }

    #[test]
    fn test_set_local() {
        let ee = build_ee_from_fn_body(
            Type::Func(vec! [ ValType::I32 ], vec! [ ValType::I32 ]),
            vec! [
                ValType::I32
            ],
            vec! [
                Opcode::GetLocal(0),
                Opcode::SetLocal(1),
                Opcode::GetLocal(1),
                Opcode::Return
            ]
        );

        //println!("{}", ee.to_string());

        let f: extern "C" fn (v: i64) -> i64 = unsafe {
            ::std::mem::transmute(ee.get_function_address(0))
        };
        let ret = f(22);
        assert_eq!(ret, 22);
    }

    #[test]
    fn test_jmp_table() {
        let ee = build_ee_from_fn_body(
            Type::Func(vec! [ ValType::I32 ], vec! [ ValType::I32 ]),
            vec! [],
            vec! [
                Opcode::GetLocal(0), // 0
                Opcode::JmpTable(vec! [
                    4,
                    6,
                    8
                ], 2), // 1
                Opcode::I32Const(-1), // 2
                Opcode::Return, // 3
                Opcode::I32Const(11), // 4
                Opcode::Return, // 5
                Opcode::I32Const(22), // 6
                Opcode::Return, // 7
                Opcode::I32Const(33), // 8
                Opcode::Return // 9
            ]
        );

        //println!("{}", ee.to_string());

        let f: extern "C" fn (v: i64) -> i64 = unsafe {
            ::std::mem::transmute(ee.get_function_address(0))
        };
        assert_eq!(f(0) as i32, 11);
        assert_eq!(f(1) as i32, 22);
        assert_eq!(f(2) as i32, 33);
        assert_eq!(f(99) as i32, -1);
    }

    #[test]
    fn test_i32_wrap_i64() {
        let ee = build_ee_from_fn_body(
            Type::Func(vec! [ ValType::I64 ], vec! [ ValType::I32 ]),
            vec! [],
            vec! [
                Opcode::GetLocal(0),
                Opcode::I32WrapI64,
                Opcode::Return
            ]
        );

        //println!("{}", ee.to_string());

        let f: extern "C" fn (v: i64) -> i64 = unsafe {
            ::std::mem::transmute(ee.get_function_address(0))
        };
        let ret = f(939230566849);
        assert_eq!(ret as i32, -1367270975);
        assert_eq!(ret, 2927696321);
    }

    #[test]
    fn test_call() {
        let ee = build_ee_from_fn_bodies(
            vec! [
                (
                    Type::Func(vec! [ ValType::I32 ], vec! [ ValType::I32 ] ),
                    vec! [],
                    vec! [
                        Opcode::GetLocal(0),
                        Opcode::I32Const(5),
                        Opcode::Call(1),
                        Opcode::Return
                    ]
                ),
                (
                    Type::Func(vec! [ ValType::I32, ValType::I32 ], vec! [ ValType::I32 ] ),
                    vec! [],
                    vec! [
                        Opcode::GetLocal(0),
                        Opcode::GetLocal(1),
                        Opcode::GetLocal(1),
                        Opcode::I32Add,
                        Opcode::I32Add,
                        Opcode::Return // Local(0) + Local(1) * 2
                    ]
                )
            ]
        );

        println!("{}", ee.ee.to_string());

        let f: extern "C" fn (v: i64) -> i64 = unsafe {
            ::std::mem::transmute(ee.get_function_address(0))
        };
        assert_eq!(f(35), 45);
    }

    #[test]
    fn test_i32_load() {
        let ee = build_ee_from_fn_body(
            Type::Func(vec! [ ValType::I32 ], vec! [ ValType::I32 ]),
            vec! [],
            vec! [
                Opcode::I32Const(12),
                Opcode::GetLocal(0),
                Opcode::I32Store(Memarg {
                    offset: 4,
                    align: 0
                }),
                Opcode::I32Const(16),
                Opcode::I32Load(Memarg {
                    offset: 0,
                    align: 0
                }),
                Opcode::Return
            ]
        );

        //println!("{}", ee.to_string());

        let f: extern "C" fn (v: i64) -> i64 = unsafe {
            ::std::mem::transmute(ee.get_function_address(0))
        };
        assert_eq!(f(42), 42);
    }

    #[test]
    fn test_i32_load_8u() {
        let ee = build_ee_from_fn_body(
            Type::Func(vec! [ ValType::I32 ], vec! [ ValType::I32 ]),
            vec! [],
            vec! [
                Opcode::I32Const(12),
                Opcode::GetLocal(0),
                Opcode::I32Store(Memarg {
                    offset: 4,
                    align: 0
                }),
                Opcode::I32Const(16),
                Opcode::I32Load8U(Memarg {
                    offset: 0,
                    align: 0
                }),
                Opcode::Return
            ]
        );

        //println!("{}", ee.to_string());

        let f: extern "C" fn (v: i64) -> i64 = unsafe {
            ::std::mem::transmute(ee.get_function_address(0))
        };
        assert_eq!(f(51328519), 7);
        assert_eq!(f(6615), 215);
    }

    #[test]
    fn test_i32_load_8s() {
        let ee = build_ee_from_fn_body(
            Type::Func(vec! [ ValType::I32 ], vec! [ ValType::I32 ]),
            vec! [],
            vec! [
                Opcode::I32Const(12),
                Opcode::GetLocal(0),
                Opcode::I32Store(Memarg {
                    offset: 4,
                    align: 0
                }),
                Opcode::I32Const(16),
                Opcode::I32Load8S(Memarg {
                    offset: 0,
                    align: 0
                }),
                Opcode::Return
            ]
        );

        //println!("{}", ee.to_string());

        let f: extern "C" fn (v: i64) -> i64 = unsafe {
            ::std::mem::transmute(ee.get_function_address(0))
        };
        assert_eq!(f(51328519), 7);
        assert_eq!(f(6615) as u32, 4294967255);
    }

    #[test]
    fn test_i32_load_16u() {
        let ee = build_ee_from_fn_body(
            Type::Func(vec! [ ValType::I32 ], vec! [ ValType::I32 ]),
            vec! [],
            vec! [
                Opcode::I32Const(12),
                Opcode::GetLocal(0),
                Opcode::I32Store(Memarg {
                    offset: 4,
                    align: 0
                }),
                Opcode::I32Const(16),
                Opcode::I32Load16U(Memarg {
                    offset: 0,
                    align: 0
                }),
                Opcode::Return
            ]
        );

        //println!("{}", ee.to_string());

        let f: extern "C" fn (v: i64) -> i64 = unsafe {
            ::std::mem::transmute(ee.get_function_address(0))
        };
        assert_eq!(f(51328519), 13831);
        assert_eq!(f(3786093), 50541);
    }

    #[test]
    fn test_i32_load_16s() {
        let ee = build_ee_from_fn_body(
            Type::Func(vec! [ ValType::I32 ], vec! [ ValType::I32 ]),
            vec! [],
            vec! [
                Opcode::I32Const(12),
                Opcode::GetLocal(0),
                Opcode::I32Store(Memarg {
                    offset: 4,
                    align: 0
                }),
                Opcode::I32Const(16),
                Opcode::I32Load16S(Memarg {
                    offset: 0,
                    align: 0
                }),
                Opcode::Return
            ]
        );

        //println!("{}", ee.to_string());

        let f: extern "C" fn (v: i64) -> i64 = unsafe {
            ::std::mem::transmute(ee.get_function_address(0))
        };
        assert_eq!(f(51328519), 13831);
        assert_eq!(f(3786093) as u32, 4294952301);
    }

    #[test]
    fn test_current_memory() {
        let ee = build_ee_from_fn_body(
            Type::Func(vec! [ ], vec! [ ValType::I32 ]),
            vec! [],
            vec! [
                Opcode::CurrentMemory,
                Opcode::Return
            ]
        );

        //println!("{}", ee.to_string());

        let f: extern "C" fn () -> i64 = unsafe {
            ::std::mem::transmute(ee.get_function_address(0))
        };

        let default_rt_config = RuntimeConfig::default();
        assert_eq!(f(), (default_rt_config.mem_default as i64) / 65536);
    }

    #[test]
    fn test_grow_memory() {
        let ee = build_ee_from_fn_body(
            Type::Func(vec! [ ], vec! [ ValType::I32 ]),
            vec! [],
            vec! [
                Opcode::I32Const(1),
                Opcode::GrowMemory,
                Opcode::Return
            ]
        );

        //println!("{}", ee.to_string());

        let f: extern "C" fn () -> i64 = unsafe {
            ::std::mem::transmute(ee.get_function_address(0))
        };

        let default_rt_config = RuntimeConfig::default();
        let v1 = f();
        assert_eq!(v1, (default_rt_config.mem_default as i64) / 65536);
        let v2 = f();
        assert_eq!(v2, v1 + 1);
    }

    #[test]
    fn test_call_indirect() {
        let mut m = Module::default();
        m.tables.push(Table {
            min: 1,
            max: None,
            elements: vec! [ Some(1) ]
        });

        let m = prepare_module_from_fn_bodies(
            m,
            vec! [
                (
                    Type::Func(vec! [ ValType::I32 ], vec! [ ValType::I32 ] ),
                    vec! [],
                    vec! [
                        Opcode::GetLocal(0),
                        Opcode::I32Const(5),
                        Opcode::I32Const(0), // indirect_index = 0
                        Opcode::CallIndirect(1), // typeidx = 1
                        Opcode::Return
                    ]
                ),
                (
                    Type::Func(vec! [ ValType::I32, ValType::I32 ], vec! [ ValType::I32 ] ),
                    vec! [],
                    vec! [
                        Opcode::GetLocal(0),
                        Opcode::GetLocal(1),
                        Opcode::GetLocal(1),
                        Opcode::I32Add,
                        Opcode::I32Add,
                        Opcode::Return // Local(0) + Local(1) * 2
                    ]
                )
            ]
        );
        let ee = m.into_execution_context();

        println!("{}", ee.ee.to_string());

        let f: extern "C" fn (v: i64) -> i64 = unsafe {
            ::std::mem::transmute(ee.get_function_address(0))
        };
        assert_eq!(f(35), 45);
    }

    #[test]
    fn test_native_invoke() {
        use executor::{NativeResolver, NativeEntry};
        use value::Value;

        struct TestResolver;
        impl NativeResolver for TestResolver {
            fn resolve(&self, module: &str, field: &str) -> Option<NativeEntry> {
                if module == "env" && field == "test" {
                    Some(Box::new(|_, args| {
                        let a = args[0].get_i32().unwrap();
                        let b = args[1].get_i32().unwrap();
                        Ok(Some(Value::I32(a + b)))
                    }))
                } else {
                    None
                }
            }
        }

        let mut m = Module::default();
        m.natives.push(Native {
            module: "env".into(),
            field: "test".into(),
            typeidx: 1
        });

        let m = prepare_module_from_fn_bodies(
            m,
            vec! [
                (
                    Type::Func(vec! [ ValType::I32 ], vec! [ ValType::I32 ] ),
                    vec! [],
                    vec! [
                        Opcode::GetLocal(0),
                        Opcode::I32Const(5),
                        Opcode::NativeInvoke(0),
                        Opcode::Return
                    ]
                ),
                ( // dummy
                    Type::Func(vec! [ ValType::I32, ValType::I32 ], vec! [ ValType::I32 ] ),
                    vec! [],
                    vec! [ Opcode::Return ]
                )
            ]
        );
        m.rt.set_native_resolver(TestResolver);

        let ee = m.into_execution_context();

        println!("{}", ee.ee.to_string());

        let f: extern "C" fn (v: i64) -> i64 = unsafe {
            ::std::mem::transmute(ee.get_function_address(0))
        };
        assert_eq!(f(35), 40);
    }
}
