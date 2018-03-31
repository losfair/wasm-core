use std::rc::Rc;
use std::ops::Deref;
use std::os::raw::c_void;
use module::*;
use cfgraph::*;
use super::llvm;
use opcode::{Opcode, Memarg};
use super::runtime::{Runtime, RuntimeConfig};
use executor::NativeResolver;
use value::Value;
use super::compiler_intrinsics::CompilerIntrinsics;
use super::ondemand::Ondemand;

fn generate_function_name(id: usize) -> String {
    format!("Wfunc_{}", id)
}

pub struct Compiler<'a> {
    _context: llvm::Context,
    source_module: &'a Module,
    rt: Rc<Runtime>
}

struct FunctionId(usize);

pub struct CompiledModule {
    rt: Rc<Runtime>,
    source_module: Module
}

pub struct ExecutionContext {
    rt: Rc<Runtime>,
    source_module: Module
}

impl CompiledModule {
    pub fn into_execution_context(self) -> ExecutionContext {
        ExecutionContext::from_compiled_module(self)
    }
}

impl ExecutionContext {
    pub fn from_compiled_module(m: CompiledModule) -> ExecutionContext {
        ExecutionContext {
            rt: m.rt,
            source_module: m.source_module
        }
    }

    pub fn get_function_address(&self, id: usize) -> *const c_void {
        self.rt.get_function_addr(id)
    }

    pub fn execute(&self, id: usize, args: &[Value]) -> Option<Value> {
        let source_fn = &self.source_module.functions[id];
        let ty = &self.source_module.types[source_fn.typeidx as usize];
        let Type::Func(ref ty_args, ref ty_ret) = *ty;

        if ty_args.len() != args.len() {
            panic!("Argument length mismatch");
        }

        for t in ty_args {
            match *t {
                ValType::I32 | ValType::I64 => {},
                _ => panic!("Unsupported type in function signature: {:?}", t)
            }
        }

        if ty_ret.len() > 0 {
            match ty_ret[0] {
                ValType::I32 | ValType::I64 => {},
                _ => panic!("Unsupported function return type: {:?}", ty_ret[0])
            }
        }

        let target_addr = self.get_function_address(id);

        unsafe {
            let tmp_ctx = llvm::Context::new();
            let m = llvm::Module::new(&tmp_ctx, "".into());
            {
                let f = llvm::Function::new(
                    &tmp_ctx,
                    &m,
                    "eval_entry",
                    llvm::Type::function(
                        &tmp_ctx,
                        llvm::Type::int64(&tmp_ctx),
                        &[]
                    )
                );
                let initial_bb = llvm::BasicBlock::new(&f);
                let builder = initial_bb.builder();
                let call_arg_types: Vec<llvm::Type> = ty_args.iter().map(|v| v.to_llvm_type(&tmp_ctx)).collect();
                let target_fn = builder.build_cast(
                    llvm::LLVMOpcode::LLVMIntToPtr,
                    builder.build_const_int(
                        llvm::Type::int64(&tmp_ctx),
                        target_addr as usize as _,
                        false
                    ),
                    llvm::Type::pointer(
                        llvm::Type::function(
                            &tmp_ctx,
                            llvm::Type::int64(&tmp_ctx), // FIXME: Undefined behavior
                            &call_arg_types
                        )
                    )
                );
                let call_args: Vec<llvm::LLVMValueRef> = args.iter().map(|v| builder.build_const_int(
                    llvm::Type::int64(&tmp_ctx),
                    v.reinterpret_as_i64() as _,
                    false
                )).collect();
                let ret = builder.build_call_raw(
                    target_fn,
                    &call_args
                );
                builder.build_ret(ret);
            }
            m.verify();
            let ee = llvm::ExecutionEngine::new(m);
            let entry = ee.get_function_address("eval_entry").unwrap();
            let entry: extern "C" fn () -> i64 = ::std::mem::transmute(entry);
            let ret = entry();
            if ty_ret.len() > 0 {
                Some(Value::reinterpret_from_i64(ret, ty_ret[0]))
            } else {
                None
            }
        }
    }
}

impl<'a> Compiler<'a> {
    pub fn new(m: &'a Module) -> OptimizeResult<Compiler<'a>> {
        Self::with_runtime_config(m, RuntimeConfig::default())
    }

    pub fn with_runtime_config(m: &'a Module, cfg: RuntimeConfig) -> OptimizeResult<Compiler<'a>> {
        Self::with_runtime(m, Rc::new(Runtime::new(cfg, m.clone())))
    }

    fn with_runtime(m: &'a Module, rt: Rc<Runtime>) -> OptimizeResult<Compiler<'a>> {
        Ok(Compiler {
            _context: llvm::Context::new(),
            source_module: m,
            rt: rt
        })
    }

    pub fn set_native_resolver<R: NativeResolver>(&self, resolver: R) {
        self.rt.set_native_resolver(resolver);
    }

    pub fn compile(&self) -> OptimizeResult<CompiledModule> {
        let target_functions: Vec<llvm::Module> = self.source_module.functions.iter().enumerate()
            .map(|(i, f)| {
                let target_module = llvm::Module::new(&self._context, "".into());
                let intrinsics = CompilerIntrinsics::new(&self._context, &target_module, &*self.rt);

                Self::gen_function_body(
                    &self._context,
                    &*self.rt,
                    &intrinsics,
                    self.source_module,
                    &target_module,
                    f,
                    "entry"
                );
                target_module
            })
            .collect();

        self.rt.set_ondemand(Rc::new(Ondemand::new(
            self.rt.clone(),
            self._context.clone(),
            target_functions
        )));
        Ok(CompiledModule {
            rt: self.rt.clone(),
            source_module: self.source_module.clone()
        })
    }

    fn gen_function_body(
        ctx: &llvm::Context,
        rt: &Runtime,
        intrinsics: &CompilerIntrinsics,
        source_module: &Module,
        target_module: &llvm::Module,
        source_func: &Function,
        target_function_name: &str
    ) -> llvm::Function {
        extern "C" fn _grow_memory(rt: &Runtime, len_inc: usize) {
            rt.grow_memory(len_inc);
        }

        const STACK_SIZE: usize = 32;
        const TAG_I32: i32 = 0x01;
        const TAG_I64: i32 = 0x02;

        let source_func_ty = &source_module.types[source_func.typeidx as usize];
        let Type::Func(ref source_func_args_ty, ref source_func_ret_ty) = *source_func_ty;

        let target_func = llvm::Function::new(
            ctx,
            target_module,
            target_function_name,
            source_func_ty.to_llvm_function_type(ctx)
        );

        let source_cfg = CFGraph::from_function(&source_func.body.opcodes).unwrap();
        source_cfg.validate().unwrap();

        let get_stack_elem_type = || llvm::Type::int64(ctx);
        let get_stack_array_type = || llvm::Type::array(
            ctx,
            get_stack_elem_type(),
            STACK_SIZE
        );

        let initializer_bb = llvm::BasicBlock::new(&target_func);
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

            for i in 0..locals_ty_info.len() {
                builder.build_store(
                    builder.build_const_int(
                        llvm::Type::int64(ctx),
                        0,
                        false
                    ),
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

        let build_stack_push_f32 = |builder: &llvm::Builder, v: llvm::LLVMValueRef| {
            unsafe {
                build_stack_push_i32(
                    builder,
                    builder.build_bitcast(
                        v,
                        llvm::Type::int32(ctx)
                    )
                )
            }
        };
        let build_stack_pop_f32 = |builder: &llvm::Builder| -> llvm::LLVMValueRef {
            unsafe {
                builder.build_bitcast(
                    build_stack_pop_i32(builder),
                    llvm::Type::float32(ctx)
                )
            }
        };

        let build_stack_push_f64 = |builder: &llvm::Builder, v: llvm::LLVMValueRef| {
            unsafe {
                build_stack_push(
                    builder,
                    builder.build_bitcast(
                        v,
                        llvm::Type::int64(ctx)
                    )
                )
            }
        };
        let build_stack_pop_f64 = |builder: &llvm::Builder| -> llvm::LLVMValueRef {
            unsafe {
                builder.build_bitcast(
                    build_stack_pop(builder),
                    llvm::Type::float64(ctx)
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

        let build_i32_unop = |
            builder: &llvm::Builder,
            op: &Fn(&llvm::Builder, llvm::LLVMValueRef) -> llvm::LLVMValueRef
        | {
            unsafe {
                let v = build_stack_pop_i32(&builder);
                build_stack_push(
                    &builder,
                    builder.build_cast(
                        llvm::LLVMOpcode::LLVMZExt,
                        op(builder, v),
                        llvm::Type::int64(ctx)
                    )
                );
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

        let build_i64_unop = |
            builder: &llvm::Builder,
            op: &Fn(&llvm::Builder, llvm::LLVMValueRef) -> llvm::LLVMValueRef
        | {
            unsafe {
                let v = build_stack_pop(&builder);
                build_stack_push(
                    &builder,
                    builder.build_cast(
                        llvm::LLVMOpcode::LLVMZExt,
                        op(builder, v),
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

        let build_f32_unop = |
            builder: &llvm::Builder,
            op: &Fn(&llvm::Builder, llvm::LLVMValueRef) -> llvm::LLVMValueRef
        | {
            unsafe {
                let v = build_stack_pop_f32(&builder);
                build_stack_push_f32(
                    &builder,
                    op(builder, v)
                );
            }
        };

        let build_f32_binop = |
            builder: &llvm::Builder,
            op: &Fn(&llvm::Builder, llvm::LLVMValueRef, llvm::LLVMValueRef) -> llvm::LLVMValueRef
        | {
            unsafe {
                let b = build_stack_pop_f32(&builder);
                let a = build_stack_pop_f32(&builder);
                build_stack_push_f32(
                    &builder,
                    op(builder, a, b)
                );
            }
        };

        let build_f32_relop = |
            builder: &llvm::Builder,
            op: &Fn(&llvm::Builder, llvm::LLVMValueRef, llvm::LLVMValueRef) -> llvm::LLVMValueRef
        | {
            unsafe {
                let b = build_stack_pop_f32(&builder);
                let a = build_stack_pop_f32(&builder);
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

        let build_f64_unop = |
            builder: &llvm::Builder,
            op: &Fn(&llvm::Builder, llvm::LLVMValueRef) -> llvm::LLVMValueRef
        | {
            unsafe {
                let v = build_stack_pop_f64(&builder);
                build_stack_push_f64(
                    &builder,
                    op(builder, v)
                );
            }
        };

        let build_f64_binop = |
            builder: &llvm::Builder,
            op: &Fn(&llvm::Builder, llvm::LLVMValueRef, llvm::LLVMValueRef) -> llvm::LLVMValueRef
        | {
            unsafe {
                let b = build_stack_pop_f64(&builder);
                let a = build_stack_pop_f64(&builder);
                build_stack_push_f64(
                    &builder,
                    op(builder, a, b)
                );
            }
        };

        let build_f64_relop = |
            builder: &llvm::Builder,
            op: &Fn(&llvm::Builder, llvm::LLVMValueRef, llvm::LLVMValueRef) -> llvm::LLVMValueRef
        | {
            unsafe {
                let b = build_stack_pop_f64(&builder);
                let a = build_stack_pop_f64(&builder);
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

        let build_get_function_addr = |
            builder: &llvm::Builder,
            id: llvm::LLVMValueRef
        | -> llvm::LLVMValueRef {
            unsafe {
                builder.build_call(
                    &intrinsics.get_function_addr,
                    &[
                        id
                    ]
                )
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
            .map(|_| llvm::BasicBlock::new(&target_func))
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
                            let ft = &source_module.types[f.typeidx as usize];
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

                            let real_addr = build_get_function_addr(
                                &builder,
                                builder.build_const_int(
                                    llvm::Type::int64(ctx),
                                    id as _,
                                    false
                                )
                            );
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
                        Opcode::Unreachable => {
                            let builder = target_bb.builder();
                            builder.build_call(
                                &intrinsics.checked_unreachable,
                                &[]
                            );
                        },
                        Opcode::GetGlobal(id) => {
                            let id = id as usize;
                            if id >= source_module.globals.len() {
                                panic!("Global index out of bound");
                            }
                            let builder = target_bb.builder();

                            let jit_info = &mut *rt.get_jit_info();
                            let global_begin_ptr = builder.build_cast(
                                llvm::LLVMOpcode::LLVMIntToPtr,
                                builder.build_const_int(
                                    llvm::Type::int64(ctx),
                                    &mut jit_info.global_begin as *mut *mut i64 as usize as _,
                                    false
                                ),
                                llvm::Type::pointer(
                                    llvm::Type::pointer(
                                        llvm::Type::int64(ctx)
                                    )
                                )
                            );

                            let global_begin = builder.build_load(global_begin_ptr);
                            let val = builder.build_load(
                                builder.build_gep(
                                    global_begin,
                                    &[
                                        builder.build_const_int(
                                            llvm::Type::int64(ctx),
                                            id as _,
                                            false
                                        )
                                    ]
                                )
                            );
                            build_stack_push(&builder, val);
                        },
                        Opcode::SetGlobal(id) => {
                            let id = id as usize;
                            if id >= source_module.globals.len() {
                                panic!("Global index out of bound");
                            }
                            let builder = target_bb.builder();

                            let val = build_stack_pop(&builder);

                            let jit_info = &mut *rt.get_jit_info();
                            let global_begin_ptr = builder.build_cast(
                                llvm::LLVMOpcode::LLVMIntToPtr,
                                builder.build_const_int(
                                    llvm::Type::int64(ctx),
                                    &mut jit_info.global_begin as *mut *mut i64 as usize as _,
                                    false
                                ),
                                llvm::Type::pointer(
                                    llvm::Type::pointer(
                                        llvm::Type::int64(ctx)
                                    )
                                )
                            );

                            let global_begin = builder.build_load(global_begin_ptr);
                            builder.build_store(
                                val,
                                builder.build_gep(
                                    global_begin,
                                    &[
                                        builder.build_const_int(
                                            llvm::Type::int64(ctx),
                                            id as _,
                                            false
                                        )
                                    ]
                                )
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
                        Opcode::I32Popcnt => {
                            build_i32_unop(
                                &target_bb.builder(),
                                &|t, v| t.build_call(
                                    &intrinsics.popcnt_i32,
                                    &[
                                        v
                                    ]
                                )
                            );
                        },
                        Opcode::I32Clz => {
                            build_i32_unop(
                                &target_bb.builder(),
                                &|t, v| t.build_call(
                                    &intrinsics.clz_i32,
                                    &[
                                        v,
                                        t.build_const_int(
                                            llvm::Type::int1(ctx),
                                            0,
                                            false
                                        )
                                    ]
                                )
                            );
                        },
                        Opcode::I32Ctz => {
                            build_i32_unop(
                                &target_bb.builder(),
                                &|t, v| t.build_call(
                                    &intrinsics.ctz_i32,
                                    &[
                                        v,
                                        t.build_const_int(
                                            llvm::Type::int1(ctx),
                                            0,
                                            false
                                        )
                                    ]
                                )
                            );
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
                        Opcode::I32And => {
                            build_i32_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_and(a, b)
                            );
                        },
                        Opcode::I32Or => {
                            build_i32_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_or(a, b)
                            );
                        },
                        Opcode::I32Xor => {
                            build_i32_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_xor(a, b)
                            );
                        },
                        Opcode::I32Rotl => {
                            build_i32_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_call(
                                    &intrinsics.rotl_i32,
                                    &[a, b]
                                )
                            );
                        },
                        Opcode::I32Rotr => {
                            build_i32_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_call(
                                    &intrinsics.rotr_i32,
                                    &[a, b]
                                )
                            );
                        },
                        Opcode::I32Eqz => {
                            let builder = target_bb.builder();
                            let v = builder.build_icmp(
                                llvm::LLVMIntEQ,
                                build_stack_pop_i32(&builder),
                                builder.build_const_int(
                                    llvm::Type::int32(ctx),
                                    0,
                                    false
                                )
                            );
                            build_stack_push_i32(&builder, v);
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
                        Opcode::I64Popcnt => {
                            build_i64_unop(
                                &target_bb.builder(),
                                &|t, v| t.build_call(
                                    &intrinsics.popcnt_i64,
                                    &[
                                        v
                                    ]
                                )
                            );
                        },
                        Opcode::I64Clz => {
                            build_i64_unop(
                                &target_bb.builder(),
                                &|t, v| t.build_call(
                                    &intrinsics.clz_i64,
                                    &[
                                        v,
                                        t.build_const_int(
                                            llvm::Type::int1(ctx),
                                            0,
                                            false
                                        )
                                    ]
                                )
                            );
                        },
                        Opcode::I64Ctz => {
                            build_i64_unop(
                                &target_bb.builder(),
                                &|t, v| t.build_call(
                                    &intrinsics.ctz_i64,
                                    &[
                                        v,
                                        t.build_const_int(
                                            llvm::Type::int1(ctx),
                                            0,
                                            false
                                        )
                                    ]
                                )
                            );
                        },
                        Opcode::I64Add => {
                            build_i64_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_add(a, b)
                            );
                        },
                        Opcode::I64Sub => {
                            build_i64_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_sub(a, b)
                            );
                        },
                        Opcode::I64Mul => {
                            build_i64_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_mul(a, b)
                            );
                        },
                        Opcode::I64DivU => {
                            build_i64_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_udiv(a, b)
                            );
                        },
                        Opcode::I64DivS => {
                            build_i64_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_sdiv(a, b)
                            );
                        },
                        Opcode::I64RemU => {
                            build_i64_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_urem(a, b)
                            );
                        },
                        Opcode::I64RemS => {
                            build_i64_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_srem(a, b)
                            );
                        },
                        Opcode::I64Shl => {
                            build_i64_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_shl(a, b)
                            );
                        },
                        Opcode::I64ShrU => {
                            build_i64_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_lshr(a, b)
                            );
                        },
                        Opcode::I64ShrS => {
                            build_i64_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_ashr(a, b)
                            );
                        },
                        Opcode::I64And => {
                            build_i64_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_and(a, b)
                            );
                        },
                        Opcode::I64Or => {
                            build_i64_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_or(a, b)
                            );
                        },
                        Opcode::I64Xor => {
                            build_i64_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_xor(a, b)
                            );
                        },
                        Opcode::I64Rotl => {
                            build_i64_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_call(
                                    &intrinsics.rotl_i64,
                                    &[a, b]
                                )
                            );
                        },
                        Opcode::I64Rotr => {
                            build_i64_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_call(
                                    &intrinsics.rotr_i64,
                                    &[a, b]
                                )
                            );
                        },
                        Opcode::I64Eqz => {
                            let builder = target_bb.builder();
                            let v = builder.build_icmp(
                                llvm::LLVMIntEQ,
                                build_stack_pop(&builder),
                                builder.build_const_int(
                                    llvm::Type::int64(ctx),
                                    0,
                                    false
                                )
                            );
                            build_stack_push_i32(&builder, v);
                        },
                        Opcode::I64Eq => {
                            build_i64_relop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_icmp(
                                    llvm::LLVMIntEQ,
                                    a,
                                    b
                                )
                            );
                        },
                        Opcode::I64Ne => {
                            build_i64_relop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_icmp(
                                    llvm::LLVMIntNE,
                                    a,
                                    b
                                )
                            );
                        },
                        Opcode::I64LtU => {
                            build_i64_relop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_icmp(
                                    llvm::LLVMIntULT,
                                    a,
                                    b
                                )
                            );
                        },
                        Opcode::I64LtS => {
                            build_i64_relop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_icmp(
                                    llvm::LLVMIntSLT,
                                    a,
                                    b
                                )
                            );
                        },
                        Opcode::I64LeU => {
                            build_i64_relop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_icmp(
                                    llvm::LLVMIntULE,
                                    a,
                                    b
                                )
                            );
                        },
                        Opcode::I64LeS => {
                            build_i64_relop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_icmp(
                                    llvm::LLVMIntSLE,
                                    a,
                                    b
                                )
                            );
                        },
                        Opcode::I64GtU => {
                            build_i64_relop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_icmp(
                                    llvm::LLVMIntUGT,
                                    a,
                                    b
                                )
                            );
                        },
                        Opcode::I64GtS => {
                            build_i64_relop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_icmp(
                                    llvm::LLVMIntSGT,
                                    a,
                                    b
                                )
                            );
                        },
                        Opcode::I64GeU => {
                            build_i64_relop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_icmp(
                                    llvm::LLVMIntUGE,
                                    a,
                                    b
                                )
                            );
                        },
                        Opcode::I64GeS => {
                            build_i64_relop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_icmp(
                                    llvm::LLVMIntSGE,
                                    a,
                                    b
                                )
                            );
                        },
                        Opcode::I64ExtendI32U => {
                            let builder = target_bb.builder();
                            build_stack_push(
                                &builder,
                                builder.build_cast(
                                    llvm::LLVMOpcode::LLVMZExt,
                                    build_stack_pop_i32(&builder),
                                    llvm::Type::int64(ctx)
                                )
                            );
                        },
                        Opcode::I64ExtendI32S => {
                            let builder = target_bb.builder();
                            build_stack_push(
                                &builder,
                                builder.build_cast(
                                    llvm::LLVMOpcode::LLVMSExt,
                                    build_stack_pop_i32(&builder),
                                    llvm::Type::int64(ctx)
                                )
                            );
                        },
                        Opcode::I64Load(ref m) => {
                            let builder = target_bb.builder();
                            let addr = build_stack_pop(&builder);
                            build_stack_push(&builder, build_std_load(
                                &builder,
                                8,
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
                        Opcode::I64Load8U(ref m) => {
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
                        Opcode::I64Load8S(ref m) => {
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
                        Opcode::I64Load16U(ref m) => {
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
                        Opcode::I64Load16S(ref m) => {
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
                        Opcode::I64Load32U(ref m) => {
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
                        Opcode::I64Load32S(ref m) => {
                            let builder = target_bb.builder();
                            let addr = build_stack_pop(&builder);
                            build_stack_push(&builder, build_std_load(
                                &builder,
                                4,
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
                        Opcode::I64Store(ref m) => {
                            let builder = target_bb.builder();
                            let val = build_stack_pop(&builder);
                            let addr = build_stack_pop(&builder);
                            build_std_store(
                                &builder,
                                8,
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
                        Opcode::I64Store8(ref m) => {
                            let builder = target_bb.builder();
                            let val = build_stack_pop(&builder);
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
                        Opcode::I64Store16(ref m) => {
                            let builder = target_bb.builder();
                            let val = build_stack_pop(&builder);
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
                        Opcode::I64Store32(ref m) => {
                            let builder = target_bb.builder();
                            let val = build_stack_pop(&builder);
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
                        Opcode::F32Const(v) => {
                            let builder = target_bb.builder();
                            build_stack_push_i32(
                                &builder,
                                builder.build_const_int(
                                    llvm::Type::int32(ctx),
                                    v as _,
                                    false
                                )
                            );
                        },
                        Opcode::F64Const(v) => {
                            let builder = target_bb.builder();
                            build_stack_push(
                                &builder,
                                builder.build_const_int(
                                    llvm::Type::int64(ctx),
                                    v as _,
                                    false
                                )
                            );
                        },
                        Opcode::I32ReinterpretF32 | Opcode::I64ReinterpretF64
                            | Opcode::F32ReinterpretI32 | Opcode::F64ReinterpretI64 => {
                                // no-op
                        },
                        Opcode::I32TruncSF32 => {
                            let builder = target_bb.builder();
                            build_stack_push_i32(
                                &builder,
                                builder.build_cast(
                                    llvm::LLVMOpcode::LLVMFPToSI,
                                    build_stack_pop_f32(&builder),
                                    llvm::Type::int32(ctx)
                                )
                            );
                        },
                        Opcode::I32TruncUF32 => {
                            let builder = target_bb.builder();
                            build_stack_push_i32(
                                &builder,
                                builder.build_cast(
                                    llvm::LLVMOpcode::LLVMFPToUI,
                                    build_stack_pop_f32(&builder),
                                    llvm::Type::int32(ctx)
                                )
                            );
                        },
                        Opcode::I32TruncSF64 => {
                            let builder = target_bb.builder();
                            build_stack_push_i32(
                                &builder,
                                builder.build_cast(
                                    llvm::LLVMOpcode::LLVMFPToSI,
                                    build_stack_pop_f64(&builder),
                                    llvm::Type::int32(ctx)
                                )
                            );
                        },
                        Opcode::I32TruncUF64 => {
                            let builder = target_bb.builder();
                            build_stack_push_i32(
                                &builder,
                                builder.build_cast(
                                    llvm::LLVMOpcode::LLVMFPToUI,
                                    build_stack_pop_f64(&builder),
                                    llvm::Type::int32(ctx)
                                )
                            );
                        },
                        Opcode::I64TruncSF32 => {
                            let builder = target_bb.builder();
                            build_stack_push(
                                &builder,
                                builder.build_cast(
                                    llvm::LLVMOpcode::LLVMFPToSI,
                                    build_stack_pop_f32(&builder),
                                    llvm::Type::int64(ctx)
                                )
                            );
                        },
                        Opcode::I64TruncUF32 => {
                            let builder = target_bb.builder();
                            build_stack_push(
                                &builder,
                                builder.build_cast(
                                    llvm::LLVMOpcode::LLVMFPToUI,
                                    build_stack_pop_f32(&builder),
                                    llvm::Type::int64(ctx)
                                )
                            );
                        },
                        Opcode::I64TruncSF64 => {
                            let builder = target_bb.builder();
                            build_stack_push(
                                &builder,
                                builder.build_cast(
                                    llvm::LLVMOpcode::LLVMFPToSI,
                                    build_stack_pop_f64(&builder),
                                    llvm::Type::int64(ctx)
                                )
                            );
                        },
                        Opcode::I64TruncUF64 => {
                            let builder = target_bb.builder();
                            build_stack_push(
                                &builder,
                                builder.build_cast(
                                    llvm::LLVMOpcode::LLVMFPToUI,
                                    build_stack_pop_f64(&builder),
                                    llvm::Type::int64(ctx)
                                )
                            );
                        },
                        Opcode::F32ConvertSI32 => {
                            let builder = target_bb.builder();
                            build_stack_push_f32(
                                &builder,
                                builder.build_cast(
                                    llvm::LLVMOpcode::LLVMSIToFP,
                                    build_stack_pop_i32(&builder),
                                    llvm::Type::float32(ctx)
                                )
                            );
                        },
                        Opcode::F32ConvertUI32 => {
                            let builder = target_bb.builder();
                            build_stack_push_f32(
                                &builder,
                                builder.build_cast(
                                    llvm::LLVMOpcode::LLVMUIToFP,
                                    build_stack_pop_i32(&builder),
                                    llvm::Type::float32(ctx)
                                )
                            );
                        },
                        Opcode::F32ConvertSI64 => {
                            let builder = target_bb.builder();
                            build_stack_push_f32(
                                &builder,
                                builder.build_cast(
                                    llvm::LLVMOpcode::LLVMSIToFP,
                                    build_stack_pop(&builder),
                                    llvm::Type::float32(ctx)
                                )
                            );
                        },
                        Opcode::F32ConvertUI64 => {
                            let builder = target_bb.builder();
                            build_stack_push_f32(
                                &builder,
                                builder.build_cast(
                                    llvm::LLVMOpcode::LLVMSIToFP,
                                    build_stack_pop(&builder),
                                    llvm::Type::float32(ctx)
                                )
                            );
                        },
                        Opcode::F64ConvertSI32 => {
                            let builder = target_bb.builder();
                            build_stack_push_f64(
                                &builder,
                                builder.build_cast(
                                    llvm::LLVMOpcode::LLVMSIToFP,
                                    build_stack_pop_i32(&builder),
                                    llvm::Type::float64(ctx)
                                )
                            );
                        },
                        Opcode::F64ConvertUI32 => {
                            let builder = target_bb.builder();
                            build_stack_push_f64(
                                &builder,
                                builder.build_cast(
                                    llvm::LLVMOpcode::LLVMUIToFP,
                                    build_stack_pop_i32(&builder),
                                    llvm::Type::float64(ctx)
                                )
                            );
                        },
                        Opcode::F64ConvertSI64 => {
                            let builder = target_bb.builder();
                            build_stack_push_f64(
                                &builder,
                                builder.build_cast(
                                    llvm::LLVMOpcode::LLVMSIToFP,
                                    build_stack_pop(&builder),
                                    llvm::Type::float64(ctx)
                                )
                            );
                        },
                        Opcode::F64ConvertUI64 => {
                            let builder = target_bb.builder();
                            build_stack_push_f64(
                                &builder,
                                builder.build_cast(
                                    llvm::LLVMOpcode::LLVMSIToFP,
                                    build_stack_pop(&builder),
                                    llvm::Type::float64(ctx)
                                )
                            );
                        },
                        Opcode::F32DemoteF64 => {
                            let builder = target_bb.builder();
                            build_stack_push_f32(
                                &builder,
                                builder.build_fp_trunc(
                                    build_stack_pop_f64(&builder),
                                    llvm::Type::float32(ctx)
                                )
                            );
                        },
                        Opcode::F64PromoteF32 => {
                            let builder = target_bb.builder();
                            build_stack_push_f64(
                                &builder,
                                builder.build_fp_ext(
                                    build_stack_pop_f32(&builder),
                                    llvm::Type::float64(ctx)
                                )
                            );
                        },
                        Opcode::F32Abs => {
                            build_f32_unop(
                                &target_bb.builder(),
                                &|t, v| t.build_call(
                                    &intrinsics.fabs_f32,
                                    &[ v ]
                                )
                            )
                        },
                        Opcode::F32Neg => {
                            build_f32_unop(
                                &target_bb.builder(),
                                &|t, v| t.build_fsub(
                                    t.build_bitcast(
                                        t.build_const_int(
                                            llvm::Type::int32(ctx),
                                            0,
                                            false
                                        ),
                                        llvm::Type::float32(ctx)
                                    ),
                                    v
                                )
                            );
                        },
                        Opcode::F32Ceil => {
                            build_f32_unop(
                                &target_bb.builder(),
                                &|t, v| t.build_call(
                                    &intrinsics.ceil_f32,
                                    &[ v ]
                                )
                            );
                        },
                        Opcode::F32Floor => {
                            build_f32_unop(
                                &target_bb.builder(),
                                &|t, v| t.build_call(
                                    &intrinsics.floor_f32,
                                    &[ v ]
                                )
                            );
                        },
                        Opcode::F32Trunc => {
                            build_f32_unop(
                                &target_bb.builder(),
                                &|t, v| t.build_call(
                                    &intrinsics.trunc_f32,
                                    &[ v ]
                                )
                            );
                        },
                        Opcode::F32Nearest => {
                            build_f32_unop(
                                &target_bb.builder(),
                                &|t, v| t.build_call(
                                    &intrinsics.nearest_f32,
                                    &[ v ]
                                )
                            );
                        },
                        Opcode::F32Sqrt => {
                            build_f32_unop(
                                &target_bb.builder(),
                                &|t, v| t.build_call(
                                    &intrinsics.sqrt_f32,
                                    &[ v ]
                                )
                            );
                        },
                        Opcode::F32Add => {
                            build_f32_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_fadd(a, b)
                            );
                        },
                        Opcode::F32Sub => {
                            build_f32_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_fsub(a, b)
                            );
                        },
                        Opcode::F32Mul => {
                            build_f32_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_fmul(a, b)
                            );
                        },
                        Opcode::F32Div => {
                            build_f32_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_fdiv(a, b)
                            );
                        },
                        Opcode::F32Min => {
                            build_f32_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_call(
                                    &intrinsics.minnum_f32,
                                    &[ a, b ]
                                )
                            );
                        },
                        Opcode::F32Max => {
                            build_f32_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_call(
                                    &intrinsics.maxnum_f32,
                                    &[ a, b ]
                                )
                            );
                        },
                        Opcode::F32Copysign => {
                            build_f32_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_call(
                                    &intrinsics.copysign_f32,
                                    &[ a, b ]
                                )
                            );
                        },
                        Opcode::F32Eq => {
                            build_f32_relop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_fcmp(
                                    llvm::LLVMRealOEQ,
                                    a, b
                                )
                            );
                        },
                        Opcode::F32Ne => {
                            build_f32_relop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_fcmp(
                                    llvm::LLVMRealONE,
                                    a, b
                                )
                            );
                        },
                        Opcode::F32Lt => {
                            build_f32_relop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_fcmp(
                                    llvm::LLVMRealOLT,
                                    a, b
                                )
                            );
                        },
                        Opcode::F32Gt => {
                            build_f32_relop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_fcmp(
                                    llvm::LLVMRealOGT,
                                    a, b
                                )
                            );
                        },
                        Opcode::F32Le => {
                            build_f32_relop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_fcmp(
                                    llvm::LLVMRealOLE,
                                    a, b
                                )
                            );
                        },
                        Opcode::F32Ge => {
                            build_f32_relop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_fcmp(
                                    llvm::LLVMRealOGE,
                                    a, b
                                )
                            );
                        },
                        Opcode::F64Abs => {
                            build_f64_unop(
                                &target_bb.builder(),
                                &|t, v| t.build_call(
                                    &intrinsics.fabs_f64,
                                    &[ v ]
                                )
                            )
                        },
                        Opcode::F64Neg => {
                            build_f64_unop(
                                &target_bb.builder(),
                                &|t, v| t.build_fsub(
                                    t.build_bitcast(
                                        t.build_const_int(
                                            llvm::Type::int64(ctx),
                                            0,
                                            false
                                        ),
                                        llvm::Type::float64(ctx)
                                    ),
                                    v
                                )
                            );
                        },
                        Opcode::F64Ceil => {
                            build_f64_unop(
                                &target_bb.builder(),
                                &|t, v| t.build_call(
                                    &intrinsics.ceil_f64,
                                    &[ v ]
                                )
                            );
                        },
                        Opcode::F64Floor => {
                            build_f64_unop(
                                &target_bb.builder(),
                                &|t, v| t.build_call(
                                    &intrinsics.floor_f64,
                                    &[ v ]
                                )
                            );
                        },
                        Opcode::F64Trunc => {
                            build_f64_unop(
                                &target_bb.builder(),
                                &|t, v| t.build_call(
                                    &intrinsics.trunc_f64,
                                    &[ v ]
                                )
                            );
                        },
                        Opcode::F64Nearest => {
                            build_f64_unop(
                                &target_bb.builder(),
                                &|t, v| t.build_call(
                                    &intrinsics.nearest_f64,
                                    &[ v ]
                                )
                            );
                        },
                        Opcode::F64Sqrt => {
                            build_f64_unop(
                                &target_bb.builder(),
                                &|t, v| t.build_call(
                                    &intrinsics.sqrt_f64,
                                    &[ v ]
                                )
                            );
                        },
                        Opcode::F64Add => {
                            build_f64_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_fadd(a, b)
                            );
                        },
                        Opcode::F64Sub => {
                            build_f64_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_fsub(a, b)
                            );
                        },
                        Opcode::F64Mul => {
                            build_f64_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_fmul(a, b)
                            );
                        },
                        Opcode::F64Div => {
                            build_f64_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_fdiv(a, b)
                            );
                        },
                        Opcode::F64Min => {
                            build_f64_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_call(
                                    &intrinsics.minnum_f64,
                                    &[ a, b ]
                                )
                            );
                        },
                        Opcode::F64Max => {
                            build_f64_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_call(
                                    &intrinsics.maxnum_f64,
                                    &[ a, b ]
                                )
                            );
                        },
                        Opcode::F64Copysign => {
                            build_f64_binop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_call(
                                    &intrinsics.copysign_f64,
                                    &[ a, b ]
                                )
                            );
                        },
                        Opcode::F64Eq => {
                            build_f64_relop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_fcmp(
                                    llvm::LLVMRealOEQ,
                                    a, b
                                )
                            );
                        },
                        Opcode::F64Ne => {
                            build_f64_relop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_fcmp(
                                    llvm::LLVMRealONE,
                                    a, b
                                )
                            );
                        },
                        Opcode::F64Lt => {
                            build_f64_relop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_fcmp(
                                    llvm::LLVMRealOLT,
                                    a, b
                                )
                            );
                        },
                        Opcode::F64Gt => {
                            build_f64_relop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_fcmp(
                                    llvm::LLVMRealOGT,
                                    a, b
                                )
                            );
                        },
                        Opcode::F64Le => {
                            build_f64_relop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_fcmp(
                                    llvm::LLVMRealOLE,
                                    a, b
                                )
                            );
                        },
                        Opcode::F64Ge => {
                            build_f64_relop(
                                &target_bb.builder(),
                                &|t, a, b| t.build_fcmp(
                                    llvm::LLVMRealOGE,
                                    a, b
                                )
                            );
                        },
                        Opcode::Memcpy | Opcode::NotImplemented(_)
                            | Opcode::I32Rotr
                            | Opcode::I64Rotl | Opcode::I64Rotr => {
                            // not implemented
                            eprintln!("Warning: Not implemented: {:?}", op);
                            let builder = target_bb.builder();
                            builder.build_call(
                                &intrinsics.checked_unreachable,
                                &[]
                            );
                        },
                        Opcode::Return
                            | Opcode::Jmp(_)
                            | Opcode::JmpIf(_)
                            | Opcode::JmpEither(_, _)
                            | Opcode::JmpTable(_, _) => unreachable!()
                        //_ => panic!("Opcode not implemented: {:?}", op)
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

        target_func
    }
}

impl ValType {
    // FIXME:
    // All stack/local values should be int64.
    // Therefore this is no longer required.
    fn to_llvm_type(&self, ctx: &llvm::Context) -> llvm::Type {
        match *self {
            ValType::I32 | ValType::I64
                | ValType::F32 | ValType::F64 => llvm::Type::int64(ctx),
            //ValType::I64 => llvm::Type::int64(ctx),
            //ValType::F32 => llvm::Type::float32(ctx),
            //ValType::F64 => llvm::Type::float64(ctx)
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
        let compiler = Compiler::new(&m).unwrap();
        let target_module = compiler.compile().unwrap();

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

        //println!("{}", ee.ee.to_string());

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

        //println!("{}", ee.ee.to_string());

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
    fn test_i32_ctz() {
        let ee = build_ee_from_fn_body(
            Type::Func(vec! [ ValType::I32 ], vec! [ ValType::I32 ]),
            vec! [],
            vec! [
                Opcode::GetLocal(0),
                Opcode::I32Ctz,
                Opcode::Return
            ]
        );

        //println!("{}", ee.to_string());

        let f: extern "C" fn (v: i64) -> i64 = unsafe {
            ::std::mem::transmute(ee.get_function_address(0))
        };
        assert_eq!(f(0), 32);
        assert_eq!(f(1), 0);
        assert_eq!(f(2), 1);
        assert_eq!(f(3), 0);
        assert_eq!(f(4), 2);
    }

    #[test]
    fn test_i32_clz() {
        let ee = build_ee_from_fn_body(
            Type::Func(vec! [ ValType::I32 ], vec! [ ValType::I32 ]),
            vec! [],
            vec! [
                Opcode::GetLocal(0),
                Opcode::I32Clz,
                Opcode::Return
            ]
        );

        //println!("{}", ee.to_string());

        let f: extern "C" fn (v: i64) -> i64 = unsafe {
            ::std::mem::transmute(ee.get_function_address(0))
        };
        assert_eq!(f(0), 32);
        assert_eq!(f(1), 31);
        assert_eq!(f(2), 30);
        assert_eq!(f(3), 30);
        assert_eq!(f(4), 29);
    }

    #[test]
    fn test_f32_add() {
        let ee = build_ee_from_fn_body(
            Type::Func(vec! [ ValType::F32, ValType::F32 ], vec! [ ValType::F32 ]),
            vec! [],
            vec! [
                Opcode::GetLocal(0),
                Opcode::GetLocal(1),
                Opcode::F32Add,
                Opcode::Return
            ]
        );

        //println!("{}", ee.to_string());

        let f: extern "C" fn (a: i64, b: i64) -> i64 = unsafe {
            ::std::mem::transmute(ee.get_function_address(0))
        };
        unsafe {
            assert_eq!(
                f(
                    ::std::mem::transmute::<f32, u32>(1.2) as i64,
                    ::std::mem::transmute::<f32, u32>(1.5) as i64
                ),
                ::std::mem::transmute::<f32, u32>(1.2 as f32 + 1.5 as f32) as i64
            )
        };
    }

    #[test]
    fn test_f32_neg() {
        let ee = build_ee_from_fn_body(
            Type::Func(vec! [ ValType::F32 ], vec! [ ValType::F32 ]),
            vec! [],
            vec! [
                Opcode::GetLocal(0),
                Opcode::F32Neg,
                Opcode::Return
            ]
        );

        //println!("{}", ee.to_string());

        let f: extern "C" fn (v: i64) -> i64 = unsafe {
            ::std::mem::transmute(ee.get_function_address(0))
        };
        unsafe {
            assert_eq!(
                f(
                    ::std::mem::transmute::<f32, u32>(1.2) as i64
                ),
                ::std::mem::transmute::<f32, u32>(-1.2 as f32) as i64
            )
        };
    }

    #[test]
    fn test_f32_sqrt() {
        let ee = build_ee_from_fn_body(
            Type::Func(vec! [ ValType::F32 ], vec! [ ValType::F32 ]),
            vec! [],
            vec! [
                Opcode::GetLocal(0),
                Opcode::F32Sqrt,
                Opcode::Return
            ]
        );

        //println!("{}", ee.to_string());

        let f: extern "C" fn (v: i64) -> i64 = unsafe {
            ::std::mem::transmute(ee.get_function_address(0))
        };
        unsafe {
            assert_eq!(
                f(
                    ::std::mem::transmute::<f32, u32>(1.2) as i64
                ),
                ::std::mem::transmute::<f32, u32>((1.2 as f32).sqrt()) as i64
            )
        };
    }

    #[test]
    fn test_i32_popcnt() {
        let ee = build_ee_from_fn_body(
            Type::Func(vec! [ ValType::I32 ], vec! [ ValType::I32 ]),
            vec! [],
            vec! [
                Opcode::GetLocal(0),
                Opcode::I32Popcnt,
                Opcode::Return
            ]
        );

        //println!("{}", ee.to_string());

        let f: extern "C" fn (v: i64) -> i64 = unsafe {
            ::std::mem::transmute(ee.get_function_address(0))
        };
        assert_eq!(f(0), 0);
        assert_eq!(f(1), 1);
        assert_eq!(f(2), 1);
        assert_eq!(f(3), 2);
        assert_eq!(f(4), 1);
    }

    #[test]
    fn test_i32_rotl() {
        let ee = build_ee_from_fn_body(
            Type::Func(vec! [ ValType::I32, ValType::I32 ], vec! [ ValType::I32 ]),
            vec! [],
            vec! [
                Opcode::GetLocal(0),
                Opcode::GetLocal(1),
                Opcode::I32Rotl,
                Opcode::Return
            ]
        );

        //println!("{}", ee.to_string());

        let f: extern "C" fn (a: i64, b: i64) -> i64 = unsafe {
            ::std::mem::transmute(ee.get_function_address(0))
        };
        assert_eq!(f(1, 0), 1);
        assert_eq!(f(1, 1), 2);
        assert_eq!(f(2, 0), 2);
        assert_eq!(f(2, 1), 4);
    }

    #[test]
    fn test_i32_rotr() {
        let ee = build_ee_from_fn_body(
            Type::Func(vec! [ ValType::I32, ValType::I32 ], vec! [ ValType::I32 ]),
            vec! [],
            vec! [
                Opcode::GetLocal(0),
                Opcode::GetLocal(1),
                Opcode::I32Rotr,
                Opcode::Return
            ]
        );

        //println!("{}", ee.to_string());

        let f: extern "C" fn (a: i64, b: i64) -> i64 = unsafe {
            ::std::mem::transmute(ee.get_function_address(0))
        };
        assert_eq!(f(1, 0), 1);
        assert_eq!(f(1, 1), 1 << 31);
        assert_eq!(f(2, 0), 2);
        assert_eq!(f(2, 1), 1);
    }

    #[test]
    fn test_i64_rotl() {
        let ee = build_ee_from_fn_body(
            Type::Func(vec! [ ValType::I64, ValType::I64 ], vec! [ ValType::I64 ]),
            vec! [],
            vec! [
                Opcode::GetLocal(0),
                Opcode::GetLocal(1),
                Opcode::I64Rotl,
                Opcode::Return
            ]
        );

        //println!("{}", ee.to_string());

        let f: extern "C" fn (a: i64, b: i64) -> i64 = unsafe {
            ::std::mem::transmute(ee.get_function_address(0))
        };
        assert_eq!(f(1, 0), 1);
        assert_eq!(f(1, 1), 2);
        assert_eq!(f(2, 0), 2);
        assert_eq!(f(2, 1), 4);
    }

    #[test]
    fn test_i64_rotr() {
        let ee = build_ee_from_fn_body(
            Type::Func(vec! [ ValType::I64, ValType::I64 ], vec! [ ValType::I64 ]),
            vec! [],
            vec! [
                Opcode::GetLocal(0),
                Opcode::GetLocal(1),
                Opcode::I64Rotr,
                Opcode::Return
            ]
        );

        //println!("{}", ee.to_string());

        let f: extern "C" fn (a: i64, b: i64) -> i64 = unsafe {
            ::std::mem::transmute(ee.get_function_address(0))
        };
        assert_eq!(f(1, 0), 1);
        assert_eq!(f(1, 1), 1 << 63);
        assert_eq!(f(2, 0), 2);
        assert_eq!(f(2, 1), 1);
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

        //println!("{}", ee.ee.to_string());

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

        //println!("{}", ee.ee.to_string());

        let f: extern "C" fn (v: i64) -> i64 = unsafe {
            ::std::mem::transmute(ee.get_function_address(0))
        };
        assert_eq!(f(35), 40);
    }

    #[test]
    fn test_globals() {
        use value::Value;

        let mut m = Module::default();
        m.globals.push(Global {
            value: Value::I32(0)
        });
        m.globals.push(Global {
            value: Value::I32(0)
        });

        let m = prepare_module_from_fn_bodies(
            m,
            vec! [
                (
                    Type::Func(vec! [ ValType::I32 ], vec! [ ] ),
                    vec! [],
                    vec! [
                        Opcode::GetLocal(0),
                        Opcode::GetLocal(0),
                        Opcode::SetGlobal(0),
                        Opcode::I32Const(1),
                        Opcode::I32Add,
                        Opcode::SetGlobal(1),
                        Opcode::Return
                    ]
                ),
                (
                    Type::Func(vec! [ ], vec! [ ValType::I32 ]),
                    vec! [],
                    vec! [
                        Opcode::GetGlobal(0),
                        Opcode::Return
                    ]
                ),
                (
                    Type::Func(vec! [ ], vec! [ ValType::I32 ]),
                    vec! [],
                    vec! [
                        Opcode::GetGlobal(1),
                        Opcode::Return
                    ]
                )
            ]
        );
        let ee = m.into_execution_context();

        //println!("{}", ee.ee.to_string());

        let setter: extern "C" fn (v: i64) = unsafe {
            ::std::mem::transmute(ee.get_function_address(0))
        };
        let getter0: extern "C" fn () -> i64 = unsafe {
            ::std::mem::transmute(ee.get_function_address(1))
        };
        let getter1: extern "C" fn () -> i64 = unsafe {
            ::std::mem::transmute(ee.get_function_address(2))
        };
        setter(42);
        assert_eq!(getter0(), 42);
        assert_eq!(getter1(), 43);
    }

    #[test]
    fn test_unreachable() {
        let ee = build_ee_from_fn_body(
            Type::Func(vec! [ ], vec! [ ]),
            vec! [],
            vec! [
                Opcode::Unreachable,
                Opcode::Return
            ]
        );

        //println!("{}", ee.to_string());

        let f: extern "C" fn () = unsafe {
            ::std::mem::transmute(ee.get_function_address(0))
        };
        let ret = catch_unwind(AssertUnwindSafe(|| f()));
        match ret {
            Ok(_) => panic!("Expecting panic"),
            Err(_) => {}
        }
    }
}
