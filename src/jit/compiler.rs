use std::rc::Rc;
use std::ops::Deref;
use std::os::raw::c_void;
use std::collections::{BTreeSet, BTreeMap};
use optimizers::RemoveDeadBasicBlocks;
use module::*;
use cfgraph::*;
use super::llvm;
use ssa::{FlowGraph, Opcode, BlockId, ValueId};
use opcode::Memarg;
use super::runtime::{Runtime, RuntimeConfig, NativeInvokeRequest};
use executor::NativeResolver;
use value::Value;
use super::compiler_intrinsics::CompilerIntrinsics;
use super::ondemand::Ondemand;

pub fn generate_function_name(id: usize) -> String {
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
    pub rt: Rc<Runtime>,
    source_module: Module
}

impl CompiledModule {
    pub fn into_execution_context(self) -> ExecutionContext {
        ExecutionContext::from_compiled_module(self)
    }
}

pub unsafe trait JitFuncTy {
    fn has_ret() -> bool;
    fn n_args() -> usize;
    unsafe fn build(raw: *const c_void) -> Self;
}

macro_rules! unsafe_impl_jit_func_ty {
    ($sig:ty, $has_ret:expr, $n_args:expr) => {
        unsafe impl JitFuncTy for $sig {
            fn has_ret() -> bool { $has_ret }
            fn n_args() -> usize { $n_args }
            unsafe fn build(raw: *const c_void) -> Self {
                ::std::mem::transmute(raw)
            }
        }
    }
}

unsafe_impl_jit_func_ty!(extern "C" fn (), false, 0);
unsafe_impl_jit_func_ty!(extern "C" fn () -> i64, true, 0);
unsafe_impl_jit_func_ty!(extern "C" fn (i64), false, 1);
unsafe_impl_jit_func_ty!(extern "C" fn (i64) -> i64, true, 1);
unsafe_impl_jit_func_ty!(extern "C" fn (i64, i64), false, 2);
unsafe_impl_jit_func_ty!(extern "C" fn (i64, i64) -> i64, true, 2);
unsafe_impl_jit_func_ty!(extern "C" fn (i64, i64, i64), false, 3);
unsafe_impl_jit_func_ty!(extern "C" fn (i64, i64, i64) -> i64, true, 3);
unsafe_impl_jit_func_ty!(extern "C" fn (i64, i64, i64, i64), false, 4);
unsafe_impl_jit_func_ty!(extern "C" fn (i64, i64, i64, i64) -> i64, true, 4);
unsafe_impl_jit_func_ty!(extern "C" fn (i64, i64, i64, i64, i64), false, 5);
unsafe_impl_jit_func_ty!(extern "C" fn (i64, i64, i64, i64, i64) -> i64, true, 5);
unsafe_impl_jit_func_ty!(extern "C" fn (i64, i64, i64, i64, i64, i64), false, 6);
unsafe_impl_jit_func_ty!(extern "C" fn (i64, i64, i64, i64, i64, i64) -> i64, true, 6);

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

    pub unsafe fn get_function_checked<F: JitFuncTy>(&self, id: usize) -> F {
        let has_ret = F::has_ret();
        let n_args = F::n_args();

        let source_fn = &self.source_module.functions[id];
        let ty = &self.source_module.types[source_fn.typeidx as usize];
        let Type::Func(ref ty_args, ref ty_ret) = *ty;

        if n_args != ty_args.len() {
            panic!("Argument length mismatch");
        }

        if ty_ret.len() > 0 {
            assert_eq!(ty_ret.len(), 1);
            if !has_ret {
                panic!("Expecting exactly one return value");
            }
        } else {
            if has_ret {
                panic!("Expecting no return values");
            }
        }

        F::build(self.get_function_address(id))
    }

    pub fn set_native_resolver<R: NativeResolver>(&self, resolver: R) {
        self.rt.set_native_resolver(resolver);
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
        let target_module = llvm::Module::new(&self._context, "".into());
        let intrinsics = CompilerIntrinsics::new(&self._context, &target_module, &*self.rt);

        let target_functions: Vec<llvm::Function> = self.source_module.functions.iter().enumerate()
            .map(|(i, f)| {
                Self::gen_function_def(
                    &self._context,
                    self.source_module,
                    &target_module,
                    f,
                    &generate_function_name(i)
                )
            })
            .collect();

        for (sf, tf) in self.source_module.functions.iter().zip(
            target_functions.iter()
         ) {
            Self::gen_function_body(
                &self._context,
                &*self.rt,
                &intrinsics,
                self.source_module,
                &target_module,
                &target_functions,
                sf,
                tf
            );
        }
        drop(target_functions);
        drop(intrinsics);

        if self.rt.opt_level > 0 {
            for _ in 0..2 {
                target_module.inline_with_threshold(500);
                target_module.optimize();
            }
        }

        self.rt.set_ondemand(Rc::new(Ondemand::new(
            self.rt.clone(),
            self._context.clone(),
            target_module
        )));
        Ok(CompiledModule {
            rt: self.rt.clone(),
            source_module: self.source_module.clone()
        })
    }

    fn gen_function_def(
        ctx: &llvm::Context,
        source_module: &Module,
        target_module: &llvm::Module,
        source_func: &Function,
        target_function_name: &str
    ) -> llvm::Function {
        let source_func_ty = &source_module.types[source_func.typeidx as usize];

        llvm::Function::new(
            ctx,
            target_module,
            target_function_name,
            source_func_ty.to_llvm_function_type(ctx)
        )
    }

    fn gen_function_body(
        ctx: &llvm::Context,
        rt: &Runtime,
        intrinsics: &CompilerIntrinsics,
        source_module: &Module,
        target_module: &llvm::Module,
        target_functions: &[llvm::Function],
        source_func: &Function,
        target_func: &llvm::Function
    ) {
        use platform::generic::MemoryManager;

        extern "C" fn _grow_memory(rt: &Runtime, len_inc: usize) {
            rt.grow_memory(len_inc);
        }

        let hints = unsafe { &*rt.mm.get() }.hints();

        const STACK_SIZE: usize = 32;
        const TAG_I32: i32 = 0x01;
        const TAG_I64: i32 = 0x02;

        let source_func_ty = &source_module.types[source_func.typeidx as usize];
        let Type::Func(ref source_func_args_ty, ref source_func_ret_ty) = *source_func_ty;

        let mut source_cfg = CFGraph::from_function(&source_func.body.opcodes).unwrap();
        source_cfg.validate().unwrap();

        // Required because unreachable basic blocks should never get code-generated.
        source_cfg.optimize(RemoveDeadBasicBlocks).unwrap();

        let fg = FlowGraph::from_cfg(&source_cfg, source_module);

        let mut ssa_values: BTreeMap<ValueId, llvm::LLVMValueRef> = BTreeMap::new();
        let ssa_block_ids: BTreeMap<ValueId, BlockId> = {
            let mut ret: BTreeMap<ValueId, BlockId> = BTreeMap::new();

            for (i, blk) in fg.blocks.iter().enumerate() {
                for (id, _) in &blk.ops {
                    if let Some(id) = *id {
                        ret.insert(id, BlockId(i));
                    }
                }
            }

            ret
        };

        let initializer_bb = llvm::BasicBlock::new(&target_func);
        let locals_base;
        let n_locals = source_func_args_ty.len() + source_func.locals.len();

        unsafe {
            let builder = initializer_bb.builder();
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

        let extract_f32 = |
            builder: &llvm::Builder,
            v: llvm::LLVMValueRef
        | -> llvm::LLVMValueRef {
            unsafe {
                builder.build_bitcast(
                    builder.build_cast(
                        llvm::LLVMOpcode::LLVMTrunc,
                        v,
                        llvm::Type::int32(ctx)
                    ),
                    llvm::Type::float32(ctx)
                )
            }
        };

        let extract_f64 = |
            builder: &llvm::Builder,
            v: llvm::LLVMValueRef
        | -> llvm::LLVMValueRef {
            unsafe {
                builder.build_bitcast(
                    v,
                    llvm::Type::float64(ctx)
                )
            }
        };

        let wrap_f32 = |
            builder: &llvm::Builder,
            v: llvm::LLVMValueRef
        | -> llvm::LLVMValueRef {
            unsafe {
                builder.build_cast(
                    llvm::LLVMOpcode::LLVMZExt,
                    builder.build_bitcast(
                        v,
                        llvm::Type::int32(ctx)
                    ),
                    llvm::Type::int64(ctx)
                )
            }
        };

        let wrap_f64 = |
            builder: &llvm::Builder,
            v: llvm::LLVMValueRef
        | -> llvm::LLVMValueRef {
            unsafe {
                builder.build_bitcast(
                    v,
                    llvm::Type::int64(ctx)
                )
            }
        };

        let build_i32_unop = |
            builder: &llvm::Builder,
            a: llvm::LLVMValueRef,
            op: &Fn(&llvm::Builder, llvm::LLVMValueRef) -> llvm::LLVMValueRef
        | -> llvm::LLVMValueRef {
            unsafe {
                let a = builder.build_cast(
                    llvm::LLVMOpcode::LLVMTrunc,
                    a,
                    llvm::Type::int32(ctx)
                );
                let v = op(builder, a);
                builder.build_cast(
                    llvm::LLVMOpcode::LLVMZExt,
                    v,
                    llvm::Type::int64(ctx)
                )
            }
        };

        let build_i32_binop = |
            builder: &llvm::Builder,
            a: llvm::LLVMValueRef,
            b: llvm::LLVMValueRef,
            op: &Fn(&llvm::Builder, llvm::LLVMValueRef, llvm::LLVMValueRef) -> llvm::LLVMValueRef
        | -> llvm::LLVMValueRef {
            unsafe {
                let a = builder.build_cast(
                    llvm::LLVMOpcode::LLVMTrunc,
                    a,
                    llvm::Type::int32(ctx)
                );
                let b = builder.build_cast(
                    llvm::LLVMOpcode::LLVMTrunc,
                    b,
                    llvm::Type::int32(ctx)
                );
                let v = op(builder, a, b);
                builder.build_cast(
                    llvm::LLVMOpcode::LLVMZExt,
                    v,
                    llvm::Type::int64(ctx)
                )
            }
        };

        let build_i32_relop = |
            builder: &llvm::Builder,
            a: llvm::LLVMValueRef,
            b: llvm::LLVMValueRef,
            op: &Fn(&llvm::Builder, llvm::LLVMValueRef, llvm::LLVMValueRef) -> llvm::LLVMValueRef
        | -> llvm::LLVMValueRef {
            unsafe {
                let a = builder.build_cast(
                    llvm::LLVMOpcode::LLVMTrunc,
                    a,
                    llvm::Type::int32(ctx)
                );
                let b = builder.build_cast(
                    llvm::LLVMOpcode::LLVMTrunc,
                    b,
                    llvm::Type::int32(ctx)
                );
                let v = op(builder, a, b);
                builder.build_cast(
                    llvm::LLVMOpcode::LLVMZExt,
                    v,
                    llvm::Type::int64(ctx)
                )
            }
        };

        let build_i64_unop = |
            builder: &llvm::Builder,
            a: llvm::LLVMValueRef,
            op: &Fn(&llvm::Builder, llvm::LLVMValueRef) -> llvm::LLVMValueRef
        | -> llvm::LLVMValueRef {
            op(builder, a)
        };

        let build_i64_binop = |
            builder: &llvm::Builder,
            a: llvm::LLVMValueRef,
            b: llvm::LLVMValueRef,
            op: &Fn(&llvm::Builder, llvm::LLVMValueRef, llvm::LLVMValueRef) -> llvm::LLVMValueRef
        | -> llvm::LLVMValueRef {
            op(builder, a, b)
        };

        let build_i64_relop = |
            builder: &llvm::Builder,
            a: llvm::LLVMValueRef,
            b: llvm::LLVMValueRef,
            op: &Fn(&llvm::Builder, llvm::LLVMValueRef, llvm::LLVMValueRef) -> llvm::LLVMValueRef
        | -> llvm::LLVMValueRef {
            unsafe {
                let v = op(builder, a, b);
                builder.build_cast(
                    llvm::LLVMOpcode::LLVMZExt,
                    v,
                    llvm::Type::int64(ctx)
                )
            }
        };

        let build_f32_unop = |
            builder: &llvm::Builder,
            a: llvm::LLVMValueRef,
            op: &Fn(&llvm::Builder, llvm::LLVMValueRef) -> llvm::LLVMValueRef
        | -> llvm::LLVMValueRef {
            let a = extract_f32(builder, a);
            let v = op(builder, a);
            wrap_f32(builder, v)
        };

        let build_f32_binop = |
            builder: &llvm::Builder,
            a: llvm::LLVMValueRef,
            b: llvm::LLVMValueRef,
            op: &Fn(&llvm::Builder, llvm::LLVMValueRef, llvm::LLVMValueRef) -> llvm::LLVMValueRef
        | -> llvm::LLVMValueRef {
            let a = extract_f32(builder, a);
            let b = extract_f32(builder, b);
            let v = op(builder, a, b);
            wrap_f32(builder, v)
        };

        let build_f32_relop = |
            builder: &llvm::Builder,
            a: llvm::LLVMValueRef,
            b: llvm::LLVMValueRef,
            op: &Fn(&llvm::Builder, llvm::LLVMValueRef, llvm::LLVMValueRef) -> llvm::LLVMValueRef
        | -> llvm::LLVMValueRef {
            unsafe {
                let a = extract_f32(builder, a);
                let b = extract_f32(builder, b);
                let v = op(builder, a, b);
                builder.build_cast(
                    llvm::LLVMOpcode::LLVMZExt,
                    v,
                    llvm::Type::int64(ctx)
                )
            }
        };

        let build_f64_unop = |
            builder: &llvm::Builder,
            a: llvm::LLVMValueRef,
            op: &Fn(&llvm::Builder, llvm::LLVMValueRef) -> llvm::LLVMValueRef
        | -> llvm::LLVMValueRef {
            let a = extract_f64(builder, a);
            let v = op(builder, a);
            wrap_f64(builder, v)
        };

        let build_f64_binop = |
            builder: &llvm::Builder,
            a: llvm::LLVMValueRef,
            b: llvm::LLVMValueRef,
            op: &Fn(&llvm::Builder, llvm::LLVMValueRef, llvm::LLVMValueRef) -> llvm::LLVMValueRef
        | -> llvm::LLVMValueRef {
            let a = extract_f64(builder, a);
            let b = extract_f64(builder, b);
            let v = op(builder, a, b);
            wrap_f64(builder, v)
        };

        let build_f64_relop = |
            builder: &llvm::Builder,
            a: llvm::LLVMValueRef,
            b: llvm::LLVMValueRef,
            op: &Fn(&llvm::Builder, llvm::LLVMValueRef, llvm::LLVMValueRef) -> llvm::LLVMValueRef
        | -> llvm::LLVMValueRef {
            unsafe {
                let a = extract_f64(builder, a);
                let b = extract_f64(builder, b);
                let v = op(builder, a, b);
                builder.build_cast(
                    llvm::LLVMOpcode::LLVMZExt,
                    v,
                    llvm::Type::int64(ctx)
                )
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

        let build_translate_pointer = |
            builder: &llvm::Builder,
            ptr: llvm::LLVMValueRef,
            trusted_len: usize
        | -> llvm::LLVMValueRef {
            unsafe {
                if !hints.needs_bounds_check && hints.static_start_address.is_some() {
                    builder.build_cast(
                        llvm::LLVMOpcode::LLVMIntToPtr,
                        builder.build_add(
                            builder.build_const_int(
                                llvm::Type::int64(ctx),
                                hints.static_start_address.unwrap() as _,
                                false
                            ),
                            builder.build_and(
                                ptr,
                                builder.build_const_int(
                                    llvm::Type::int64(ctx),
                                    hints.address_mask as _,
                                    false
                                )
                            )
                        ),
                        llvm::Type::pointer(llvm::Type::void(ctx))
                    )
                } else {
                    builder.build_call(
                        &intrinsics.translate_pointer,
                        &[
                            ptr,
                            builder.build_const_int(
                                llvm::Type::int64(ctx),
                                trusted_len as _,
                                false
                            )
                        ]
                    )
                }
            }
        };

        let build_std_load = |
            builder: &llvm::Builder,
            trusted_len: usize,
            signed: bool,
            ptr: llvm::LLVMValueRef
        | -> llvm::LLVMValueRef {
            unsafe {
                let real_addr = build_translate_pointer(builder, ptr, trusted_len);
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
                let real_addr = build_translate_pointer(builder, ptr, trusted_len);
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

        let mut br_unreachable: BTreeSet<usize> = BTreeSet::new();

        for (i, bb) in fg.blocks.iter().enumerate() {
            let target_bb = &target_basic_blocks[i];

            for (ssa_target, op) in &bb.ops {
                unsafe {
                    match *op {
                        Opcode::Select(cond, if_true, if_false) => {
                            let builder = target_bb.builder();
                            let v = builder.build_call(
                                &intrinsics.select,
                                &[
                                    *ssa_values.get(&cond).unwrap(),
                                    *ssa_values.get(&if_true).unwrap(),
                                    *ssa_values.get(&if_false).unwrap()
                                ]
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::Call(id, ref args) => {
                            let builder = target_bb.builder();

                            let f = &source_module.functions[id as usize];
                            let ft = &source_module.types[f.typeidx as usize];
                            let Type::Func(ref ft_args, ref ft_ret) = *ft;

                            assert_eq!(ft_args.len(), args.len());

                            let call_args: Vec<llvm::LLVMValueRef> = args.iter()
                                .map(|v| *ssa_values.get(v).unwrap())
                                .collect();

                            let ret = builder.build_call(
                                &target_functions[id as usize],
                                &call_args
                            );

                            if ft_ret.len() > 0 {
                                ssa_values.insert(ssa_target.unwrap(), ret);
                            } else {
                                assert!(ssa_target.is_none());
                            }
                        },
                        Opcode::CallIndirect(typeidx, indirect_index_val, ref args) => {
                            let builder = target_bb.builder();

                            let indirect_index = *ssa_values.get(&indirect_index_val).unwrap();

                            let ft = &source_module.types[typeidx as usize];
                            let Type::Func(ref ft_args, ref ft_ret) = *ft;

                            let call_args: Vec<llvm::LLVMValueRef> = args.iter()
                                .map(|v| *ssa_values.get(v).unwrap())
                                .collect();

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
                                ssa_values.insert(ssa_target.unwrap(), ret);
                            } else {
                                assert!(ssa_target.is_none());
                            }
                        },
                        Opcode::NativeInvoke(id, ref args) => {
                            let builder = target_bb.builder();

                            let native = &source_module.natives[id as usize];

                            let ft = &source_module.types[native.typeidx as usize];
                            let Type::Func(ref ft_args, ref ft_ret) = *ft;

                            let call_args: Vec<llvm::LLVMValueRef> = args.iter()
                                .map(|v| *ssa_values.get(v).unwrap())
                                .collect();

                            let raw_stack_state = builder.build_call(
                                &intrinsics.stacksave,
                                &[]
                            );

                            let nir_place = builder.build_alloca(
                                llvm::Type::array(
                                    ctx,
                                    llvm::Type::int8(ctx),
                                    ::std::mem::size_of::<NativeInvokeRequest>()
                                )
                            );
                            let nir_place = builder.build_bitcast(
                                nir_place,
                                llvm::Type::pointer(llvm::Type::int8(ctx))
                            );

                            builder.build_call_raw(
                                builder.build_cast(
                                    llvm::LLVMOpcode::LLVMIntToPtr,
                                    builder.build_const_int(
                                        llvm::Type::int64(ctx),
                                        (Runtime::_jit_native_invoke_request as usize) as u64,
                                        false
                                    ),
                                    llvm::Type::pointer(llvm::Type::function(
                                        ctx,
                                        llvm::Type::void(ctx),
                                        &[
                                            llvm::Type::pointer(llvm::Type::int8(ctx)), // ret_place
                                            llvm::Type::int_native(ctx) // n_args
                                        ]
                                    ))
                                ),
                                &[
                                    nir_place,
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
                                                llvm::Type::pointer(llvm::Type::int8(ctx)), // req ptr
                                                llvm::Type::int64(ctx) // arg
                                            ]
                                        ))
                                    ),
                                    &[
                                        nir_place,
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
                                            llvm::Type::pointer(llvm::Type::int8(ctx)),
                                            llvm::Type::int_native(ctx), // rt
                                            llvm::Type::int_native(ctx) // id
                                        ]
                                    ))
                                ),
                                &[
                                    nir_place,
                                    builder.build_const_int(
                                        llvm::Type::int_native(ctx),
                                        rt as *const Runtime as usize as _,
                                        false
                                    ),
                                    builder.build_const_int(
                                        llvm::Type::int_native(ctx),
                                        id as _,
                                        false
                                    )
                                ]
                            );
                            builder.build_call(
                                &intrinsics.stackrestore,
                                &[
                                    raw_stack_state
                                ]
                            );
                            if ft_ret.len() > 0 {
                                ssa_values.insert(ssa_target.unwrap(), ret);
                            } else {
                                assert!(ssa_target.is_none());
                            }
                        },
                        Opcode::CurrentMemory => {
                            let builder = target_bb.builder();
                            ssa_values.insert(ssa_target.unwrap(), build_current_memory(&builder));
                        },
                        Opcode::GrowMemory(n_pages) => {
                            let builder = target_bb.builder();
                            let n_pages = *ssa_values.get(&n_pages).unwrap();
                            ssa_values.insert(ssa_target.unwrap(), build_grow_memory(&builder, n_pages));
                        },
                        Opcode::Unreachable => {
                            let builder = target_bb.builder();
                            builder.build_call(
                                &intrinsics.checked_unreachable,
                                &[]
                            );
                            br_unreachable.insert(i);
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

                            ssa_values.insert(ssa_target.unwrap(), val);
                        },
                        Opcode::SetGlobal(id, val) => {
                            let id = id as usize;
                            if id >= source_module.globals.len() {
                                panic!("Global index out of bound");
                            }
                            let builder = target_bb.builder();

                            let val = *ssa_values.get(&val).unwrap();

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
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::SetLocal(id, val) => {
                            let builder = target_bb.builder();
                            let v = *ssa_values.get(&val).unwrap();
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
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I32Popcnt(a) => {
                            let builder = target_bb.builder();
                            let a = *ssa_values.get(&a).unwrap();
                            let v = build_i32_unop(
                                &builder,
                                a,
                                &|t, v| t.build_call(&intrinsics.popcnt_i32, &[v])
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I32Clz(a) => {
                            let builder = target_bb.builder();
                            let a = *ssa_values.get(&a).unwrap();
                            let v = build_i32_unop(
                                &builder,
                                a,
                                &|t, v| t.build_call(&intrinsics.clz_i32, &[
                                    v,
                                    t.build_const_int(
                                        llvm::Type::int1(ctx),
                                        0,
                                        false
                                    )
                                ])
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I32Ctz(a) => {
                            let builder = target_bb.builder();
                            let a = *ssa_values.get(&a).unwrap();
                            let v = build_i32_unop(
                                &builder,
                                a,
                                &|t, v| t.build_call(&intrinsics.ctz_i32, &[
                                    v,
                                    t.build_const_int(
                                        llvm::Type::int1(ctx),
                                        0,
                                        false
                                    )
                                ])
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I32Add(a, b) => {
                            let v = build_i32_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_add(a, b)
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I32Sub(a, b) => {
                            let v = build_i32_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_sub(a, b)
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I32Mul(a, b) => {
                            let v = build_i32_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_mul(a, b)
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I32DivU(a, b) => {
                            let v = build_i32_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_udiv(a, b)
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I32DivS(a, b) => {
                            let v = build_i32_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_sdiv(a, b)
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I32RemU(a, b) => {
                            let v = build_i32_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_urem(a, b)
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I32RemS(a, b) => {
                            let v = build_i32_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_srem(a, b)
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        
                        Opcode::I32Shl(a, b) => {
                            let v = build_i32_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_shl(a, b)
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I32ShrU(a, b) => {
                            let v = build_i32_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_lshr(a, b)
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I32ShrS(a, b) => {
                            let v = build_i32_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_ashr(a, b)
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I32And(a, b) => {
                            let v = build_i32_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_and(a, b)
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I32Or(a, b) => {
                            let v = build_i32_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_or(a, b)
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I32Xor(a, b) => {
                            let v = build_i32_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_xor(a, b)
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I32Rotl(a, b) => {
                            let v = build_i32_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_call(
                                    &intrinsics.rotl_i32,
                                    &[a, b]
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I32Rotr(a, b) => {
                            let v = build_i32_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_call(
                                    &intrinsics.rotr_i32,
                                    &[a, b]
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I32Eqz(a) => {
                            let builder = target_bb.builder();
                            let v = builder.build_icmp(
                                llvm::LLVMIntEQ,
                                *ssa_values.get(&a).unwrap(),
                                builder.build_const_int(
                                    llvm::Type::int64(ctx),
                                    0,
                                    false
                                )
                            );
                            ssa_values.insert(
                                ssa_target.unwrap(),
                                builder.build_cast(
                                    llvm::LLVMOpcode::LLVMZExt,
                                    v,
                                    llvm::Type::int64(ctx)
                                )
                            );
                        },
                        Opcode::I32Eq(a, b) => {
                            let v = build_i32_relop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_icmp(
                                    llvm::LLVMIntEQ,
                                    a,
                                    b
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I32Ne(a, b) => {
                            let v = build_i32_relop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_icmp(
                                    llvm::LLVMIntNE,
                                    a,
                                    b
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I32LtU(a, b) => {
                            let v = build_i32_relop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_icmp(
                                    llvm::LLVMIntULT,
                                    a,
                                    b
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I32LtS(a, b) => {
                            let v = build_i32_relop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_icmp(
                                    llvm::LLVMIntSLT,
                                    a,
                                    b
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I32LeU(a, b) => {
                            let v = build_i32_relop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_icmp(
                                    llvm::LLVMIntULE,
                                    a,
                                    b
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I32LeS(a, b) => {
                            let v = build_i32_relop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_icmp(
                                    llvm::LLVMIntSLE,
                                    a,
                                    b
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I32GtU(a, b) => {
                            let v = build_i32_relop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_icmp(
                                    llvm::LLVMIntUGT,
                                    a,
                                    b
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I32GtS(a, b) => {
                            let v = build_i32_relop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_icmp(
                                    llvm::LLVMIntSGT,
                                    a,
                                    b
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I32GeU(a, b) => {
                            let v = build_i32_relop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_icmp(
                                    llvm::LLVMIntUGE,
                                    a,
                                    b
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I32GeS(a, b) => {
                            let v = build_i32_relop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_icmp(
                                    llvm::LLVMIntSGE,
                                    a,
                                    b
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I32WrapI64(a) => {
                            let builder = target_bb.builder();
                            let a = *ssa_values.get(&a).unwrap();
                            let v = builder.build_cast(
                                llvm::LLVMOpcode::LLVMZExt,
                                builder.build_cast(
                                    llvm::LLVMOpcode::LLVMTrunc,
                                    a,
                                    llvm::Type::int32(ctx)
                                ),
                                llvm::Type::int64(ctx)
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I32Load(m, addr) => {
                            let builder = target_bb.builder();

                            let addr = *ssa_values.get(&addr).unwrap();

                            let v = build_std_load(
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
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I32Load8U(m, addr) => {
                            let builder = target_bb.builder();

                            let addr = *ssa_values.get(&addr).unwrap();

                            let v = build_std_load(
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
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I32Load8S(m, addr) => {
                            let builder = target_bb.builder();

                            let addr = *ssa_values.get(&addr).unwrap();

                            let v = build_std_load(
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
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I32Load16U(m, addr) => {
                            let builder = target_bb.builder();

                            let addr = *ssa_values.get(&addr).unwrap();

                            let v = build_std_load(
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
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I32Load16S(m, addr) => {
                            let builder = target_bb.builder();

                            let addr = *ssa_values.get(&addr).unwrap();

                            let v = build_std_load(
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
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I32Store(m, addr, val) => {
                            let builder = target_bb.builder();

                            let addr = *ssa_values.get(&addr).unwrap();
                            let val = *ssa_values.get(&val).unwrap();

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
                        Opcode::I32Store8(m, addr, val) => {
                            let builder = target_bb.builder();

                            let addr = *ssa_values.get(&addr).unwrap();
                            let val = *ssa_values.get(&val).unwrap();

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
                        Opcode::I32Store16(m, addr, val) => {
                            let builder = target_bb.builder();

                            let addr = *ssa_values.get(&addr).unwrap();
                            let val = *ssa_values.get(&val).unwrap();

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
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I64Popcnt(a) => {
                            let builder = target_bb.builder();
                            let a = *ssa_values.get(&a).unwrap();
                            let v = build_i64_unop(
                                &builder,
                                a,
                                &|t, v| t.build_call(&intrinsics.popcnt_i64, &[v])
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I64Clz(a) => {
                            let builder = target_bb.builder();
                            let a = *ssa_values.get(&a).unwrap();
                            let v = build_i64_unop(
                                &builder,
                                a,
                                &|t, v| t.build_call(&intrinsics.clz_i64, &[
                                    v,
                                    t.build_const_int(
                                        llvm::Type::int1(ctx),
                                        0,
                                        false
                                    )
                                ])
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I64Ctz(a) => {
                            let builder = target_bb.builder();
                            let a = *ssa_values.get(&a).unwrap();
                            let v = build_i64_unop(
                                &builder,
                                a,
                                &|t, v| t.build_call(&intrinsics.ctz_i64, &[
                                    v,
                                    t.build_const_int(
                                        llvm::Type::int1(ctx),
                                        0,
                                        false
                                    )
                                ])
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I64Add(a, b) => {
                            let v = build_i64_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_add(a, b)
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I64Sub(a, b) => {
                            let v = build_i64_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_sub(a, b)
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I64Mul(a, b) => {
                            let v = build_i64_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_mul(a, b)
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I64DivU(a, b) => {
                            let v = build_i64_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_udiv(a, b)
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I64DivS(a, b) => {
                            let v = build_i64_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_sdiv(a, b)
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I64RemU(a, b) => {
                            let v = build_i64_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_urem(a, b)
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I64RemS(a, b) => {
                            let v = build_i64_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_srem(a, b)
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        
                        Opcode::I64Shl(a, b) => {
                            let v = build_i64_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_shl(a, b)
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I64ShrU(a, b) => {
                            let v = build_i64_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_lshr(a, b)
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I64ShrS(a, b) => {
                            let v = build_i64_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_ashr(a, b)
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I64And(a, b) => {
                            let v = build_i64_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_and(a, b)
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I64Or(a, b) => {
                            let v = build_i64_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_or(a, b)
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I64Xor(a, b) => {
                            let v = build_i64_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_xor(a, b)
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I64Rotl(a, b) => {
                            let v = build_i64_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_call(
                                    &intrinsics.rotl_i64,
                                    &[a, b]
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I64Rotr(a, b) => {
                            let v = build_i64_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_call(
                                    &intrinsics.rotr_i64,
                                    &[a, b]
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I64Eqz(a) => {
                            let builder = target_bb.builder();
                            let v = builder.build_icmp(
                                llvm::LLVMIntEQ,
                                *ssa_values.get(&a).unwrap(),
                                builder.build_const_int(
                                    llvm::Type::int64(ctx),
                                    0,
                                    false
                                )
                            );
                            ssa_values.insert(
                                ssa_target.unwrap(),
                                builder.build_cast(
                                    llvm::LLVMOpcode::LLVMZExt,
                                    v,
                                    llvm::Type::int64(ctx)
                                )
                            );
                        },
                        Opcode::I64Eq(a, b) => {
                            let v = build_i64_relop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_icmp(
                                    llvm::LLVMIntEQ,
                                    a,
                                    b
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I64Ne(a, b) => {
                            let v = build_i64_relop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_icmp(
                                    llvm::LLVMIntNE,
                                    a,
                                    b
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I64LtU(a, b) => {
                            let v = build_i64_relop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_icmp(
                                    llvm::LLVMIntULT,
                                    a,
                                    b
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I64LtS(a, b) => {
                            let v = build_i64_relop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_icmp(
                                    llvm::LLVMIntSLT,
                                    a,
                                    b
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I64LeU(a, b) => {
                            let v = build_i64_relop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_icmp(
                                    llvm::LLVMIntULE,
                                    a,
                                    b
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I64LeS(a, b) => {
                            let v = build_i64_relop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_icmp(
                                    llvm::LLVMIntSLE,
                                    a,
                                    b
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I64GtU(a, b) => {
                            let v = build_i64_relop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_icmp(
                                    llvm::LLVMIntUGT,
                                    a,
                                    b
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I64GtS(a, b) => {
                            let v = build_i64_relop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_icmp(
                                    llvm::LLVMIntSGT,
                                    a,
                                    b
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I64GeU(a, b) => {
                            let v = build_i64_relop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_icmp(
                                    llvm::LLVMIntUGE,
                                    a,
                                    b
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I64GeS(a, b) => {
                            let v = build_i64_relop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_icmp(
                                    llvm::LLVMIntSGE,
                                    a,
                                    b
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I64ExtendI32U(a) => {
                            let builder = target_bb.builder();
                            let a = *ssa_values.get(&a).unwrap();
                            let v = builder.build_cast(
                                llvm::LLVMOpcode::LLVMZExt,
                                builder.build_cast(
                                    llvm::LLVMOpcode::LLVMTrunc,
                                    a,
                                    llvm::Type::int32(ctx)
                                ),
                                llvm::Type::int64(ctx)
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I64ExtendI32S(a) => {
                            let builder = target_bb.builder();
                            let a = *ssa_values.get(&a).unwrap();
                            let v = builder.build_cast(
                                llvm::LLVMOpcode::LLVMSExt,
                                builder.build_cast(
                                    llvm::LLVMOpcode::LLVMTrunc,
                                    a,
                                    llvm::Type::int32(ctx)
                                ),
                                llvm::Type::int64(ctx)
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I64Load(m, addr) => {
                            let builder = target_bb.builder();

                            let addr = *ssa_values.get(&addr).unwrap();

                            let v = build_std_load(
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
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I64Load8U(m, addr) => {
                            let builder = target_bb.builder();

                            let addr = *ssa_values.get(&addr).unwrap();

                            let v = build_std_load(
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
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I64Load8S(m, addr) => {
                            let builder = target_bb.builder();

                            let addr = *ssa_values.get(&addr).unwrap();

                            let v = build_std_load(
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
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I64Load16U(m, addr) => {
                            let builder = target_bb.builder();

                            let addr = *ssa_values.get(&addr).unwrap();

                            let v = build_std_load(
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
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I64Load16S(m, addr) => {
                            let builder = target_bb.builder();

                            let addr = *ssa_values.get(&addr).unwrap();

                            let v = build_std_load(
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
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I64Load32U(m, addr) => {
                            let builder = target_bb.builder();

                            let addr = *ssa_values.get(&addr).unwrap();

                            let v = build_std_load(
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
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I64Load32S(m, addr) => {
                            let builder = target_bb.builder();

                            let addr = *ssa_values.get(&addr).unwrap();

                            let v = build_std_load(
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
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I64Store(m, addr, val) => {
                            let builder = target_bb.builder();

                            let addr = *ssa_values.get(&addr).unwrap();
                            let val = *ssa_values.get(&val).unwrap();

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
                        Opcode::I64Store8(m, addr, val) => {
                            let builder = target_bb.builder();

                            let addr = *ssa_values.get(&addr).unwrap();
                            let val = *ssa_values.get(&val).unwrap();

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
                        Opcode::I64Store16(m, addr, val) => {
                            let builder = target_bb.builder();

                            let addr = *ssa_values.get(&addr).unwrap();
                            let val = *ssa_values.get(&val).unwrap();

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
                        Opcode::I64Store32(m, addr, val) => {
                            let builder = target_bb.builder();

                            let addr = *ssa_values.get(&addr).unwrap();
                            let val = *ssa_values.get(&val).unwrap();

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
                            let v = builder.build_cast(
                                llvm::LLVMOpcode::LLVMZExt,
                                builder.build_const_int(
                                    llvm::Type::int32(ctx),
                                    v as _,
                                    false
                                ),
                                llvm::Type::int64(ctx)
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::F64Const(v) => {
                            let builder = target_bb.builder();
                            let v = builder.build_const_int(
                                llvm::Type::int64(ctx),
                                v as _,
                                false
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I32ReinterpretF32(t) => {
                            let v = *ssa_values.get(&t).unwrap();
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I64ReinterpretF64(t) => {
                            let v = *ssa_values.get(&t).unwrap();
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::F32ReinterpretI32(t) => {
                            let v = *ssa_values.get(&t).unwrap();
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::F64ReinterpretI64(t) => {
                            let v = *ssa_values.get(&t).unwrap();
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I32TruncSF32(a) => {
                            let builder = target_bb.builder();

                            let v = builder.build_cast(
                                llvm::LLVMOpcode::LLVMFPToSI,
                                extract_f32(&builder, *ssa_values.get(&a).unwrap()),
                                llvm::Type::int32(ctx)
                            );
                            let v = builder.build_cast(
                                llvm::LLVMOpcode::LLVMZExt,
                                v,
                                llvm::Type::int64(ctx)
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I32TruncUF32(a) => {
                            let builder = target_bb.builder();

                            let v = builder.build_cast(
                                llvm::LLVMOpcode::LLVMFPToUI,
                                extract_f32(&builder, *ssa_values.get(&a).unwrap()),
                                llvm::Type::int32(ctx)
                            );
                            let v = builder.build_cast(
                                llvm::LLVMOpcode::LLVMZExt,
                                v,
                                llvm::Type::int64(ctx)
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I32TruncSF64(a) => {
                            let builder = target_bb.builder();
                            
                            let v = builder.build_cast(
                                llvm::LLVMOpcode::LLVMFPToSI,
                                extract_f64(&builder, *ssa_values.get(&a).unwrap()),
                                llvm::Type::int32(ctx)
                            );
                            let v = builder.build_cast(
                                llvm::LLVMOpcode::LLVMZExt,
                                v,
                                llvm::Type::int64(ctx)
                            );

                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I32TruncUF64(a) => {
                            let builder = target_bb.builder();

                            let v = builder.build_cast(
                                llvm::LLVMOpcode::LLVMFPToUI,
                                extract_f64(&builder, *ssa_values.get(&a).unwrap()),
                                llvm::Type::int32(ctx)
                            );
                            let v = builder.build_cast(
                                llvm::LLVMOpcode::LLVMZExt,
                                v,
                                llvm::Type::int64(ctx)
                            );

                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I64TruncSF32(a) => {
                            let builder = target_bb.builder();
                            
                            let v = builder.build_cast(
                                llvm::LLVMOpcode::LLVMFPToSI,
                                extract_f32(&builder, *ssa_values.get(&a).unwrap()),
                                llvm::Type::int64(ctx)
                            );

                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I64TruncUF32(a) => {
                            let builder = target_bb.builder();
                            
                            let v = builder.build_cast(
                                llvm::LLVMOpcode::LLVMFPToUI,
                                extract_f32(&builder, *ssa_values.get(&a).unwrap()),
                                llvm::Type::int64(ctx)
                            );

                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I64TruncSF64(a) => {
                            let builder = target_bb.builder();
                            
                            let v = builder.build_cast(
                                llvm::LLVMOpcode::LLVMFPToSI,
                                extract_f64(&builder, *ssa_values.get(&a).unwrap()),
                                llvm::Type::int64(ctx)
                            );

                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::I64TruncUF64(a) => {
                            let builder = target_bb.builder();
                            
                            let v = builder.build_cast(
                                llvm::LLVMOpcode::LLVMFPToUI,
                                extract_f64(&builder, *ssa_values.get(&a).unwrap()),
                                llvm::Type::int64(ctx)
                            );

                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::F32ConvertSI32(a) => {
                            let builder = target_bb.builder();

                            let v = builder.build_cast(
                                llvm::LLVMOpcode::LLVMSIToFP,
                                builder.build_cast(
                                    llvm::LLVMOpcode::LLVMTrunc,
                                    *ssa_values.get(&a).unwrap(),
                                    llvm::Type::int32(ctx)
                                ),
                                llvm::Type::float32(ctx)
                            );

                            ssa_values.insert(ssa_target.unwrap(), wrap_f32(&builder, v));
                        },
                        Opcode::F32ConvertUI32(a) => {
                            let builder = target_bb.builder();

                            let v = builder.build_cast(
                                llvm::LLVMOpcode::LLVMUIToFP,
                                builder.build_cast(
                                    llvm::LLVMOpcode::LLVMTrunc,
                                    *ssa_values.get(&a).unwrap(),
                                    llvm::Type::int32(ctx)
                                ),
                                llvm::Type::float32(ctx)
                            );

                            ssa_values.insert(ssa_target.unwrap(), wrap_f32(&builder, v));
                        },
                        Opcode::F32ConvertSI64(a) => {
                            let builder = target_bb.builder();

                            let v = builder.build_cast(
                                llvm::LLVMOpcode::LLVMSIToFP,
                                *ssa_values.get(&a).unwrap(),
                                llvm::Type::float32(ctx)
                            );

                            ssa_values.insert(ssa_target.unwrap(), wrap_f32(&builder, v));
                        },
                        Opcode::F32ConvertUI64(a) => {
                            let builder = target_bb.builder();

                            let v = builder.build_cast(
                                llvm::LLVMOpcode::LLVMUIToFP,
                                *ssa_values.get(&a).unwrap(),
                                llvm::Type::float32(ctx)
                            );

                            ssa_values.insert(ssa_target.unwrap(), wrap_f32(&builder, v));
                        },
                        Opcode::F64ConvertSI32(a) => {
                            let builder = target_bb.builder();

                            let v = builder.build_cast(
                                llvm::LLVMOpcode::LLVMSIToFP,
                                builder.build_cast(
                                    llvm::LLVMOpcode::LLVMTrunc,
                                    *ssa_values.get(&a).unwrap(),
                                    llvm::Type::int32(ctx)
                                ),
                                llvm::Type::float64(ctx)
                            );

                            ssa_values.insert(ssa_target.unwrap(), wrap_f64(&builder, v));
                        },
                        Opcode::F64ConvertUI32(a) => {
                            let builder = target_bb.builder();

                            let v = builder.build_cast(
                                llvm::LLVMOpcode::LLVMUIToFP,
                                builder.build_cast(
                                    llvm::LLVMOpcode::LLVMTrunc,
                                    *ssa_values.get(&a).unwrap(),
                                    llvm::Type::int32(ctx)
                                ),
                                llvm::Type::float64(ctx)
                            );

                            ssa_values.insert(ssa_target.unwrap(), wrap_f64(&builder, v));
                        },
                        Opcode::F64ConvertSI64(a) => {
                            let builder = target_bb.builder();

                            let v = builder.build_cast(
                                llvm::LLVMOpcode::LLVMSIToFP,
                                *ssa_values.get(&a).unwrap(),
                                llvm::Type::float64(ctx)
                            );

                            ssa_values.insert(ssa_target.unwrap(), wrap_f64(&builder, v));
                        },
                        Opcode::F64ConvertUI64(a) => {
                            let builder = target_bb.builder();

                            let v = builder.build_cast(
                                llvm::LLVMOpcode::LLVMUIToFP,
                                *ssa_values.get(&a).unwrap(),
                                llvm::Type::float64(ctx)
                            );

                            ssa_values.insert(ssa_target.unwrap(), wrap_f64(&builder, v));
                        },
                        Opcode::F32DemoteF64(a) => {
                            let builder = target_bb.builder();
                            let v = builder.build_fp_trunc(
                                extract_f64(&builder, *ssa_values.get(&a).unwrap()),
                                llvm::Type::float32(ctx)
                            );
                            ssa_values.insert(ssa_target.unwrap(), wrap_f32(&builder, v));
                        },
                        Opcode::F64PromoteF32(a) => {
                            let builder = target_bb.builder();
                            let v = builder.build_fp_ext(
                                extract_f32(&builder, *ssa_values.get(&a).unwrap()),
                                llvm::Type::float64(ctx)
                            );
                            ssa_values.insert(ssa_target.unwrap(), wrap_f64(&builder, v));
                        },
                        Opcode::F32Abs(a) => {
                            let v = build_f32_unop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                &|t, v| t.build_call(
                                    &intrinsics.fabs_f32,
                                    &[ v ]
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::F32Neg(a) => {
                            let v = build_f32_unop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
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
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::F32Ceil(a) => {
                            let v = build_f32_unop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                &|t, v| t.build_call(
                                    &intrinsics.ceil_f32,
                                    &[ v ]
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::F32Floor(a) => {
                            let v = build_f32_unop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                &|t, v| t.build_call(
                                    &intrinsics.floor_f32,
                                    &[ v ]
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::F32Trunc(a) => {
                            let v = build_f32_unop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                &|t, v| t.build_call(
                                    &intrinsics.trunc_f32,
                                    &[ v ]
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::F32Nearest(a) => {
                            let v = build_f32_unop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                &|t, v| t.build_call(
                                    &intrinsics.nearest_f32,
                                    &[ v ]
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::F32Sqrt(a) => {
                            let v = build_f32_unop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                &|t, v| t.build_call(
                                    &intrinsics.sqrt_f32,
                                    &[ v ]
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::F32Add(a, b) => {
                            let v = build_f32_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_fadd(a, b)
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::F32Sub(a, b) => {
                            let v = build_f32_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_fsub(a, b)
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::F32Mul(a, b) => {
                            let v = build_f32_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_fmul(a, b)
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::F32Div(a, b) => {
                            let v = build_f32_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_fdiv(a, b)
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::F32Min(a, b) => {
                            let v = build_f32_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_call(
                                    &intrinsics.minnum_f32,
                                    &[ a, b ]
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::F32Max(a, b) => {
                            let v = build_f32_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_call(
                                    &intrinsics.maxnum_f32,
                                    &[ a, b ]
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::F32Copysign(a, b) => {
                            let v = build_f32_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_call(
                                    &intrinsics.copysign_f32,
                                    &[ a, b ]
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::F32Eq(a, b) => {
                            let v = build_f32_relop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_fcmp(
                                    llvm::LLVMRealOEQ,
                                    a, b
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::F32Ne(a, b) => {
                            let v = build_f32_relop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_fcmp(
                                    llvm::LLVMRealONE,
                                    a, b
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::F32Lt(a, b) => {
                            let v = build_f32_relop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_fcmp(
                                    llvm::LLVMRealOLT,
                                    a, b
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::F32Gt(a, b) => {
                            let v = build_f32_relop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_fcmp(
                                    llvm::LLVMRealOGT,
                                    a, b
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::F32Le(a, b) => {
                            let v = build_f32_relop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_fcmp(
                                    llvm::LLVMRealOLE,
                                    a, b
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::F32Ge(a, b) => {
                            let v = build_f32_relop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_fcmp(
                                    llvm::LLVMRealOGE,
                                    a, b
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::F64Abs(a) => {
                            let v = build_f64_unop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                &|t, v| t.build_call(
                                    &intrinsics.fabs_f64,
                                    &[ v ]
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::F64Neg(a) => {
                            let v = build_f64_unop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
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
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::F64Ceil(a) => {
                            let v = build_f64_unop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                &|t, v| t.build_call(
                                    &intrinsics.ceil_f64,
                                    &[ v ]
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::F64Floor(a) => {
                            let v = build_f64_unop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                &|t, v| t.build_call(
                                    &intrinsics.floor_f64,
                                    &[ v ]
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::F64Trunc(a) => {
                            let v = build_f64_unop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                &|t, v| t.build_call(
                                    &intrinsics.trunc_f64,
                                    &[ v ]
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::F64Nearest(a) => {
                            let v = build_f64_unop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                &|t, v| t.build_call(
                                    &intrinsics.nearest_f64,
                                    &[ v ]
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::F64Sqrt(a) => {
                            let v = build_f64_unop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                &|t, v| t.build_call(
                                    &intrinsics.sqrt_f64,
                                    &[ v ]
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::F64Add(a, b) => {
                            let v = build_f64_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_fadd(a, b)
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::F64Sub(a, b) => {
                            let v = build_f64_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_fsub(a, b)
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::F64Mul(a, b) => {
                            let v = build_f64_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_fmul(a, b)
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::F64Div(a, b) => {
                            let v = build_f64_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_fdiv(a, b)
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::F64Min(a, b) => {
                            let v = build_f64_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_call(
                                    &intrinsics.minnum_f64,
                                    &[ a, b ]
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::F64Max(a, b) => {
                            let v = build_f64_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_call(
                                    &intrinsics.maxnum_f64,
                                    &[ a, b ]
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::F64Copysign(a, b) => {
                            let v = build_f64_binop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_call(
                                    &intrinsics.copysign_f64,
                                    &[ a, b ]
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::F64Eq(a, b) => {
                            let v = build_f64_relop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_fcmp(
                                    llvm::LLVMRealOEQ,
                                    a, b
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::F64Ne(a, b) => {
                            let v = build_f64_relop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_fcmp(
                                    llvm::LLVMRealONE,
                                    a, b
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::F64Lt(a, b) => {
                            let v = build_f64_relop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_fcmp(
                                    llvm::LLVMRealOLT,
                                    a, b
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::F64Gt(a, b) => {
                            let v = build_f64_relop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_fcmp(
                                    llvm::LLVMRealOGT,
                                    a, b
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::F64Le(a, b) => {
                            let v = build_f64_relop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_fcmp(
                                    llvm::LLVMRealOLE,
                                    a, b
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::F64Ge(a, b) => {
                            let v = build_f64_relop(
                                &target_bb.builder(),
                                *ssa_values.get(&a).unwrap(),
                                *ssa_values.get(&b).unwrap(),
                                &|t, a, b| t.build_fcmp(
                                    llvm::LLVMRealOGE,
                                    a, b
                                )
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        },
                        Opcode::Memcpy(_, _, _) => {
                            // not implemented
                            eprintln!("Warning: Not implemented: {:?}", op);
                            let builder = target_bb.builder();
                            builder.build_call(
                                &intrinsics.checked_unreachable,
                                &[]
                            );
                        },
                        Opcode::Phi(ref incoming_values) => {
                            let builder = target_bb.builder();
                            let incoming: Vec<(llvm::LLVMValueRef, &llvm::BasicBlock)> = incoming_values.iter()
                                .map(|id| (
                                    *ssa_values.get(id).unwrap(),
                                    &target_basic_blocks[ssa_block_ids.get(id).unwrap().0]
                                ))
                                .collect();
                            let v = builder.build_phi(
                                &incoming,
                                llvm::Type::int64(ctx)
                            );
                            ssa_values.insert(ssa_target.unwrap(), v);
                        }
                    }
                }
            }
        }

        for (i, bb) in fg.blocks.iter().enumerate() {
            use ::ssa::Branch;

            let target_bb = &target_basic_blocks[i];
            let builder = target_bb.builder();

            if br_unreachable.contains(&i) {
                unsafe {
                    builder.build_unreachable();
                }
                continue;
            }

            unsafe {
                match *bb.br.as_ref().unwrap() {
                    Branch::Br(BlockId(id)) => {
                        builder.build_br(&target_basic_blocks[id]);
                    },
                    Branch::BrEither(cond, BlockId(a), BlockId(b)) => {
                        let v = *ssa_values.get(&cond).unwrap();
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
                            &target_basic_blocks[a],
                            &target_basic_blocks[b]
                        );
                    },
                    Branch::BrTable(cond, ref targets, otherwise) => {
                        let v = *ssa_values.get(&cond).unwrap();

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
                    Branch::Return(v) => {
                        if source_func_ret_ty.len() == 0 {
                            assert!(v.is_none());
                            builder.build_ret_void();
                        } else {
                            let v = *ssa_values.get(&v.unwrap()).unwrap();
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
    use ::opcode::Opcode;
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
            ee.get_function_checked(0)
        };
        let ret = f();
        assert_eq!(ret, 42);
    }

    // FIXME: The new backend silently discards the invalid operations
    // so these tests never terminate. However, it should be a compile-time
    // error instead of silent discarding.
    /*
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
            ee.get_function_checked(0)
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
            ee.get_function_checked(0)
        };
        match catch_unwind(AssertUnwindSafe(|| f())) {
            Ok(_) => panic!("Expecting panic"),
            Err(_) => {}
        }
    }*/

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
            ee.get_function_checked(0)
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
            ee.get_function_checked(0)
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
            ee.get_function_checked(0)
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
            ee.get_function_checked(0)
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
            ee.get_function_checked(0)
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
            ee.get_function_checked(0)
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
            ee.get_function_checked(0)
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
            ee.get_function_checked(0)
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
            ee.get_function_checked(0)
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
            ee.get_function_checked(0)
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
            ee.get_function_checked(0)
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
            ee.get_function_checked(0)
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
            ee.get_function_checked(0)
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
            ee.get_function_checked(0)
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
            ee.get_function_checked(0)
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
            ee.get_function_checked(0)
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
            ee.get_function_checked(0)
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
            ee.get_function_checked(0)
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
            ee.get_function_checked(0)
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
            ee.get_function_checked(0)
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
            ee.get_function_checked(0)
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
            ee.get_function_checked(0)
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
            ee.get_function_checked(0)
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
            ee.get_function_checked(0)
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
            ee.get_function_checked(0)
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
                    vec! [ ValType::I32, ValType::I32 ],
                    vec! [
                        Opcode::GetLocal(1),
                        Opcode::I32Const(100000),
                        Opcode::I32Eq,
                        Opcode::JmpIf(13),
                        Opcode::GetLocal(0),
                        Opcode::I32Const(5),
                        Opcode::NativeInvoke(0),
                        Opcode::SetLocal(2),
                        Opcode::GetLocal(1),
                        Opcode::I32Const(1),
                        Opcode::I32Add,
                        Opcode::SetLocal(1),
                        Opcode::Jmp(0),
                        Opcode::GetLocal(2),
                        Opcode::Return
                    ]
                ),
                ( // dummy
                    Type::Func(vec! [ ValType::I32, ValType::I32 ], vec! [ ValType::I32 ] ),
                    vec! [],
                    vec! [
                        Opcode::I32Const(-1),
                        Opcode::Return
                    ]
                )
            ]
        );
        m.rt.set_native_resolver(TestResolver);

        let ee = m.into_execution_context();

        //println!("{}", ee.ee.to_string());

        let f: extern "C" fn (v: i64) -> i64 = unsafe {
            ee.get_function_checked(0)
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
            ee.get_function_checked(0)
        };
        let getter0: extern "C" fn () -> i64 = unsafe {
            ee.get_function_checked(1)
        };
        let getter1: extern "C" fn () -> i64 = unsafe {
            ee.get_function_checked(2)
        };
        setter(42);
        assert_eq!(getter0(), 42);
        assert_eq!(getter1(), 43);
    }

    #[test]
    fn test_memory_access() {
        let ee = build_ee_from_fn_body(
            Type::Func(vec! [ ValType::I32 ], vec! [ ValType::I32 ]),
            vec! [],
            vec! [
                Opcode::GetLocal(0),
                Opcode::I32Load(Memarg { offset: 0, align: 0 }),
                Opcode::Return
            ]
        );

        let f: extern "C" fn (i64) -> i64 = unsafe {
            ee.get_function_checked(0)
        };

        assert_eq!(catch_unwind(AssertUnwindSafe(|| ee.rt.protected_call(|| f(100)))).unwrap(), 0);
        assert!(catch_unwind(AssertUnwindSafe(|| ee.rt.protected_call(|| f(80000000)))).is_err());
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
            ee.get_function_checked(0)
        };
        let ret = catch_unwind(AssertUnwindSafe(|| f()));
        match ret {
            Ok(_) => panic!("Expecting panic"),
            Err(_) => {}
        }
    }

    #[test]
    fn test_phi() {
        let ee = build_ee_from_fn_body(
            Type::Func(vec! [ ValType::I32 ], vec! [ ValType::I32 ]),
            vec! [],
            vec! [
                Opcode::GetLocal(0), // 0
                Opcode::I32Const(1), // 1
                Opcode::I32Eq, // 2
                Opcode::JmpIf(6), // 3
                Opcode::I32Const(100), // 4
                Opcode::Jmp(7), // 5
                Opcode::I32Const(101), // 6
                Opcode::Return // 7
            ]
        );

        //println!("{}", ee.to_string());

        let f: extern "C" fn (i64) -> i64 = unsafe {
            ee.get_function_checked(0)
        };
        assert_eq!(f(0), 100);
        assert_eq!(f(1), 101);
        assert_eq!(f(2), 100);
    }

    #[test]
    fn test_get_function_checked() {
        let ee = build_ee_from_fn_bodies(
            vec! [
                (
                    Type::Func(vec! [ ValType::I32 ], vec! [ ]),
                    vec! [],
                    vec! [
                        Opcode::Return
                    ]
                ),
                (
                    Type::Func(vec! [ ], vec! [ ValType::I32 ]),
                    vec! [],
                    vec! [
                        Opcode::I32Const(0),
                        Opcode::Return
                    ]
                )
            ]
        );

        //println!("{}", ee.to_string());

        catch_unwind(AssertUnwindSafe(|| {
            let _: extern "C" fn () = unsafe { ee.get_function_checked(0) };
        })).unwrap_err();

        catch_unwind(AssertUnwindSafe(|| {
            let _: extern "C" fn (i64) = unsafe { ee.get_function_checked(0) };
        })).unwrap();

        catch_unwind(AssertUnwindSafe(|| {
            let _: extern "C" fn () = unsafe { ee.get_function_checked(1) };
        })).unwrap_err();

        catch_unwind(AssertUnwindSafe(|| {
            let _: extern "C" fn () -> i64 = unsafe { ee.get_function_checked(1) };
        })).unwrap();
    }
}
