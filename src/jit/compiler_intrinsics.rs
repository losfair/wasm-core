use super::llvm;
use super::runtime::Runtime;
use platform::generic::MemoryManager;

pub struct CompilerIntrinsics {
    pub(super) stacksave: llvm::Function,
    pub(super) stackrestore: llvm::Function,
    pub(super) popcnt_i32: llvm::Function,
    pub(super) popcnt_i64: llvm::Function,
    pub(super) clz_i32: llvm::Function,
    pub(super) clz_i64: llvm::Function,
    pub(super) ctz_i32: llvm::Function,
    pub(super) ctz_i64: llvm::Function,
    pub(super) rotl_i32: llvm::Function,
    pub(super) rotl_i64: llvm::Function,
    pub(super) rotr_i32: llvm::Function,
    pub(super) rotr_i64: llvm::Function,
    pub(super) fabs_f32: llvm::Function,
    pub(super) fabs_f64: llvm::Function,
    pub(super) ceil_f32: llvm::Function,
    pub(super) ceil_f64: llvm::Function,
    pub(super) floor_f32: llvm::Function,
    pub(super) floor_f64: llvm::Function,
    pub(super) nearest_f32: llvm::Function,
    pub(super) nearest_f64: llvm::Function,
    pub(super) sqrt_f32: llvm::Function,
    pub(super) sqrt_f64: llvm::Function,
    pub(super) trunc_f32: llvm::Function,
    pub(super) trunc_f64: llvm::Function,
    pub(super) copysign_f32: llvm::Function,
    pub(super) copysign_f64: llvm::Function,
    pub(super) minnum_f32: llvm::Function,
    pub(super) minnum_f64: llvm::Function,
    pub(super) maxnum_f32: llvm::Function,
    pub(super) maxnum_f64: llvm::Function,
    pub(super) checked_unreachable: llvm::Function,
    pub(super) check_stack: llvm::Function,
    pub(super) select: llvm::Function,
    pub(super) translate_pointer: llvm::Function,
    pub(super) get_function_addr: llvm::Function,
    pub(super) indirect_get_function_addr: llvm::Function,
    pub(super) enforce_indirect_fn_typeck: llvm::Function,
    pub(super) grow_memory: llvm::Function,
    pub(super) current_memory: llvm::Function
}

impl CompilerIntrinsics {
    pub fn new(ctx: &llvm::Context, m: &llvm::Module, rt: &Runtime) -> CompilerIntrinsics {
        CompilerIntrinsics {
            stacksave: llvm::Function::new(
                ctx,
                m,
                "llvm.stacksave",
                llvm::Type::function(
                    ctx,
                    llvm::Type::pointer(llvm::Type::int8(ctx)),
                    &[]
                )
            ),
            stackrestore: llvm::Function::new(
                ctx,
                m,
                "llvm.stackrestore",
                llvm::Type::function(
                    ctx,
                    llvm::Type::void(ctx),
                    &[
                        llvm::Type::pointer(llvm::Type::int8(ctx))
                    ]
                )
            ),
            popcnt_i32: llvm::Function::new(
                ctx,
                m,
                "llvm.ctpop.i32",
                llvm::Type::function(
                    ctx,
                    llvm::Type::int32(ctx),
                    &[
                        llvm::Type::int32(ctx)
                    ]
                )
            ),
            popcnt_i64: llvm::Function::new(
                ctx,
                m,
                "llvm.ctpop.i64",
                llvm::Type::function(
                    ctx,
                    llvm::Type::int64(ctx),
                    &[
                        llvm::Type::int64(ctx)
                    ]
                )
            ),
            clz_i32: llvm::Function::new(
                ctx,
                m,
                "llvm.ctlz.i32",
                llvm::Type::function(
                    ctx,
                    llvm::Type::int32(ctx),
                    &[
                        llvm::Type::int32(ctx),
                        llvm::Type::int1(ctx)
                    ]
                )
            ),
            clz_i64: llvm::Function::new(
                ctx,
                m,
                "llvm.ctlz.i64",
                llvm::Type::function(
                    ctx,
                    llvm::Type::int64(ctx),
                    &[
                        llvm::Type::int64(ctx),
                        llvm::Type::int1(ctx)
                    ]
                )
            ),
            ctz_i32: llvm::Function::new(
                ctx,
                m,
                "llvm.cttz.i32",
                llvm::Type::function(
                    ctx,
                    llvm::Type::int32(ctx),
                    &[
                        llvm::Type::int32(ctx),
                        llvm::Type::int1(ctx)
                    ]
                )
            ),
            ctz_i64: llvm::Function::new(
                ctx,
                m,
                "llvm.cttz.i64",
                llvm::Type::function(
                    ctx,
                    llvm::Type::int64(ctx),
                    &[
                        llvm::Type::int64(ctx),
                        llvm::Type::int1(ctx)
                    ]
                )
            ),
            fabs_f32: llvm::Function::new(
                ctx,
                m,
                "llvm.fabs.f32",
                llvm::Type::function(
                    ctx,
                    llvm::Type::float32(ctx),
                    &[
                        llvm::Type::float32(ctx)
                    ]
                )
            ),
            fabs_f64: llvm::Function::new(
                ctx,
                m,
                "llvm.fabs.f64",
                llvm::Type::function(
                    ctx,
                    llvm::Type::float64(ctx),
                    &[
                        llvm::Type::float64(ctx)
                    ]
                )
            ),
            ceil_f32: llvm::Function::new(
                ctx,
                m,
                "llvm.ceil.f32",
                llvm::Type::function(
                    ctx,
                    llvm::Type::float32(ctx),
                    &[
                        llvm::Type::float32(ctx)
                    ]
                )
            ),
            ceil_f64: llvm::Function::new(
                ctx,
                m,
                "llvm.ceil.f64",
                llvm::Type::function(
                    ctx,
                    llvm::Type::float64(ctx),
                    &[
                        llvm::Type::float64(ctx)
                    ]
                )
            ),
            floor_f32: llvm::Function::new(
                ctx,
                m,
                "llvm.floor.f32",
                llvm::Type::function(
                    ctx,
                    llvm::Type::float32(ctx),
                    &[
                        llvm::Type::float32(ctx)
                    ]
                )
            ),
            floor_f64: llvm::Function::new(
                ctx,
                m,
                "llvm.floor.f64",
                llvm::Type::function(
                    ctx,
                    llvm::Type::float64(ctx),
                    &[
                        llvm::Type::float64(ctx)
                    ]
                )
            ),
            nearest_f32: llvm::Function::new(
                ctx,
                m,
                "llvm.round.f32",
                llvm::Type::function(
                    ctx,
                    llvm::Type::float32(ctx),
                    &[
                        llvm::Type::float32(ctx)
                    ]
                )
            ),
            nearest_f64: llvm::Function::new(
                ctx,
                m,
                "llvm.round.f64",
                llvm::Type::function(
                    ctx,
                    llvm::Type::float64(ctx),
                    &[
                        llvm::Type::float64(ctx)
                    ]
                )
            ),
            sqrt_f32: llvm::Function::new(
                ctx,
                m,
                "llvm.sqrt.f32",
                llvm::Type::function(
                    ctx,
                    llvm::Type::float32(ctx),
                    &[
                        llvm::Type::float32(ctx)
                    ]
                )
            ),
            sqrt_f64: llvm::Function::new(
                ctx,
                m,
                "llvm.sqrt.f64",
                llvm::Type::function(
                    ctx,
                    llvm::Type::float64(ctx),
                    &[
                        llvm::Type::float64(ctx)
                    ]
                )
            ),
            trunc_f32: llvm::Function::new(
                ctx,
                m,
                "llvm.trunc.f32",
                llvm::Type::function(
                    ctx,
                    llvm::Type::float32(ctx),
                    &[
                        llvm::Type::float32(ctx)
                    ]
                )
            ),
            trunc_f64: llvm::Function::new(
                ctx,
                m,
                "llvm.trunc.f64",
                llvm::Type::function(
                    ctx,
                    llvm::Type::float64(ctx),
                    &[
                        llvm::Type::float64(ctx)
                    ]
                )
            ),
            copysign_f32: llvm::Function::new(
                ctx,
                m,
                "llvm.copysign.f32",
                llvm::Type::function(
                    ctx,
                    llvm::Type::float32(ctx),
                    &[
                        llvm::Type::float32(ctx),
                        llvm::Type::float32(ctx)
                    ]
                )
            ),
            copysign_f64: llvm::Function::new(
                ctx,
                m,
                "llvm.copysign.f64",
                llvm::Type::function(
                    ctx,
                    llvm::Type::float64(ctx),
                    &[
                        llvm::Type::float64(ctx),
                        llvm::Type::float64(ctx)
                    ]
                )
            ),
            minnum_f32: llvm::Function::new(
                ctx,
                m,
                "llvm.minnum.f32",
                llvm::Type::function(
                    ctx,
                    llvm::Type::float32(ctx),
                    &[
                        llvm::Type::float32(ctx),
                        llvm::Type::float32(ctx)
                    ]
                )
            ),
            minnum_f64: llvm::Function::new(
                ctx,
                m,
                "llvm.minnum.f64",
                llvm::Type::function(
                    ctx,
                    llvm::Type::float64(ctx),
                    &[
                        llvm::Type::float64(ctx),
                        llvm::Type::float64(ctx)
                    ]
                )
            ),
            maxnum_f32: llvm::Function::new(
                ctx,
                m,
                "llvm.maxnum.f32",
                llvm::Type::function(
                    ctx,
                    llvm::Type::float32(ctx),
                    &[
                        llvm::Type::float32(ctx),
                        llvm::Type::float32(ctx)
                    ]
                )
            ),
            maxnum_f64: llvm::Function::new(
                ctx,
                m,
                "llvm.maxnum.f64",
                llvm::Type::function(
                    ctx,
                    llvm::Type::float64(ctx),
                    &[
                        llvm::Type::float64(ctx),
                        llvm::Type::float64(ctx)
                    ]
                )
            ),
            rotl_i32: Self::build_rotl_i32(ctx, m),
            rotl_i64: Self::build_rotl_i64(ctx, m),
            rotr_i32: Self::build_rotr_i32(ctx, m),
            rotr_i64: Self::build_rotr_i64(ctx, m),
            checked_unreachable: Self::build_checked_unreachable(ctx, m),
            check_stack: Self::build_check_stack(ctx, m),
            select: Self::build_select(ctx, m),
            translate_pointer: Self::build_translate_pointer(ctx, m, rt),
            get_function_addr: Self::build_get_function_addr(ctx, m, rt),
            indirect_get_function_addr: Self::build_indirect_get_function_addr(ctx, m, rt),
            enforce_indirect_fn_typeck: Self::build_enforce_indirect_fn_typeck(ctx, m, rt),
            grow_memory: Self::build_grow_memory(ctx, m, rt),
            current_memory: Self::build_current_memory(ctx, m, rt)
        }
    }

    extern "C" fn stack_check_failed() {
        panic!("Stack check failed");
    }

    extern "C" fn mem_bounds_check_failed(vaddr: u64, len: u64) {
        panic!("Memory bounds check failed (vaddr = {}, len = {})", vaddr, len);
    }

    extern "C" fn on_checked_unreachable() {
        panic!("Unreachable executed");
    }

    extern "C" fn do_enforce_indirect_fn_typeck(rt: &Runtime, expected_typeidx: usize, indirect_index: usize) {
        let ty1 = &rt.source_module.types[expected_typeidx];

        let got_fn = rt.source_module.tables[0].elements[indirect_index].unwrap() as usize;
        let got_typeidx = rt.source_module.functions[got_fn].typeidx as usize;
        let ty2 = &rt.source_module.types[got_typeidx];

        if expected_typeidx == got_typeidx {
            return;
        }

        if ty1 != ty2 {
            panic!("enforce_typeck: Type mismatch: Expected {:?}, got {:?}", ty1, ty2);
        }
    }

    fn build_rotl_i32(ctx: &llvm::Context, m: &llvm::Module) -> llvm::Function {
        let f: llvm::Function = llvm::Function::new(
            ctx,
            m,
            "rotl_i32",
            llvm::Type::function(
                ctx,
                llvm::Type::int32(ctx),
                &[
                    llvm::Type::int32(ctx),
                    llvm::Type::int32(ctx)
                ]
            )
        );
        f.set_linkage(llvm::LLVMInternalLinkage);
        let check_bb = llvm::BasicBlock::new(&f);
        let ret_bb = llvm::BasicBlock::new(&f);
        let calc_bb = llvm::BasicBlock::new(&f);

        unsafe {
            let builder = check_bb.builder();
            let cmp_result = builder.build_icmp(
                llvm::LLVMIntEQ,
                f.get_param(1),
                builder.build_const_int(
                    llvm::Type::int32(ctx),
                    0,
                    false
                )
            );
            builder.build_cond_br(
                cmp_result,
                &ret_bb,
                &calc_bb
            );
        }

        unsafe {
            let builder = ret_bb.builder();
            builder.build_ret(f.get_param(0));
        }

        unsafe {
            let builder = calc_bb.builder();
            let ret = builder.build_or(
                builder.build_shl(
                    f.get_param(0),
                    f.get_param(1)
                ),
                builder.build_lshr(
                    f.get_param(0),
                    builder.build_sub(
                        builder.build_const_int(
                            llvm::Type::int32(ctx),
                            32,
                            false
                        ),
                        f.get_param(1)
                    )
                )
            );
            builder.build_ret(ret);
        }

        f.verify();
        f
    }

    fn build_rotl_i64(ctx: &llvm::Context, m: &llvm::Module) -> llvm::Function {
        let f: llvm::Function = llvm::Function::new(
            ctx,
            m,
            "rotl_i64",
            llvm::Type::function(
                ctx,
                llvm::Type::int64(ctx),
                &[
                    llvm::Type::int64(ctx),
                    llvm::Type::int64(ctx)
                ]
            )
        );
        f.set_linkage(llvm::LLVMInternalLinkage);
        let check_bb = llvm::BasicBlock::new(&f);
        let ret_bb = llvm::BasicBlock::new(&f);
        let calc_bb = llvm::BasicBlock::new(&f);

        unsafe {
            let builder = check_bb.builder();
            let cmp_result = builder.build_icmp(
                llvm::LLVMIntEQ,
                f.get_param(1),
                builder.build_const_int(
                    llvm::Type::int64(ctx),
                    0,
                    false
                )
            );
            builder.build_cond_br(
                cmp_result,
                &ret_bb,
                &calc_bb
            );
        }

        unsafe {
            let builder = ret_bb.builder();
            builder.build_ret(f.get_param(0));
        }

        unsafe {
            let builder = calc_bb.builder();
            let ret = builder.build_or(
                builder.build_shl(
                    f.get_param(0),
                    f.get_param(1)
                ),
                builder.build_lshr(
                    f.get_param(0),
                    builder.build_sub(
                        builder.build_const_int(
                            llvm::Type::int64(ctx),
                            64,
                            false
                        ),
                        f.get_param(1)
                    )
                )
            );
            builder.build_ret(ret);
        }

        f.verify();
        f
    }

    fn build_rotr_i32(ctx: &llvm::Context, m: &llvm::Module) -> llvm::Function {
        let f: llvm::Function = llvm::Function::new(
            ctx,
            m,
            "rotr_i32",
            llvm::Type::function(
                ctx,
                llvm::Type::int32(ctx),
                &[
                    llvm::Type::int32(ctx),
                    llvm::Type::int32(ctx)
                ]
            )
        );
        f.set_linkage(llvm::LLVMInternalLinkage);
        let check_bb = llvm::BasicBlock::new(&f);
        let ret_bb = llvm::BasicBlock::new(&f);
        let calc_bb = llvm::BasicBlock::new(&f);

        unsafe {
            let builder = check_bb.builder();
            let cmp_result = builder.build_icmp(
                llvm::LLVMIntEQ,
                f.get_param(1),
                builder.build_const_int(
                    llvm::Type::int32(ctx),
                    0,
                    false
                )
            );
            builder.build_cond_br(
                cmp_result,
                &ret_bb,
                &calc_bb
            );
        }

        unsafe {
            let builder = ret_bb.builder();
            builder.build_ret(f.get_param(0));
        }

        unsafe {
            let builder = calc_bb.builder();
            let ret = builder.build_or(
                builder.build_lshr(
                    f.get_param(0),
                    f.get_param(1)
                ),
                builder.build_shl(
                    f.get_param(0),
                    builder.build_sub(
                        builder.build_const_int(
                            llvm::Type::int32(ctx),
                            32,
                            false
                        ),
                        f.get_param(1)
                    )
                )
            );
            builder.build_ret(ret);
        }

        f.verify();
        f
    }

    fn build_rotr_i64(ctx: &llvm::Context, m: &llvm::Module) -> llvm::Function {
        let f: llvm::Function = llvm::Function::new(
            ctx,
            m,
            "rotr_i64",
            llvm::Type::function(
                ctx,
                llvm::Type::int64(ctx),
                &[
                    llvm::Type::int64(ctx),
                    llvm::Type::int64(ctx)
                ]
            )
        );
        f.set_linkage(llvm::LLVMInternalLinkage);
        let check_bb = llvm::BasicBlock::new(&f);
        let ret_bb = llvm::BasicBlock::new(&f);
        let calc_bb = llvm::BasicBlock::new(&f);

        unsafe {
            let builder = check_bb.builder();
            let cmp_result = builder.build_icmp(
                llvm::LLVMIntEQ,
                f.get_param(1),
                builder.build_const_int(
                    llvm::Type::int64(ctx),
                    0,
                    false
                )
            );
            builder.build_cond_br(
                cmp_result,
                &ret_bb,
                &calc_bb
            );
        }

        unsafe {
            let builder = ret_bb.builder();
            builder.build_ret(f.get_param(0));
        }

        unsafe {
            let builder = calc_bb.builder();
            let ret = builder.build_or(
                builder.build_lshr(
                    f.get_param(0),
                    f.get_param(1)
                ),
                builder.build_shl(
                    f.get_param(0),
                    builder.build_sub(
                        builder.build_const_int(
                            llvm::Type::int64(ctx),
                            64,
                            false
                        ),
                        f.get_param(1)
                    )
                )
            );
            builder.build_ret(ret);
        }

        f.verify();
        f
    }

    fn build_checked_unreachable(ctx: &llvm::Context, m: &llvm::Module) -> llvm::Function {
        let f: llvm::Function = llvm::Function::new(
            ctx,
            m,
            "checked_unreachable",
            llvm::Type::function(
                ctx,
                llvm::Type::void(ctx),
                &[]
            )
        );
        f.set_linkage(llvm::LLVMInternalLinkage);
        let initial_bb = llvm::BasicBlock::new(&f);

        unsafe {
            let builder = initial_bb.builder();
            builder.build_call_raw(
                builder.build_cast(
                    llvm::LLVMOpcode::LLVMIntToPtr,
                    builder.build_const_int(
                        llvm::Type::int64(ctx),
                        Self::on_checked_unreachable as usize as u64,
                        false
                    ),
                    llvm::Type::pointer(
                        llvm::Type::function(
                            ctx,
                            llvm::Type::void(ctx),
                            &[]
                        )
                    )
                ),
                &[]
            );
            builder.build_unreachable();
        }

        f.verify();
        f
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
        f.set_linkage(llvm::LLVMInternalLinkage);
        let initial_bb = llvm::BasicBlock::new(&f);

        unsafe {
            let builder = initial_bb.builder();
            let jit_info = &mut *rt.get_jit_info();
            let hints = unsafe { &*rt.mm.get() }.hints();

            let indirect_len_ptr: *const usize = hints.indirect_len_ptr;

            let ret = builder.build_cast(
                llvm::LLVMOpcode::LLVMZExt,
                builder.build_load(
                    builder.build_cast(
                        llvm::LLVMOpcode::LLVMIntToPtr,
                        builder.build_const_int(
                            llvm::Type::int64(ctx),
                            (indirect_len_ptr as usize) as u64,
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
        f.set_linkage(llvm::LLVMInternalLinkage);
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
        f.set_linkage(llvm::LLVMInternalLinkage);
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
        f.set_linkage(llvm::LLVMInternalLinkage);

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
        f.set_linkage(llvm::LLVMInternalLinkage);

        let jit_info = unsafe {
            &mut *rt.get_jit_info()
        };
        let hints = unsafe { &*rt.mm.get() }.hints();
        let mem_begin_ptr: *const *mut u8 = hints.indirect_start_address_ptr;
        let mem_len_ptr: *const usize = hints.indirect_len_ptr;

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
                        &[
                            llvm::Type::int64(ctx),
                            llvm::Type::int64(ctx)
                        ]
                    )
                )
            );
            builder.build_call_raw(call_target, &[
                f.get_param(0),
                f.get_param(1)
            ]);
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
        f.set_linkage(llvm::LLVMInternalLinkage);

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

    fn build_get_function_addr(ctx: &llvm::Context, m: &llvm::Module, rt: &Runtime) -> llvm::Function {
        let f: llvm::Function = llvm::Function::new(
            ctx,
            m,
            "get_function_addr",
            llvm::Type::function(
                ctx,
                llvm::Type::pointer(llvm::Type::void(ctx)),
                &[
                    llvm::Type::int64(ctx) // function id
                ]
            )
        );
        f.set_linkage(llvm::LLVMInternalLinkage);

        let initial_bb = llvm::BasicBlock::new(&f);

        unsafe {
            let builder = initial_bb.builder();
            let ret = builder.build_call_raw(
                builder.build_cast(
                    llvm::LLVMOpcode::LLVMIntToPtr,
                    builder.build_const_int(
                        llvm::Type::int64(ctx),
                        (Runtime::_jit_get_function_addr as usize) as u64,
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
        f.set_linkage(llvm::LLVMInternalLinkage);
        
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
