use llvm_sys;
use llvm_sys::prelude::*;
use llvm_sys::core::*;
use llvm_sys::execution_engine::*;
use llvm_sys::target::*;
use llvm_sys::analysis::*;
use llvm_sys::transforms::pass_manager_builder::*;
use llvm_sys::{
    LLVMIntPredicate,
    LLVMRealPredicate,
    LLVMLinkage
};
use std::rc::Rc;
use std::cell::Cell;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};

pub use llvm_sys::LLVMOpcode;
pub use llvm_sys::LLVMIntPredicate::*;
pub use llvm_sys::LLVMRealPredicate::*;
pub use llvm_sys::prelude::LLVMValueRef;
pub use llvm_sys::LLVMLinkage::*;

fn empty_cstr() -> *const c_char {
    b"\0".as_ptr() as _
}

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

#[derive(Clone)]
pub struct Module {
    inner: Rc<ModuleImpl>
}

pub struct ModuleImpl {
    _context: Context,
    _ref: LLVMModuleRef,
    _ref_invalidated: Cell<bool>
}

impl Module {
    pub fn new(ctx: &Context, name: String) -> Module {
        let name = CString::new(name).unwrap();

        Module {
            inner: Rc::new(ModuleImpl {
                _context: ctx.clone(),
                _ref: unsafe { LLVMModuleCreateWithNameInContext(
                    name.as_ptr(),
                    ctx.inner._ref
                ) },
                _ref_invalidated: Cell::new(false)
            })
        }
    }

    pub fn deep_clone(&self) -> Module {
        Module {
            inner: Rc::new(ModuleImpl {
                _context: self.inner._context.clone(),
                _ref: unsafe { LLVMCloneModule(self.inner._ref) },
                _ref_invalidated: Cell::new(false)
            })
        }
    }

    pub fn verify(&self) {
        unsafe {
            LLVMVerifyModule(
                self.inner._ref,
                LLVMVerifierFailureAction::LLVMAbortProcessAction,
                ::std::ptr::null_mut()
            );
        }
    }

    pub fn optimize(&self) {
        self.verify();

        unsafe {
            let pm = LLVMCreatePassManager();

            {
                let pmb = LLVMPassManagerBuilderCreate();

                LLVMPassManagerBuilderSetOptLevel(pmb, 1);
                LLVMPassManagerBuilderUseInlinerWithThreshold(pmb, 120);
                LLVMPassManagerBuilderPopulateModulePassManager(pmb, pm);

                LLVMPassManagerBuilderDispose(pmb);
            }

            LLVMRunPassManager(pm, self.inner._ref);
            LLVMRunPassManager(pm, self.inner._ref);

            LLVMDisposePassManager(pm);
        }
    }
}

impl Drop for ModuleImpl {
    fn drop(&mut self) {
        unsafe {
            if !self._ref_invalidated.get() {
                LLVMDisposeModule(self._ref);
            }
        }
    }
}

pub struct ExecutionEngine {
    _context: Context,
    _ref: LLVMExecutionEngineRef,
    _module_ref: LLVMModuleRef
}

impl ExecutionEngine {
    pub fn new(m: Module) -> ExecutionEngine {
        Self::with_opt_level(m, 1)
    }

    pub fn with_opt_level(mut m: Module, opt_level: usize) -> ExecutionEngine {
        // Ensure that LLVM JIT has been initialized
        assert_eq!(*LLVM_EXEC, true);

        let m = Rc::try_unwrap(m.inner).unwrap_or_else(|_| {
            panic!("Attempting to create an execution engine from a module while there are still some strong references to it");
        });

        unsafe {
            LLVMVerifyModule(
                m._ref,
                LLVMVerifierFailureAction::LLVMAbortProcessAction,
                ::std::ptr::null_mut()
            );
        }

        unsafe {
            let mut ee: LLVMExecutionEngineRef = ::std::mem::uninitialized();
            let mut err: *mut c_char = ::std::ptr::null_mut();

            let mut mcjit_opts: LLVMMCJITCompilerOptions = ::std::mem::uninitialized();
            LLVMInitializeMCJITCompilerOptions(
                &mut mcjit_opts,
                ::std::mem::size_of::<LLVMMCJITCompilerOptions>() as _
            );
            mcjit_opts.OptLevel = opt_level as _;

            let ret = LLVMCreateMCJITCompilerForModule(
                &mut ee,
                m._ref,
                &mut mcjit_opts,
                ::std::mem::size_of::<LLVMMCJITCompilerOptions>() as _,
                &mut err
            );
            if ret != 0 {
                let e = CStr::from_ptr(err).to_str().unwrap_or_else(|_| {
                    eprintln!("Fatal error: Unable to read error string");
                    ::std::process::abort();
                });
                eprintln!("Fatal error: Unable to create execution engine from module: {}", e);
                ::std::process::abort();
            }

            m._ref_invalidated.set(true);

            ExecutionEngine {
                _context: m._context.clone(),
                _ref: ee,
                _module_ref: m._ref
            }
        }
    }

    pub fn deep_clone_module(&self) -> Module {
        Module {
            inner: Rc::new(ModuleImpl {
                _context: self._context.clone(),
                _ref: unsafe { LLVMCloneModule(self._module_ref) },
                _ref_invalidated: Cell::new(false)
            })
        }
    }

    pub fn get_function_address(&self, name: &str) -> Option<*const c_void> {
        let name = CString::new(name).unwrap();

        let addr = unsafe { LLVMGetFunctionAddress(
            self._ref,
            name.as_ptr()
        ) as *const c_void };
        if addr.is_null() {
            None
        } else {
            Some(addr)
        }
    }

    pub fn to_string(&self) -> String {
        unsafe {
            let raw_s = LLVMPrintModuleToString(self._module_ref);
            let s = CStr::from_ptr(
                raw_s
            ).to_str().unwrap().to_string();
            LLVMDisposeMessage(raw_s);
            s
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

pub struct Type {
    _context: Context,
    _ref: LLVMTypeRef
}

impl Type {
    pub fn function(ctx: &Context, ret: Type, params: &[Type]) -> Type {
        let mut params: Vec<LLVMTypeRef> = params.iter().map(|v| v._ref).collect();

        Type {
            _context: ctx.clone(),
            _ref: unsafe { LLVMFunctionType(
                ret._ref,
                if params.len() == 0 {
                    ::std::ptr::null_mut()
                } else {
                    &mut params[0]
                },
                params.len() as _,
                0
            ) }
        }
    }

    pub fn int_native(ctx: &Context) -> Type {
        match NativePointerWidth::detect() {
            NativePointerWidth::W32 => Self::int32(ctx),
            NativePointerWidth::W64 => Self::int64(ctx)
        }
    }

    pub fn int1(ctx: &Context) -> Type {
        Type {
            _context: ctx.clone(),
            _ref: unsafe { LLVMInt1TypeInContext(ctx.inner._ref) }
        }
    }

    pub fn int8(ctx: &Context) -> Type {
        Type {
            _context: ctx.clone(),
            _ref: unsafe { LLVMInt8TypeInContext(ctx.inner._ref) }
        }
    }

    pub fn int16(ctx: &Context) -> Type {
        Type {
            _context: ctx.clone(),
            _ref: unsafe { LLVMInt16TypeInContext(ctx.inner._ref) }
        }
    }

    pub fn int32(ctx: &Context) -> Type {
        Type {
            _context: ctx.clone(),
            _ref: unsafe { LLVMInt32TypeInContext(ctx.inner._ref) }
        }
    }

    pub fn int64(ctx: &Context) -> Type {
        Type {
            _context: ctx.clone(),
            _ref: unsafe { LLVMInt64TypeInContext(ctx.inner._ref) }
        }
    }

    pub fn float32(ctx: &Context) -> Type {
        Type {
            _context: ctx.clone(),
            _ref: unsafe { LLVMFloatTypeInContext(ctx.inner._ref) }
        }
    }

    pub fn float64(ctx: &Context) -> Type {
        Type {
            _context: ctx.clone(),
            _ref: unsafe { LLVMDoubleTypeInContext(ctx.inner._ref) }
        }
    }

    pub fn void(ctx: &Context) -> Type {
        Type {
            _context: ctx.clone(),
            _ref: unsafe { LLVMVoidTypeInContext(ctx.inner._ref) }
        }
    }

    pub fn pointer(inner: Type) -> Type {
        Type {
            _context: inner._context.clone(),
            _ref: unsafe { LLVMPointerType(inner._ref, 0) }
        }
    }

    pub fn struct_type(ctx: &Context, inner: &[Type], packed: bool) -> Type {
        let mut llvm_types: Vec<LLVMTypeRef> = inner.iter().map(|t| t._ref).collect();
        let n_elements = llvm_types.len();

        Type {
            _context: ctx.clone(),
            _ref: unsafe {
                LLVMStructTypeInContext(
                    ctx.inner._ref,
                    if llvm_types.len() == 0 {
                        ::std::ptr::null_mut()
                    } else {
                        &mut llvm_types[0]
                    },
                    n_elements as _,
                    if packed {
                        1
                    } else {
                        0
                    }
                )
            }
        }
    }

    pub fn array(ctx: &Context, elem_type: Type, n_elements: usize) -> Type {
        Type {
            _context: ctx.clone(),
            _ref: unsafe {
                LLVMArrayType(
                    elem_type._ref,
                    n_elements as _
                )
            }
        }
    }
}

pub struct Function {
    _context: Context,
    _module: Module,
    _ref: LLVMValueRef
}

impl Function {
    pub fn new(ctx: &Context, m: &Module, name: &str, ty: Type) -> Function {
        let name = CString::new(name).unwrap();

        Function {
            _context: ctx.clone(),
            _module: m.clone(),
            _ref: unsafe { LLVMAddFunction(m.inner._ref, name.as_ptr(), ty._ref) }
        }
    }

    pub unsafe fn get_param(&self, idx: usize) -> LLVMValueRef {
        unsafe {
            LLVMGetParam(self._ref, idx as _)
        }
    }

    pub fn to_string(&self) -> String {
        unsafe {
            let raw_s = LLVMPrintValueToString(self._ref);
            let s = CStr::from_ptr(
                raw_s
            ).to_str().unwrap().to_string();
            LLVMDisposeMessage(raw_s);
            s
        }
    }

    pub fn verify(&self) {
        unsafe {
            LLVMVerifyFunction(
                self._ref,
                LLVMVerifierFailureAction::LLVMAbortProcessAction
            );
        }
    }

    pub fn set_linkage(&self, linkage: LLVMLinkage) {
        unsafe {
            LLVMSetLinkage(self._ref, linkage);
        }
    }
}

pub struct BasicBlock<'a> {
    _func: &'a Function,
    _ref: LLVMBasicBlockRef
}

impl<'a> BasicBlock<'a> {
    pub fn new(f: &'a Function) -> BasicBlock<'a> {
        BasicBlock {
            _func: f,
            _ref: unsafe { LLVMAppendBasicBlockInContext(
                f._context.inner._ref,
                f._ref,
                b"\0".as_ptr() as _
            ) }
        }
    }

    pub fn builder(&'a self) -> Builder<'a> {
        Builder::new(self)
    }
}

pub struct Builder<'a> {
    _bb: &'a BasicBlock<'a>,
    _ref: LLVMBuilderRef
}

impl<'a> Builder<'a> {
    pub fn new(bb: &'a BasicBlock<'a>) -> Builder<'a> {
        unsafe {
            let b_ref = LLVMCreateBuilderInContext(
                bb._func._context.inner._ref
            );
            LLVMPositionBuilderAtEnd(b_ref, bb._ref);

            Builder {
                _bb: bb,
                _ref: b_ref
            }
        }
    }

    pub unsafe fn build_br(&self, other: &BasicBlock) -> LLVMValueRef {
        LLVMBuildBr(self._ref, other._ref)
    }

    pub unsafe fn build_cond_br(
        &self,
        val: LLVMValueRef,
        if_true: &BasicBlock,
        if_false: &BasicBlock
    ) -> LLVMValueRef {
        LLVMBuildCondBr(self._ref, val, if_true._ref, if_false._ref)
    }

    pub unsafe fn build_alloca(&self, ty: Type) -> LLVMValueRef {
        LLVMBuildAlloca(
            self._ref,
            ty._ref,
            empty_cstr()
        )
    }

    pub unsafe fn build_const_int(&self, ty: Type, v: u64, sign_extend: bool) -> LLVMValueRef {
        LLVMConstInt(
            ty._ref,
            v as _,
            if sign_extend {
                1
            } else {
                0
            }
        )
    }

    pub unsafe fn build_store(&self, val: LLVMValueRef, ptr: LLVMValueRef) -> LLVMValueRef {
        LLVMBuildStore(
            self._ref,
            val,
            ptr
        )
    }

    pub unsafe fn build_load(&self, ptr: LLVMValueRef) -> LLVMValueRef {
        LLVMBuildLoad(
            self._ref,
            ptr,
            empty_cstr()
        )
    }

    pub unsafe fn build_cast(
        &self,
        op: LLVMOpcode,
        val: LLVMValueRef,
        dest_ty: Type
    ) -> LLVMValueRef {
        LLVMBuildCast(
            self._ref,
            op,
            val,
            dest_ty._ref,
            empty_cstr()
        )
    }

    pub unsafe fn build_gep(
        &self,
        ptr: LLVMValueRef,
        indices: &[LLVMValueRef]
    ) -> LLVMValueRef {
        let mut indices: Vec<LLVMValueRef> = indices.iter().map(|v| *v).collect();

        LLVMBuildGEP(
            self._ref,
            ptr,
            if indices.len() == 0 {
                ::std::ptr::null_mut()
            } else {
                &mut indices[0]
            },
            indices.len() as _,
            empty_cstr()
        )
    }

    pub unsafe fn build_call_raw(
        &self,
        f: LLVMValueRef,
        args: &[LLVMValueRef]
    ) -> LLVMValueRef {
        let mut args: Vec<LLVMValueRef> = args.iter().map(|v| *v).collect();
        LLVMBuildCall(
            self._ref,
            f,
            if args.len() == 0 {
                ::std::ptr::null_mut()
            } else {
                &mut args[0]
            },
            args.len() as _,
            empty_cstr()
        )
    }

    pub unsafe fn build_call(
        &self,
        f: &Function,
        args: &[LLVMValueRef]
    ) -> LLVMValueRef {
        self.build_call_raw(f._ref, args)
    }

    pub unsafe fn build_icmp(
        &self,
        op: LLVMIntPredicate,
        lhs: LLVMValueRef,
        rhs: LLVMValueRef
    ) -> LLVMValueRef {
        LLVMBuildICmp(
            self._ref,
            op,
            lhs,
            rhs,
            empty_cstr()
        )
    }

    pub unsafe fn build_fcmp(
        &self,
        op: LLVMRealPredicate,
        lhs: LLVMValueRef,
        rhs: LLVMValueRef
    ) -> LLVMValueRef {
        LLVMBuildFCmp(
            self._ref,
            op,
            lhs,
            rhs,
            empty_cstr()
        )
    }

    pub unsafe fn build_ret(&self, v: LLVMValueRef) -> LLVMValueRef {
        LLVMBuildRet(
            self._ref,
            v
        )
    }

    pub unsafe fn build_ret_void(&self) -> LLVMValueRef {
        LLVMBuildRetVoid(
            self._ref
        )
    }

    pub unsafe fn build_unreachable(&self) -> LLVMValueRef {
        LLVMBuildUnreachable(self._ref)
    }

    pub unsafe fn build_and(
        &self,
        lhs: LLVMValueRef,
        rhs: LLVMValueRef
    ) -> LLVMValueRef {
        LLVMBuildAnd(
            self._ref,
            lhs,
            rhs,
            empty_cstr()
        )
    }

    pub unsafe fn build_or(
        &self,
        lhs: LLVMValueRef,
        rhs: LLVMValueRef
    ) -> LLVMValueRef {
        LLVMBuildOr(
            self._ref,
            lhs,
            rhs,
            empty_cstr()
        )
    }

    pub unsafe fn build_xor(
        &self,
        lhs: LLVMValueRef,
        rhs: LLVMValueRef
    ) -> LLVMValueRef {
        LLVMBuildXor(
            self._ref,
            lhs,
            rhs,
            empty_cstr()
        )
    }

    pub unsafe fn build_shl(
        &self,
        lhs: LLVMValueRef,
        rhs: LLVMValueRef
    ) -> LLVMValueRef {
        LLVMBuildShl(
            self._ref,
            lhs,
            rhs,
            empty_cstr()
        )
    }

    pub unsafe fn build_ashr(
        &self,
        lhs: LLVMValueRef,
        rhs: LLVMValueRef
    ) -> LLVMValueRef {
        LLVMBuildAShr(
            self._ref,
            lhs,
            rhs,
            empty_cstr()
        )
    }

    pub unsafe fn build_lshr(
        &self,
        lhs: LLVMValueRef,
        rhs: LLVMValueRef
    ) -> LLVMValueRef {
        LLVMBuildLShr(
            self._ref,
            lhs,
            rhs,
            empty_cstr()
        )
    }

    pub unsafe fn build_add(
        &self,
        lhs: LLVMValueRef,
        rhs: LLVMValueRef
    ) -> LLVMValueRef {
        LLVMBuildAdd(
            self._ref,
            lhs,
            rhs,
            empty_cstr()
        )
    }

    pub unsafe fn build_sub(
        &self,
        lhs: LLVMValueRef,
        rhs: LLVMValueRef
    ) -> LLVMValueRef {
        LLVMBuildSub(
            self._ref,
            lhs,
            rhs,
            empty_cstr()
        )
    }

    pub unsafe fn build_mul(
        &self,
        lhs: LLVMValueRef,
        rhs: LLVMValueRef
    ) -> LLVMValueRef {
        LLVMBuildMul(
            self._ref,
            lhs,
            rhs,
            empty_cstr()
        )
    }

    pub unsafe fn build_udiv(
        &self,
        lhs: LLVMValueRef,
        rhs: LLVMValueRef
    ) -> LLVMValueRef {
        LLVMBuildUDiv(
            self._ref,
            lhs,
            rhs,
            empty_cstr()
        )
    }

    pub unsafe fn build_sdiv(
        &self,
        lhs: LLVMValueRef,
        rhs: LLVMValueRef
    ) -> LLVMValueRef {
        LLVMBuildSDiv(
            self._ref,
            lhs,
            rhs,
            empty_cstr()
        )
    }

    pub unsafe fn build_urem(
        &self,
        lhs: LLVMValueRef,
        rhs: LLVMValueRef
    ) -> LLVMValueRef {
        LLVMBuildURem(
            self._ref,
            lhs,
            rhs,
            empty_cstr()
        )
    }

    pub unsafe fn build_srem(
        &self,
        lhs: LLVMValueRef,
        rhs: LLVMValueRef
    ) -> LLVMValueRef {
        LLVMBuildSRem(
            self._ref,
            lhs,
            rhs,
            empty_cstr()
        )
    }

    pub unsafe fn build_fp_trunc(
        &self,
        v: LLVMValueRef,
        dest_ty: Type
    ) -> LLVMValueRef {
        LLVMBuildFPTrunc(
            self._ref,
            v,
            dest_ty._ref,
            empty_cstr()
        )
    }

    pub unsafe fn build_fp_ext(
        &self,
        v: LLVMValueRef,
        dest_ty: Type
    ) -> LLVMValueRef {
        LLVMBuildFPExt(
            self._ref,
            v,
            dest_ty._ref,
            empty_cstr()
        )
    }

    pub unsafe fn build_fadd(
        &self,
        lhs: LLVMValueRef,
        rhs: LLVMValueRef
    ) -> LLVMValueRef {
        LLVMBuildFAdd(
            self._ref,
            lhs,
            rhs,
            empty_cstr()
        )
    }

    pub unsafe fn build_fsub(
        &self,
        lhs: LLVMValueRef,
        rhs: LLVMValueRef
    ) -> LLVMValueRef {
        LLVMBuildFSub(
            self._ref,
            lhs,
            rhs,
            empty_cstr()
        )
    }

    pub unsafe fn build_fmul(
        &self,
        lhs: LLVMValueRef,
        rhs: LLVMValueRef
    ) -> LLVMValueRef {
        LLVMBuildFMul(
            self._ref,
            lhs,
            rhs,
            empty_cstr()
        )
    }

    pub unsafe fn build_fdiv(
        &self,
        lhs: LLVMValueRef,
        rhs: LLVMValueRef
    ) -> LLVMValueRef {
        LLVMBuildFDiv(
            self._ref,
            lhs,
            rhs,
            empty_cstr()
        )
    }

    pub unsafe fn build_switch(
        &self,
        v: LLVMValueRef,
        cases: &[(LLVMValueRef, &BasicBlock)],
        otherwise: &BasicBlock
    ) -> LLVMValueRef {
        let s = LLVMBuildSwitch(
            self._ref,
            v,
            otherwise._ref,
            cases.len() as _
        );
        for &(ref exp, ref bb) in cases {
            LLVMAddCase(
                s,
                *exp,
                bb._ref
            );
        }
        s
    }

    pub unsafe fn build_bitcast(
        &self,
        v: LLVMValueRef,
        dest_ty: Type
    ) -> LLVMValueRef {
        LLVMBuildBitCast(
            self._ref,
            v,
            dest_ty._ref,
            empty_cstr()
        )
    }
}

impl<'a> Drop for Builder<'a> {
    fn drop(&mut self) {
        unsafe {
            LLVMDisposeBuilder(self._ref);
        }
    }
}

#[derive(Eq, PartialEq)]
pub enum NativePointerWidth {
    W32,
    W64
}

impl NativePointerWidth {
    pub fn detect() -> NativePointerWidth {
        let size = ::std::mem::size_of::<usize>();
        if size == 4 {
            NativePointerWidth::W32
        } else if size == 8 {
            NativePointerWidth::W64
        } else {
            panic!("Unsupported native pointer width: {}", size);
        }
    }
}
