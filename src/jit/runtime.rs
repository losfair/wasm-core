use std::cell::{UnsafeCell, RefCell};
use std::rc::Rc;
use std::os::raw::c_void;
use executor::{NativeResolver, NativeFunction, NativeEntry, NativeFunctionInfo, GlobalStateProvider};
use module::{Module, Type, ValType};
use value::Value;
use platform::current as host;
use platform::generic::MemoryManager;
use super::ondemand::Ondemand;

use smallvec::SmallVec;

pub struct Runtime {
    pub(super) opt_level: u32,
    pub(super) mm: UnsafeCell<host::NativeMemoryManager>,
    pub source_module: Module,
    function_addrs: UnsafeCell<Option<Vec<*const c_void>>>,
    globals: Box<[i64]>,
    native_functions: Box<[RefCell<NativeFunctionInfo>]>,
    native_resolver: RefCell<Option<Box<NativeResolver>>>,
    ondemand: RefCell<Option<Rc<Ondemand>>>,
    jit_info: Box<UnsafeCell<JitInfo>>
}

#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    pub mem_default: usize,
    pub mem_max: usize,
    pub opt_level: u32
}

#[repr(C)]
pub struct JitInfo {
    pub global_begin: *mut i64
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        RuntimeConfig {
            mem_default: 4096 * 1024,
            mem_max: 16384 * 1024,
            opt_level: 0
        }
    }
}

impl Runtime {
    pub fn new(cfg: RuntimeConfig, m: Module) -> Runtime {
        if cfg.mem_max < cfg.mem_default {
            panic!("mem_max < mem_default");
        }

        if cfg.mem_default == 0 {
            panic!("mem_default == 0");
        }

        let mut globals: Box<[i64]> = vec! [0; m.globals.len()].into_boxed_slice();
        for (i, g) in m.globals.iter().enumerate() {
            globals[i] = g.value.reinterpret_as_i64();
        }

        let native_functions: Vec<RefCell<NativeFunctionInfo>> = m.natives.iter()
            .map(|n| RefCell::new(NativeFunctionInfo {
                f: NativeFunction::Uninitialized(
                    n.module.clone(),
                    n.field.clone()
                ),
                typeidx: n.typeidx as usize
            }))
            .collect();

        let jit_info = JitInfo {
            global_begin: if globals.len() == 0 {
                ::std::ptr::null_mut()
            } else {
                &mut globals[0]
            }
        };

        let mut mm = host::NativeMemoryManager::new(::platform::generic::MemInitOptions {
            min: cfg.mem_default,
            max: cfg.mem_max
        });
        {
            let mem = mm.get_ref_mut();
            for ds in &m.data_segments {
                let offset = ds.offset as usize;
                mem[offset..offset + ds.data.len()].copy_from_slice(&ds.data);
            }
        }

        Runtime {
            opt_level: cfg.opt_level,
            mm: UnsafeCell::new(mm),
            source_module: m,
            function_addrs: UnsafeCell::new(None),
            globals: globals,
            native_functions: native_functions.into_boxed_slice(),
            native_resolver: RefCell::new(None),
            ondemand: RefCell::new(None),
            jit_info: Box::new(UnsafeCell::new(jit_info))
        }
    }

    pub fn set_ondemand(&self, ondemand: Rc<Ondemand>) {
        let mut current = self.ondemand.borrow_mut();
        if current.is_some() {
            panic!("Attempting to re-set ondemand");
        }

        *current = Some(ondemand);
    }

    pub fn get_function_addr(&self, id: usize) -> *const c_void {
        self.ondemand.borrow().as_ref().unwrap().get_function_addr(id)
    }

    pub fn set_native_resolver<R: NativeResolver>(&self, resolver: R) {
        *self.native_resolver.borrow_mut() = Some(Box::new(resolver));
    }

    pub fn indirect_get_function_addr(&self, id_in_table: usize) -> *const c_void {
        let id = self.source_module.tables[0].elements[id_in_table].unwrap() as usize;
        self.get_function_addr(id)
    }

    pub fn grow_memory(&self, len_inc: usize) -> usize {
        let mm = unsafe {
            &mut *self.mm.get()
        };
        let prev_len = mm.len();
        mm.grow(len_inc);
        prev_len
    }

    pub fn get_jit_info(&self) -> *mut JitInfo {
        self.jit_info.get()
    }

    pub fn protected_call<T, F: FnOnce() -> T>(&self, f: F) -> T {
        let mm = unsafe { &mut *self.mm.get() };
        mm.protected_call(|_| f())
    }

    pub(super) unsafe extern "C" fn _jit_native_invoke_request(ret_place: *mut NativeInvokeRequest, n_args: usize) {
        ::std::ptr::write(ret_place, NativeInvokeRequest::new(n_args));
    }

    pub(super) extern "C" fn _jit_native_invoke_push_arg(req: &mut NativeInvokeRequest, arg: i64) {
        req.args.push(arg);
    }

    pub(super) unsafe extern "C" fn _jit_native_invoke_complete(req: *mut NativeInvokeRequest, rt: &Runtime, id: usize) -> i64 {
        let req: NativeInvokeRequest = ::std::ptr::read(req);

        let nf = &rt.native_functions[id];
        let ty = &rt.source_module.types[nf.borrow().typeidx];
        let Type::Func(ref ty_args, ref ty_ret) = *ty;

        assert_eq!(req.args.len(), ty_args.len());

        let native_resolver = rt.native_resolver.borrow();

        let mut invoke_ctx = JitNativeInvokeContext {
            mem: unsafe {
                &mut *rt.mm.get()
            }.get_ref_mut(),
            resolver: if let Some(ref v) = *native_resolver {
                Some(&**v)
            } else {
                None
            }
        };

        let mut call_args: SmallVec<[Value; 16]> = SmallVec::with_capacity(req.args.len());

        for i in 0..req.args.len() {
            call_args.push(Value::reinterpret_from_i64(req.args[i], ty_args[i]));
        }

        let ret = nf.borrow_mut().f.invoke(&mut invoke_ctx, &call_args).unwrap();

        if let Some(ret) = ret {
            ret.reinterpret_as_i64()
        } else {
            0
        }
    }

    pub(super) extern "C" fn _jit_grow_memory(rt: &Runtime, len_inc: usize) -> usize {
        rt.grow_memory(len_inc)
    }

    pub(super) extern "C" fn _jit_get_function_addr(rt: &Runtime, id: usize) -> *const c_void {
        rt.get_function_addr(id)
    }

    pub(super) extern "C" fn _jit_indirect_get_function_addr(rt: &Runtime, id: usize) -> *const c_void {
        rt.indirect_get_function_addr(id)
    }
}

pub struct NativeInvokeRequest {
    args: SmallVec<[i64; 16]>
}

impl NativeInvokeRequest {
    fn new(n_args: usize) -> NativeInvokeRequest {
        NativeInvokeRequest {
            args: SmallVec::with_capacity(n_args)
        }
    }
}

pub struct JitNativeInvokeContext<'a> {
    mem: &'a mut [u8],
    resolver: Option<&'a NativeResolver>
}

impl<'a> GlobalStateProvider for JitNativeInvokeContext<'a> {
    fn get_memory(&self) -> &[u8] {
        self.mem
    }

    fn get_memory_mut(&mut self) -> &mut [u8] {
        self.mem
    }

    fn resolve(&self, module: &str, field: &str) -> Option<NativeEntry> {
        if let Some(r) = self.resolver {
            r.resolve(module, field)
        } else {
            None
        }
    }
}
