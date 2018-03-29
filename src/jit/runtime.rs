use std::cell::{UnsafeCell, RefCell};
use std::os::raw::c_void;
use executor::{NativeResolver, NativeFunction, NativeEntry, NativeFunctionInfo, GlobalStateProvider};
use module::{Module, Type, ValType};
use value::Value;

pub struct Runtime {
    mem: UnsafeCell<Vec<u8>>,
    mem_max: usize,
    pub source_module: Module,
    function_addrs: UnsafeCell<Option<Vec<*const c_void>>>,
    globals: Box<[i64]>,
    native_functions: Box<[RefCell<NativeFunctionInfo>]>,
    native_resolver: RefCell<Option<Box<NativeResolver>>>,
    jit_info: Box<UnsafeCell<JitInfo>>
}

#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    pub mem_default: usize,
    pub mem_max: usize
}

#[repr(C)]
pub struct JitInfo {
    pub mem_begin: *mut u8,
    pub mem_len: usize,
    pub global_begin: *mut i64
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        RuntimeConfig {
            mem_default: 4096 * 1024,
            mem_max: 16384 * 1024
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

        let mut mem_vec: Vec<u8> = vec! [ 0; cfg.mem_default ];
        for ds in &m.data_segments {
            let offset = ds.offset as usize;
            for i in 0..ds.data.len() {
                mem_vec[offset + i] = ds.data[i];
            }
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
            mem_begin: &mut mem_vec[0],
            mem_len: mem_vec.len(),
            global_begin: if globals.len() == 0 {
                ::std::ptr::null_mut()
            } else {
                &mut globals[0]
            }
        };

        Runtime {
            mem: UnsafeCell::new(mem_vec),
            mem_max: cfg.mem_max,
            source_module: m,
            function_addrs: UnsafeCell::new(None),
            globals: globals,
            native_functions: native_functions.into_boxed_slice(),
            native_resolver: RefCell::new(None),
            jit_info: Box::new(UnsafeCell::new(jit_info))
        }
    }

    pub fn set_function_addrs(&self, new_addrs: Vec<*const c_void>) {
        unsafe {
            let addrs = &mut *self.function_addrs.get();
            *addrs = Some(new_addrs);
        }
    }

    pub fn get_function_addr(&self, id: usize) -> *const c_void {
        unsafe {
            let addrs = &*self.function_addrs.get();
            let addrs = addrs.as_ref().unwrap();
            addrs[id]
        }
    }

    pub fn set_native_resolver<R: NativeResolver>(&self, resolver: R) {
        *self.native_resolver.borrow_mut() = Some(Box::new(resolver));
    }

    pub fn indirect_get_function_addr(&self, id_in_table: usize) -> *const c_void {
        let id = self.source_module.tables[0].elements[id_in_table].unwrap() as usize;
        self.get_function_addr(id)
    }

    pub fn grow_memory(&self, len_inc: usize) -> usize {
        unsafe {
            let mem: &mut Vec<u8> = &mut *self.mem.get();
            let prev_len = mem.len();

            if mem.len().checked_add(len_inc).unwrap() > self.mem_max {
                panic!("Memory limit exceeded");
            }
            mem.extend((0..len_inc).map(|_| 0));

            let jit_info = &mut *self.jit_info.get();
            jit_info.mem_begin = &mut mem[0];
            jit_info.mem_len = mem.len();

            prev_len
        }
    }

    pub fn get_jit_info(&self) -> *mut JitInfo {
        self.jit_info.get()
    }

    pub(super) extern "C" fn _jit_native_invoke_request(n_args: usize) -> *mut NativeInvokeRequest {
        Box::into_raw(Box::new(NativeInvokeRequest::new(n_args)))
    }

    pub(super) extern "C" fn _jit_native_invoke_push_arg(req: &mut NativeInvokeRequest, arg: i64) {
        req.args.push(arg);
    }

    pub(super) extern "C" fn _jit_native_invoke_complete(rt: &Runtime, id: usize, req: *mut NativeInvokeRequest) -> i64 {
        let req = unsafe { Box::from_raw(req) };
        let nf = &rt.native_functions[id];
        let ty = &rt.source_module.types[nf.borrow().typeidx];
        let Type::Func(ref ty_args, ref ty_ret) = *ty;

        assert_eq!(req.args.len(), ty_args.len());

        let native_resolver = rt.native_resolver.borrow();

        let mut invoke_ctx = JitNativeInvokeContext {
            mem: unsafe { &mut *rt.mem.get() },
            resolver: if let Some(ref v) = *native_resolver {
                Some(&**v)
            } else {
                None
            }
        };

        let mut call_args: Vec<Value> = Vec::new();

        for i in 0..req.args.len() {
            call_args.push(unsafe {
                match ty_args[i] {
                    ValType::I32 => Value::I32(req.args[i] as i32),
                    ValType::I64 => Value::I64(req.args[i]),
                    ValType::F32 => Value::F32(::std::mem::transmute(req.args[i] as u64 as u32)),
                    ValType::F64 => Value::F64(::std::mem::transmute(req.args[i]))
                }
            });
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
    args: Vec<i64>
}

impl NativeInvokeRequest {
    fn new(n_args: usize) -> NativeInvokeRequest {
        NativeInvokeRequest {
            args: Vec::with_capacity(n_args)
        }
    }
}

pub struct JitNativeInvokeContext<'a> {
    mem: &'a mut Vec<u8>,
    resolver: Option<&'a NativeResolver>
}

impl<'a> GlobalStateProvider for JitNativeInvokeContext<'a> {
    fn get_memory(&self) -> &[u8] {
        self.mem.as_slice()
    }

    fn get_memory_mut(&mut self) -> &mut [u8] {
        self.mem.as_mut_slice()
    }

    fn resolve(&self, module: &str, field: &str) -> Option<NativeEntry> {
        if let Some(r) = self.resolver {
            r.resolve(module, field)
        } else {
            None
        }
    }
}
