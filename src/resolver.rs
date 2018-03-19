use executor::{NativeResolver, NativeEntry, ExecuteError};
use value::Value;

pub struct NullResolver {

}

impl NativeResolver for NullResolver {
    fn resolve(&self, _module: &str, _field: &str) -> Option<NativeEntry> {
        None
    }
}

impl NullResolver {
    pub fn new() -> NullResolver {
        NullResolver {}
    }
}

pub struct EmscriptenResolver<I: NativeResolver> {
    inner: I
}

impl<I: NativeResolver> EmscriptenResolver<I> {
    pub fn new(inner: I) -> EmscriptenResolver<I> {
        EmscriptenResolver {
            inner: inner
        }
    }
}

impl<I: NativeResolver> NativeResolver for EmscriptenResolver<I> {
    fn resolve(&self, module: &str, field: &str) -> Option<NativeEntry> {
        if module != "env" {
            return self.inner.resolve(module, field);
        }

        match field {
            "abortStackOverflow" => {
                Some(Box::new(|_, _| {
                    Err(ExecuteError::Custom("Emscripten stack overflow".into()))
                }))
            },
            "getTotalMemory" => {
                Some(Box::new(|rt, _| {
                    Ok(Some(Value::I32(rt.get_memory().len() as i32)))
                }))
            },
            _ => self.inner.resolve(module, field)
        }
    }
}
