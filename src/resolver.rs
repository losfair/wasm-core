use executor::{NativeResolver, NativeEntry};

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
