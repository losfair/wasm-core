pub mod generic;

macro_rules! import_platform {
    ($arch:expr, $os:expr, $m:ident) => {
        #[cfg(all(target_arch = $arch, target_os = $os))]
        pub mod $m;
        #[cfg(all(target_arch = $arch, target_os = $os))]
        pub use self::$m as current;
    }
}

import_platform!("x86_64", "linux", x86_64_linux);
