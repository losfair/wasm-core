/*#[cfg(feature = "debug_print")]
//#[macro_export]
macro_rules! dprintln {
    ($fmt:expr) => (eprintln!($fmt));
    ($fmt:expr, $($arg:tt)*) => (eprintln!($fmt, $($arg)*));
}

#[cfg(not(feature = "debug_print"))]
//#[macro_export]
*/
macro_rules! dprintln {
    ($fmt:expr) => ();
    ($fmt:expr, $($arg:tt)*) => ();
}
