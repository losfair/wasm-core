#![cfg_attr(not(feature = "std"), no_std)]

#![feature(alloc)]
#![feature(nll)]
#![feature(core_intrinsics)]
#![feature(underscore_lifetimes)]
#![feature(test)]

#[cfg(test)]
extern crate test;

#[cfg(not(feature = "std"))]
#[macro_use]
extern crate alloc;

extern crate serde;

#[macro_use]
extern crate serde_derive;

#[cfg(not(feature = "std"))]
extern crate bincode_no_std;
#[cfg(not(feature = "std"))]
use bincode_no_std as bincode;

#[cfg(feature = "std")]
extern crate bincode;

#[macro_use]
extern crate lazy_static;

#[cfg(feature = "jit")]
extern crate llvm_sys;

#[cfg(feature = "jit")]
pub mod jit;

#[cfg(feature = "std")]
mod prelude;

#[cfg(not(feature = "std"))]
mod prelude_no_std;
#[cfg(not(feature = "std"))]
use prelude_no_std as prelude;

pub mod opcode;
pub mod executor;
pub mod module;
pub mod int_ops;
pub mod value;
pub mod resolver;
pub mod fp_ops;
pub mod cfgraph;
pub mod optimizers;
