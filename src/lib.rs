#![no_std]
#![feature(alloc)]
#![feature(nll)]

#[macro_use]
extern crate alloc;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate bincode;

pub mod opcode;
pub mod executor;
pub mod module;
