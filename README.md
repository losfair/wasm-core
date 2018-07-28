# wasm-core

[![Crates.io](https://img.shields.io/crates/v/wasm-core.svg)](https://crates.io/crates/wasm-core)

**NOTE: This repository is now deprecated due to multiple design issues. LLVM-generated WebAssembly code is still expected to run without problems (by July 28, 2018), but users are advised to move to other implementations for future usage.**

Portable WebAssembly implementation intended to run everywhere.

-----

# Features

### Efficient

wasm-core includes two execution engines, an interpreter and a JIT based on LLVM. While the former one is mainly for use on platforms with constrained resources or not supported by LLVM MCJIT, the latter one is designed for high performance and should be used whenever possible.

With LLVM optimizations and on-demand compilation, the JIT engine of wasm-core is able to achieve near-native performance on x86-64.

### Portable

wasm-core supports `no_std`. This means that it can run on any platforms with `libcore` and `liballoc` available, which include a lot of embedded devices and even OS kernels.

### Secure

The default execution environment is fully sandboxed, which means that user code cannot access the outside environment without explicit native imports.

### Easy integration

External functions can be easily imported by implementing the `NativeResolver` trait and exported WebAssembly functions can be easily called from outside. See [Ice Core](https://github.com/losfair/IceCore/tree/lssa) as an example.

# How to use

Instead of loading WebAssembly files directly, wasm-core takes code in an IR format generated from the raw WebAssembly code by the `wasm-translator` crate under `translator/`. 

See [Ice Core](https://github.com/losfair/IceCore/tree/lssa) as an example for executing pure-wasm code and `wa/` as an example for executing wasm generated by Emscripten. (note: support for Emscripten-generated code may be removed in the future once the WebAssembly backend of LLVM becomes stable and fully usable from clang. )

# Bugs

See issues with the `bug` tag.

# Contribute

Contribution to this project is always welcomed! Open a pull request directly if you've fixed a bug and open a issue for discussion first if you want to add new features.
