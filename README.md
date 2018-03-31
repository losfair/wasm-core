# wasm-core

[![Crates.io](https://img.shields.io/crates/v/wasm-core.svg)](https://crates.io/crates/wasm-core)

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

# Bugs

See issues with the `bug` tag.

# Contribute

Contribution to this project is always welcomed! Open a pull request directly if you've fixed a bug and open a issue for discussion first if you want to add new features.
