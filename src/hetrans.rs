use module::*;
use opcode::{Opcode, Memarg};
use byteorder::{LittleEndian, ByteOrder};

#[derive(Copy, Clone, Debug)]
#[repr(u8)]
pub enum TargetOp {
    Drop = 1,
    Dup,
    Swap2,
    Select,

    Call,
    Return,
    Halt,

    GetLocal,
    SetLocal,
    TeeLocal,

    NativeInvoke,

    CurrentMemory,
    GrowMemory,

    Nop,
    Unreachable,
    NotSupported,

    Jmp,
    JmpIf,
    JmpEither,
    JmpTable,

    I32Load,
    I32Load8U,
    I32Load8S,
    I32Load16U,
    I32Load16S,
    I32Store,
    I32Store8,
    I32Store16,

    I32Const,
    I32Ctz,
    I32Clz,
    I32Popcnt,
    I32Add,
    I32Sub,
    I32Mul,
    I32DivU,
    I32DivS,
    I32RemU,
    I32RemS,
    I32And,
    I32Or,
    I32Xor,
    I32Shl,
    I32ShrU,
    I32ShrS,
    I32Rotl,
    I32Rotr,

    I32Eq,
    I32Ne,
    I32LtU,
    I32LtS,
    I32LeU,
    I32LeS,
    I32GtU,
    I32GtS,
    I32GeU,
    I32GeS,

    I32WrapI64,

    I64Load,
    I64Load8U,
    I64Load8S,
    I64Load16U,
    I64Load16S,
    I64Load32U,
    I64Load32S,
    I64Store,
    I64Store8,
    I64Store16,
    I64Store32,

    I64Const,
    I64Ctz,
    I64Clz,
    I64Popcnt,
    I64Add,
    I64Sub,
    I64Mul,
    I64DivU,
    I64DivS,
    I64RemU,
    I64RemS,
    I64And,
    I64Or,
    I64Xor,
    I64Shl,
    I64ShrU,
    I64ShrS,
    I64Rotl,
    I64Rotr,

    I64Eq,
    I64Ne,
    I64LtU,
    I64LtS,
    I64LeU,
    I64LeS,
    I64GtU,
    I64GtS,
    I64GeU,
    I64GeS,

    I64ExtendI32U,
    I64ExtendI32S,

    Never
}

struct Reloc {
    code_loc: usize,
    ty: RelocType
}

enum RelocType {
    Function(usize /* function id */),
    LocalJmp(usize /* local opcode index */)
}

#[derive(Debug)]
struct OffsetTable {
    table_offset: usize,
    globals_offset: usize,
    mem_offset: usize,

    table_ds_id: usize
}

struct TargetFunction {
    code: Vec<u8>,
    opcode_relocs: Vec<usize>, // source_op_id => target_op_id
    generic_relocs: Vec<Reloc>
}

pub fn translate_module(m: &Module, entry_fn: usize) -> Vec<u8> {
    let mut target_code: Vec<u8> = Vec::new();

    let (target_dss, offset_table) = build_initializers(m);
    let init_data_relocs = write_initializers(&target_dss, &mut target_code);

    eprintln!("Offsets: {:?}", offset_table);

    let mut functions: Vec<TargetFunction> = Vec::with_capacity(m.functions.len());

    for f in &m.functions {
        functions.push(translate_function(&m, f, &offset_table));
    }

    let mut function_relocs: Vec<usize> = Vec::with_capacity(functions.len());
    let mut executable: Vec<u8> = Vec::new();

    let mut entry_reloc_point = build_call(m, &mut executable, entry_fn);
    executable.push(TargetOp::Halt as u8);

    for (i, f) in functions.iter().enumerate() {
        //eprintln!("Relocating function: {} -> {}", i, executable.len());
        function_relocs.push(executable.len());
        executable.extend_from_slice(&f.code);
    }

    // Relocate entry
    LittleEndian::write_u32(
        &mut executable[entry_reloc_point .. entry_reloc_point + 4],
        function_relocs[entry_fn] as u32
    );

    // Relocate code
    for (i, f) in functions.iter().enumerate() {
        let target_section = &mut executable[function_relocs[i] .. function_relocs[i] + f.code.len()];
        for reloc in &f.generic_relocs {
            let slot = &mut target_section[reloc.code_loc .. reloc.code_loc + 4];
            match reloc.ty {
                RelocType::Function(id) => {
                    LittleEndian::write_u32(slot, function_relocs[id] as u32);
                },
                RelocType::LocalJmp(pos) => {
                    LittleEndian::write_u32(slot, (function_relocs[i] + f.opcode_relocs[pos]) as u32);
                }
            }
        }
    }

    target_code.extend_from_slice(&executable);

    // Relocate table
    assert_eq!(target_dss[offset_table.table_ds_id].data.len(), m.tables[0].elements.len() * (4 + 4));
    for i in 0..m.tables[0].elements.len() {
        let base = init_data_relocs[offset_table.table_ds_id] + i * (4 + 4);
        let elem = &mut target_code[base .. base + 4];

        let function_id = LittleEndian::read_u32(elem);
        if function_id != ::std::u32::MAX {
            LittleEndian::write_u32(elem, function_relocs[function_id as usize] as u32);
        }
    }

    target_code
}

fn build_call(m: &Module, out: &mut Vec<u8>, target: usize) -> usize /* reloc */ {
    let tf: &Function = &m.functions[target];
    let Type::Func(ref ty_args, ref ty_rets) = &m.types[tf.typeidx as usize];

    // target
    out.push(TargetOp::I32Const as u8);
    let reloc_point = out.len();
    write_u32(out, ::std::u32::MAX);

    // n_locals
    out.push(TargetOp::I32Const as u8);
    write_u32(out, tf.locals.len() as u32);

    out.push(TargetOp::Call as u8);
    write_u32(out, ty_args.len() as u32);

    reloc_point
}

fn translate_function(m: &Module, f: &Function, offset_table: &OffsetTable) -> TargetFunction {
    let mut result: Vec<u8> = Vec::new();
    let mut relocs: Vec<Reloc> = Vec::new();
    let opcodes = &f.body.opcodes;
    let mut opcode_relocs: Vec<usize> = Vec::with_capacity(opcodes.len());

    for op in opcodes {
        opcode_relocs.push(result.len());
        match *op {
            Opcode::Drop => {
                result.push(TargetOp::Drop as u8);
            },
            Opcode::Select => {
                result.push(TargetOp::Select as u8);
            },
            Opcode::Call(target) => {
                let reloc_point = build_call(m, &mut result, target as usize);
                relocs.push(Reloc {
                    code_loc: reloc_point,
                    ty: RelocType::Function(target as usize)
                });
            },
            Opcode::CallIndirect(target_ty) => {
                let Type::Func(ref ty_args, ref ty_rets) = &m.types[target_ty as usize];

                // We've got the index into table at stack top.
                // [index] * (4 + 4) + table_offset
                result.push(TargetOp::I32Const as u8);
                write_u32(&mut result, 4 + 4);
                result.push(TargetOp::I32Mul as u8);
                result.push(TargetOp::Dup as u8);

                result.push(TargetOp::I32Load as u8);
                write_u32(&mut result, offset_table.table_offset as u32); // target
                result.push(TargetOp::Swap2 as u8);
                result.push(TargetOp::I32Load as u8);
                write_u32(&mut result, offset_table.table_offset as u32 + 4); // n_locals

                // Now we have (target, n_locals)
                result.push(TargetOp::Call as u8);
                write_u32(&mut result, ty_args.len() as u32);
            },
            Opcode::Return => {
                result.push(TargetOp::Return as u8);
            },
            Opcode::Nop => {},
            Opcode::Unreachable => result.push(TargetOp::Unreachable as u8),
            Opcode::GetLocal(id) => {
                result.push(TargetOp::GetLocal as u8);
                write_u32(&mut result, id);
            },
            Opcode::SetLocal(id) => {
                result.push(TargetOp::SetLocal as u8);
                write_u32(&mut result, id);
            },
            Opcode::TeeLocal(id) => {
                result.push(TargetOp::TeeLocal as u8);
                write_u32(&mut result, id);
            },
            Opcode::GetGlobal(id) => {
                result.push(TargetOp::I32Const as u8);
                write_u32(&mut result, id * 8);
                result.push(TargetOp::I64Load as u8);
                write_u32(&mut result, offset_table.globals_offset as u32);
            },
            Opcode::SetGlobal(id) => {
                result.push(TargetOp::I32Const as u8);
                write_u32(&mut result, id * 8);
                // (val, addr)

                result.push(TargetOp::Swap2 as u8);
                // (addr, val)

                result.push(TargetOp::I64Store as u8);
                write_u32(&mut result, offset_table.globals_offset as u32);
            },
            Opcode::Jmp(loc) => {
                result.push(TargetOp::Jmp as u8);
                relocs.push(Reloc {
                    code_loc: result.len(),
                    ty: RelocType::LocalJmp(loc as usize)
                });
                write_u32(&mut result, ::std::u32::MAX);
            },
            Opcode::JmpIf(loc) => {
                result.push(TargetOp::JmpIf as u8);
                relocs.push(Reloc {
                    code_loc: result.len(),
                    ty: RelocType::LocalJmp(loc as usize)
                });
                write_u32(&mut result, ::std::u32::MAX);
            },
            Opcode::JmpEither(loc_a, loc_b) => {
                result.push(TargetOp::JmpEither as u8);
                relocs.push(Reloc {
                    code_loc: result.len(),
                    ty: RelocType::LocalJmp(loc_a as usize)
                });
                write_u32(&mut result, ::std::u32::MAX);
                relocs.push(Reloc {
                    code_loc: result.len(),
                    ty: RelocType::LocalJmp(loc_b as usize)
                });
                write_u32(&mut result, ::std::u32::MAX);
            },
            Opcode::JmpTable(ref targets, otherwise) => {
                result.push(TargetOp::JmpTable as u8);
                relocs.push(Reloc {
                    code_loc: result.len(),
                    ty: RelocType::LocalJmp(otherwise as usize)
                });
                write_u32(&mut result, ::std::u32::MAX);

                write_u32(&mut result, targets.len() as u32);
                for t in targets {
                    relocs.push(Reloc {
                        code_loc: result.len(),
                        ty: RelocType::LocalJmp(*t as usize)
                    });
                    write_u32(&mut result, ::std::u32::MAX);
                }
            },
            Opcode::CurrentMemory => {
                // [current_memory] / 65536 = n_pages
                result.push(TargetOp::CurrentMemory as u8);
                result.push(TargetOp::I32Const as u8);
                write_u32(&mut result, 65536 as u32);
                result.push(TargetOp::I32DivU as u8);
            },
            Opcode::GrowMemory => {
                // len_inc = n_pages * 65536
                result.push(TargetOp::I32Const as u8);
                write_u32(&mut result, 65536 as u32);
                result.push(TargetOp::I32Mul as u8);

                result.push(TargetOp::GrowMemory as u8);

                // [current_memory] / 65536 = n_pages
                result.push(TargetOp::I32Const as u8);
                write_u32(&mut result, 65536 as u32);
                result.push(TargetOp::I32DivU as u8);
            },
            Opcode::I32Const(v) => {
                result.push(TargetOp::I32Const as u8);
                write_u32(&mut result, v as u32);
            },
            Opcode::I32Clz => result.push(TargetOp::I32Clz as u8),
            Opcode::I32Ctz => result.push(TargetOp::I32Ctz as u8),
            Opcode::I32Popcnt => result.push(TargetOp::I32Popcnt as u8),
            Opcode::I32Add => result.push(TargetOp::I32Add as u8),
            Opcode::I32Sub => result.push(TargetOp::I32Sub as u8),
            Opcode::I32Mul => result.push(TargetOp::I32Mul as u8),
            Opcode::I32DivU => result.push(TargetOp::I32DivU as u8),
            Opcode::I32DivS => result.push(TargetOp::I32DivS as u8),
            Opcode::I32RemU => result.push(TargetOp::I32RemU as u8),
            Opcode::I32RemS => result.push(TargetOp::I32RemS as u8),
            Opcode::I32And => result.push(TargetOp::I32And as u8),
            Opcode::I32Or => result.push(TargetOp::I32Or as u8),
            Opcode::I32Xor => result.push(TargetOp::I32Xor as u8),
            Opcode::I32Shl => result.push(TargetOp::I32Shl as u8),
            Opcode::I32ShrU => result.push(TargetOp::I32ShrU as u8),
            Opcode::I32ShrS => result.push(TargetOp::I32ShrS as u8),
            Opcode::I32Rotl => result.push(TargetOp::I32Rotl as u8),
            Opcode::I32Rotr => result.push(TargetOp::I32Rotr as u8),
            Opcode::I32Eqz => {
                result.push(TargetOp::I32Const as u8);
                write_u32(&mut result, 0);
                result.push(TargetOp::I32Eq as u8);
            },
            Opcode::I32Eq => result.push(TargetOp::I32Eq as u8),
            Opcode::I32Ne => result.push(TargetOp::I32Ne as u8),
            Opcode::I32LtU => result.push(TargetOp::I32LtU as u8),
            Opcode::I32LtS => result.push(TargetOp::I32LtS as u8),
            Opcode::I32LeU => result.push(TargetOp::I32LeU as u8),
            Opcode::I32LeS => result.push(TargetOp::I32LeS as u8),
            Opcode::I32GtU => result.push(TargetOp::I32GtU as u8),
            Opcode::I32GtS => result.push(TargetOp::I32GtS as u8),
            Opcode::I32GeU => result.push(TargetOp::I32GeU as u8),
            Opcode::I32GeS => result.push(TargetOp::I32GeS as u8),
            Opcode::I32WrapI64 => result.push(TargetOp::I32WrapI64 as u8),
            Opcode::I32Load(Memarg { offset, align }) => {
                result.push(TargetOp::I32Load as u8);
                write_u32(&mut result, offset_table.mem_offset as u32 + offset);
            },
            Opcode::I32Load8U(Memarg { offset, align }) => {
                result.push(TargetOp::I32Load8U as u8);
                write_u32(&mut result, offset_table.mem_offset as u32 + offset);
            },
            Opcode::I32Load8S(Memarg { offset, align }) => {
                result.push(TargetOp::I32Load8S as u8);
                write_u32(&mut result, offset_table.mem_offset as u32 + offset);
            },
            Opcode::I32Load16U(Memarg { offset, align }) => {
                result.push(TargetOp::I32Load16U as u8);
                write_u32(&mut result, offset_table.mem_offset as u32 + offset);
            },
            Opcode::I32Load16S(Memarg { offset, align }) => {
                result.push(TargetOp::I32Load16S as u8);
                write_u32(&mut result, offset_table.mem_offset as u32 + offset);
            },
            Opcode::I32Store(Memarg { offset, align }) => {
                result.push(TargetOp::I32Store as u8);
                write_u32(&mut result, offset_table.mem_offset as u32 + offset);
            },
            Opcode::I32Store8(Memarg { offset, align }) => {
                result.push(TargetOp::I32Store8 as u8);
                write_u32(&mut result, offset_table.mem_offset as u32 + offset);
            },
            Opcode::I32Store16(Memarg { offset, align }) => {
                result.push(TargetOp::I32Store16 as u8);
                write_u32(&mut result, offset_table.mem_offset as u32 + offset);
            },
            Opcode::I64Const(v) => {
                result.push(TargetOp::I64Const as u8);
                write_u64(&mut result, v as u64);
            },
            Opcode::I64Clz => result.push(TargetOp::I64Clz as u8),
            Opcode::I64Ctz => result.push(TargetOp::I64Ctz as u8),
            Opcode::I64Popcnt => result.push(TargetOp::I64Popcnt as u8),
            Opcode::I64Add => result.push(TargetOp::I64Add as u8),
            Opcode::I64Sub => result.push(TargetOp::I64Sub as u8),
            Opcode::I64Mul => result.push(TargetOp::I64Mul as u8),
            Opcode::I64DivU => result.push(TargetOp::I64DivU as u8),
            Opcode::I64DivS => result.push(TargetOp::I64DivS as u8),
            Opcode::I64RemU => result.push(TargetOp::I64RemU as u8),
            Opcode::I64RemS => result.push(TargetOp::I64RemS as u8),
            Opcode::I64And => result.push(TargetOp::I64And as u8),
            Opcode::I64Or => result.push(TargetOp::I64Or as u8),
            Opcode::I64Xor => result.push(TargetOp::I64Xor as u8),
            Opcode::I64Shl => result.push(TargetOp::I64Shl as u8),
            Opcode::I64ShrU => result.push(TargetOp::I64ShrU as u8),
            Opcode::I64ShrS => result.push(TargetOp::I64ShrS as u8),
            Opcode::I64Rotl => result.push(TargetOp::I64Rotl as u8),
            Opcode::I64Rotr => result.push(TargetOp::I64Rotr as u8),
            Opcode::I64Eqz => {
                result.push(TargetOp::I64Const as u8);
                write_u64(&mut result, 0);
                result.push(TargetOp::I64Eq as u8);
            },
            Opcode::I64Eq => result.push(TargetOp::I64Eq as u8),
            Opcode::I64Ne => result.push(TargetOp::I64Ne as u8),
            Opcode::I64LtU => result.push(TargetOp::I64LtU as u8),
            Opcode::I64LtS => result.push(TargetOp::I64LtS as u8),
            Opcode::I64LeU => result.push(TargetOp::I64LeU as u8),
            Opcode::I64LeS => result.push(TargetOp::I64LeS as u8),
            Opcode::I64GtU => result.push(TargetOp::I64GtU as u8),
            Opcode::I64GtS => result.push(TargetOp::I64GtS as u8),
            Opcode::I64GeU => result.push(TargetOp::I64GeU as u8),
            Opcode::I64GeS => result.push(TargetOp::I64GeS as u8),
            Opcode::I64ExtendI32U => result.push(TargetOp::I64ExtendI32U as u8),
            Opcode::I64ExtendI32S => result.push(TargetOp::I64ExtendI32S as u8),
            Opcode::I64Load(Memarg { offset, align }) => {
                result.push(TargetOp::I64Load as u8);
                write_u32(&mut result, offset_table.mem_offset as u32 + offset);
            },
            Opcode::I64Load8U(Memarg { offset, align }) => {
                result.push(TargetOp::I64Load8U as u8);
                write_u32(&mut result, offset_table.mem_offset as u32 + offset);
            },
            Opcode::I64Load8S(Memarg { offset, align }) => {
                result.push(TargetOp::I64Load8S as u8);
                write_u32(&mut result, offset_table.mem_offset as u32 + offset);
            },
            Opcode::I64Load16U(Memarg { offset, align }) => {
                result.push(TargetOp::I64Load16U as u8);
                write_u32(&mut result, offset_table.mem_offset as u32 + offset);
            },
            Opcode::I64Load16S(Memarg { offset, align }) => {
                result.push(TargetOp::I64Load16S as u8);
                write_u32(&mut result, offset_table.mem_offset as u32 + offset);
            },
            Opcode::I64Load32U(Memarg { offset, align }) => {
                result.push(TargetOp::I64Load32U as u8);
                write_u32(&mut result, offset_table.mem_offset as u32 + offset);
            },
            Opcode::I64Load32S(Memarg { offset, align }) => {
                result.push(TargetOp::I64Load32S as u8);
                write_u32(&mut result, offset_table.mem_offset as u32 + offset);
            },
            Opcode::I64Store(Memarg { offset, align }) => {
                result.push(TargetOp::I64Store as u8);
                write_u32(&mut result, offset_table.mem_offset as u32 + offset);
            },
            Opcode::I64Store8(Memarg { offset, align }) => {
                result.push(TargetOp::I64Store8 as u8);
                write_u32(&mut result, offset_table.mem_offset as u32 + offset);
            },
            Opcode::I64Store16(Memarg { offset, align }) => {
                result.push(TargetOp::I64Store16 as u8);
                write_u32(&mut result, offset_table.mem_offset as u32 + offset);
            },
            Opcode::I64Store32(Memarg { offset, align }) => {
                result.push(TargetOp::I64Store32 as u8);
                write_u32(&mut result, offset_table.mem_offset as u32 + offset);
            },
            Opcode::F32Const(v) => {
                result.push(TargetOp::I32Const as u8);
                write_u32(&mut result, v as u32);
            },
            Opcode::F64Const(v) => {
                result.push(TargetOp::I64Const as u8);
                write_u64(&mut result, v as u64);
            },
            Opcode::F32ReinterpretI32 | Opcode::I32ReinterpretF32
                | Opcode::F64ReinterpretI64 | Opcode::I64ReinterpretF64 => {},
            Opcode::NativeInvoke(id) => {
                let native = &m.natives[id as usize];

                if native.module != "hexagon_e" {
                    panic!("NativeInvoke with a module other than `hexagon_e` is not supported. Got: {}", native.module);
                }

                if !native.field.starts_with("syscall_") {
                    panic!("Invalid NativeInvoke field prefix; Expecting `syscall_`");
                }

                let ni_id: u32 = native.field.splitn(2, "_").nth(1).unwrap().parse().unwrap_or_else(|_| {
                    panic!("Unable to parse NativeInvoke id");
                });

                result.push(TargetOp::NativeInvoke as u8);
                write_u32(&mut result, ni_id);
            },
            _ => {
                eprintln!("Not implemented: {:?}", op);
                result.push(TargetOp::NotSupported as u8);
            }
        }
    }

    TargetFunction {
        code: result,
        opcode_relocs: opcode_relocs,
        generic_relocs: relocs
    }
}

fn write_initializers(dss: &[DataSegment], target: &mut Vec<u8>) -> Vec<usize> /* code relocs */ {
    let mut relocs: Vec<usize> = Vec::with_capacity(dss.len());

    assert_eq!(target.len(), 0);

    // placeholder
    write_u32(target, ::std::u32::MAX);

    let initial_len = target.len(); // 4

    // (addr, len, data)
    for ds in dss {
        write_u32(target, ds.offset);
        write_u32(target, ds.data.len() as u32);
        relocs.push(target.len());
        target.extend_from_slice(&ds.data);
    }

    let actual_len = target.len() - initial_len;
    LittleEndian::write_u32(&mut target[0..4], actual_len as u32);

    relocs
}

// DataSegment with target offsets.
fn build_initializers(m: &Module) -> (Vec<DataSegment>, OffsetTable) {
    let mut dss: Vec<DataSegment> = Vec::new();

    let wasm_table = &m.tables[0];
    let wasm_globals = &m.globals;
    let wasm_table_offset: usize = 0;
    let wasm_globals_offset = wasm_table_offset + wasm_table.elements.len() * (4 + 4);
    let wasm_mem_offset = wasm_globals_offset + wasm_globals.len() * 8;

    let mut table_init_memory: Vec<u8> = Vec::new();
    for elem in &wasm_table.elements {
        let elem = elem.unwrap_or(::std::u32::MAX);
        write_u32(&mut table_init_memory, elem);
        // n_locals
        write_u32(&mut table_init_memory, if (elem as usize) < m.functions.len() {
            m.functions[elem as usize].locals.len() as u32
        } else {
            ::std::u32::MAX
        });
    }
    assert_eq!(table_init_memory.len(), wasm_table.elements.len() * (4 + 4));

    let table_ds_id = dss.len();
    dss.push(DataSegment {
        offset: wasm_table_offset as u32,
        data: table_init_memory
    });

    let mut globals_init_memory: Vec<u8> = Vec::new();
    for g in wasm_globals {
        let val = g.value.reinterpret_as_i64();
        write_u64(&mut globals_init_memory, val as u64);
    }
    assert_eq!(globals_init_memory.len(), wasm_globals.len() * 8);
    dss.push(DataSegment {
        offset: wasm_globals_offset as u32,
        data: globals_init_memory
    });

    dss.extend(m.data_segments.iter().map(|ds| {
        let mut ds = ds.clone();
        ds.offset += wasm_mem_offset as u32;
        ds
    }));

    (dss, OffsetTable {
        table_offset: wasm_table_offset,
        globals_offset: wasm_globals_offset,
        mem_offset: wasm_mem_offset,

        table_ds_id: table_ds_id
    })
}

fn write_u32(target: &mut Vec<u8>, val: u32) {
    let val = unsafe { ::std::mem::transmute::<u32, [u8; 4]>(val) };
    target.extend_from_slice(&val);
}

fn write_u64(target: &mut Vec<u8>, val: u64) {
    let val = unsafe { ::std::mem::transmute::<u64, [u8; 8]>(val) };
    target.extend_from_slice(&val);
}
