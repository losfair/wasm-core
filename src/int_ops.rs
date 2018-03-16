use core::intrinsics;
use value::Value;
use opcode::Memarg;
use executor::Memory;
use executor::{ExecuteResult, ExecuteError};

#[inline]
pub fn i32_clz(v: i32) -> Value {
    Value::I32(unsafe {
        intrinsics::ctlz(v)
    })
}

#[inline]
pub fn i32_ctz(v: i32) -> Value {
    Value::I32(unsafe {
        intrinsics::cttz(v)
    })
}

#[inline]
pub fn i32_popcnt(v: i32) -> Value {
    Value::I32(unsafe {
        intrinsics::ctpop(v)
    })
}

#[inline]
pub fn i32_add(a: i32, b: i32) -> Value {
    Value::I32(a + b)
}

#[inline]
pub fn i32_sub(a: i32, b: i32) -> Value {
    Value::I32(a - b)
}

#[inline]
pub fn i32_mul(a: i32, b: i32) -> Value {
    Value::I32(a * b)
}

#[inline]
pub fn i32_div_u(a: i32, b: i32) -> Value {
    Value::I32(((a as u32) / (b as u32)) as i32)
}

#[inline]
pub fn i32_div_s(a: i32, b: i32) -> Value {
    Value::I32(a / b)
}

#[inline]
pub fn i32_rem_u(a: i32, b: i32) -> Value {
    Value::I32(((a as u32) % (b as u32)) as i32)
}

#[inline]
pub fn i32_rem_s(a: i32, b: i32) -> Value {
    Value::I32(a % b)
}

#[inline]
pub fn i32_and(a: i32, b: i32) -> Value {
    Value::I32(a & b)
}

#[inline]
pub fn i32_or(a: i32, b: i32) -> Value {
    Value::I32(a | b)
}

#[inline]
pub fn i32_xor(a: i32, b: i32) -> Value {
    Value::I32(a ^ b)
}

#[inline]
pub fn i32_shl(a: i32, b: i32) -> Value {
    Value::I32(a << b)
}

#[inline]
pub fn i32_shr_u(a: i32, b: i32) -> Value {
    Value::I32(((a as u32) >> b) as i32)
}

#[inline]
pub fn i32_shr_s(a: i32, b: i32) -> Value {
    Value::I32(a >> b)
}

#[inline]
pub fn i32_rotl(a: i32, b: i32) -> Value {
    Value::I32(a.rotate_left(b as u32))
}

#[inline]
pub fn i32_rotr(a: i32, b: i32) -> Value {
    Value::I32(a.rotate_right(b as u32))
}

#[inline]
pub fn i32_eqz(v: i32) -> Value {
    if v == 0 {
        Value::I32(1)
    } else {
        Value::I32(0)
    }
}

#[inline]
pub fn i32_eq(a: i32, b: i32) -> Value {
    if a == b {
        Value::I32(1)
    } else {
        Value::I32(0)
    }
}

#[inline]
pub fn i32_ne(a: i32, b: i32) -> Value {
    if a == b {
        Value::I32(0)
    } else {
        Value::I32(1)
    }
}

#[inline]
pub fn i32_lt_u(a: i32, b: i32) -> Value {
    if (a as u32) < (b as u32) {
        Value::I32(1)
    } else {
        Value::I32(0)
    }
}

#[inline]
pub fn i32_lt_s(a: i32, b: i32) -> Value {
    if a < b {
        Value::I32(1)
    } else {
        Value::I32(0)
    }
}

#[inline]
pub fn i32_le_u(a: i32, b: i32) -> Value {
    if (a as u32) <= (b as u32) {
        Value::I32(1)
    } else {
        Value::I32(0)
    }
}

#[inline]
pub fn i32_le_s(a: i32, b: i32) -> Value {
    if a <= b {
        Value::I32(1)
    } else {
        Value::I32(0)
    }
}

#[inline]
pub fn i32_gt_u(a: i32, b: i32) -> Value {
    if (a as u32) > (b as u32) {
        Value::I32(1)
    } else {
        Value::I32(0)
    }
}

#[inline]
pub fn i32_gt_s(a: i32, b: i32) -> Value {
    if a > b {
        Value::I32(1)
    } else {
        Value::I32(0)
    }
}

#[inline]
pub fn i32_ge_u(a: i32, b: i32) -> Value {
    if (a as u32) >= (b as u32) {
        Value::I32(1)
    } else {
        Value::I32(0)
    }
}

#[inline]
pub fn i32_ge_s(a: i32, b: i32) -> Value {
    if a >= b {
        Value::I32(1)
    } else {
        Value::I32(0)
    }
}

#[inline]
pub fn i32_wrap_i64(a: i32) -> Value {
    Value::I64(a as i64)
}

#[inline]
pub fn i32_load(index: u32, m: &Memarg, storage: &mut Memory, n: u32) -> ExecuteResult<Value> {
    let ea = index + m.offset;
    if (ea + n) as usize > storage.data.len() {
        return Err(ExecuteError::AddrOutOfBound(ea, n));
    }

    let mut result: u32 = 0;

    // Little endian
    for i in 0..n {
        let r = n - 1 - i;
        result <<= 8;
        result |= storage.data[(ea + r) as usize] as u32;
    }

    Ok(Value::I32(result as i32))
}

#[inline]
pub fn i32_store(index: u32, val: Value, m: &Memarg, storage: &mut Memory, n: u32) -> ExecuteResult<()> {
    let ea = index + m.offset;
    if (ea + n) as usize > storage.data.len() {
        return Err(ExecuteError::AddrOutOfBound(ea, n));
    }

    let mut uni = val.get_i32()? as u32;

    // Little endian
    for i in 0..n {
        storage.data[(ea + i) as usize] = (uni & 0xff) as u8;
        uni >>= 8;
    }

    Ok(())
}

#[inline]
pub fn i64_clz(v: i64) -> Value {
    Value::I64(unsafe {
        intrinsics::ctlz(v)
    })
}

#[inline]
pub fn i64_ctz(v: i64) -> Value {
    Value::I64(unsafe {
        intrinsics::cttz(v)
    })
}

#[inline]
pub fn i64_popcnt(v: i64) -> Value {
    Value::I64(unsafe {
        intrinsics::ctpop(v)
    })
}

#[inline]
pub fn i64_add(a: i64, b: i64) -> Value {
    Value::I64(a + b)
}

#[inline]
pub fn i64_sub(a: i64, b: i64) -> Value {
    Value::I64(a - b)
}

#[inline]
pub fn i64_mul(a: i64, b: i64) -> Value {
    Value::I64(a * b)
}

#[inline]
pub fn i64_div_u(a: i64, b: i64) -> Value {
    Value::I64(((a as u32) / (b as u32)) as i64)
}

#[inline]
pub fn i64_div_s(a: i64, b: i64) -> Value {
    Value::I64(a / b)
}

#[inline]
pub fn i64_rem_u(a: i64, b: i64) -> Value {
    Value::I64(((a as u32) % (b as u32)) as i64)
}

#[inline]
pub fn i64_rem_s(a: i64, b: i64) -> Value {
    Value::I64(a % b)
}

#[inline]
pub fn i64_and(a: i64, b: i64) -> Value {
    Value::I64(a & b)
}

#[inline]
pub fn i64_or(a: i64, b: i64) -> Value {
    Value::I64(a | b)
}

#[inline]
pub fn i64_xor(a: i64, b: i64) -> Value {
    Value::I64(a ^ b)
}

#[inline]
pub fn i64_shl(a: i64, b: i64) -> Value {
    Value::I64(a << b)
}

#[inline]
pub fn i64_shr_u(a: i64, b: i64) -> Value {
    Value::I64(((a as u32) >> b) as i64)
}

#[inline]
pub fn i64_shr_s(a: i64, b: i64) -> Value {
    Value::I64(a >> b)
}

#[inline]
pub fn i64_rotl(a: i64, b: i64) -> Value {
    Value::I64(a.rotate_left(b as u32))
}

#[inline]
pub fn i64_rotr(a: i64, b: i64) -> Value {
    Value::I64(a.rotate_right(b as u32))
}

#[inline]
pub fn i64_eqz(v: i64) -> Value {
    if v == 0 {
        Value::I64(1)
    } else {
        Value::I64(0)
    }
}

#[inline]
pub fn i64_eq(a: i64, b: i64) -> Value {
    if a == b {
        Value::I64(1)
    } else {
        Value::I64(0)
    }
}

#[inline]
pub fn i64_ne(a: i64, b: i64) -> Value {
    if a == b {
        Value::I64(0)
    } else {
        Value::I64(1)
    }
}

#[inline]
pub fn i64_lt_u(a: i64, b: i64) -> Value {
    if (a as u32) < (b as u32) {
        Value::I64(1)
    } else {
        Value::I64(0)
    }
}

#[inline]
pub fn i64_lt_s(a: i64, b: i64) -> Value {
    if a < b {
        Value::I64(1)
    } else {
        Value::I64(0)
    }
}

#[inline]
pub fn i64_le_u(a: i64, b: i64) -> Value {
    if (a as u32) <= (b as u32) {
        Value::I64(1)
    } else {
        Value::I64(0)
    }
}

#[inline]
pub fn i64_le_s(a: i64, b: i64) -> Value {
    if a <= b {
        Value::I64(1)
    } else {
        Value::I64(0)
    }
}

#[inline]
pub fn i64_gt_u(a: i64, b: i64) -> Value {
    if (a as u32) > (b as u32) {
        Value::I64(1)
    } else {
        Value::I64(0)
    }
}

#[inline]
pub fn i64_gt_s(a: i64, b: i64) -> Value {
    if a > b {
        Value::I64(1)
    } else {
        Value::I64(0)
    }
}

#[inline]
pub fn i64_ge_u(a: i64, b: i64) -> Value {
    if (a as u32) >= (b as u32) {
        Value::I64(1)
    } else {
        Value::I64(0)
    }
}

#[inline]
pub fn i64_ge_s(a: i64, b: i64) -> Value {
    if a >= b {
        Value::I64(1)
    } else {
        Value::I64(0)
    }
}

#[inline]
pub fn i64_extend_i32_u(v: i64) -> Value {
    // FIXME: Is this correct?
    Value::I32((v as u32) as i32)
}

#[inline]
pub fn i64_extend_i32_s(v: i64) -> Value {
    Value::I32(v as i32)
}

#[inline]
pub fn i64_load(index: u32, m: &Memarg, storage: &mut Memory, n: u32) -> ExecuteResult<Value> {
    let ea = index + m.offset;
    if (ea + n) as usize > storage.data.len() {
        return Err(ExecuteError::AddrOutOfBound(ea, n));
    }

    let mut result: u64 = 0;

    // Little endian
    for i in 0..n {
        let r = n - 1 - i;
        result <<= 8;
        result |= storage.data[(ea + r) as usize] as u64;
    }

    Ok(Value::I64(result as i64))
}

#[inline]
pub fn i64_store(index: u32, val: Value, m: &Memarg, storage: &mut Memory, n: u32) -> ExecuteResult<()> {
    let ea = index + m.offset;
    if (ea + n) as usize > storage.data.len() {
        return Err(ExecuteError::AddrOutOfBound(ea, n));
    }

    let mut uni = val.get_i64()? as u64;

    // Little endian
    for i in 0..n {
        storage.data[(ea + i) as usize] = (uni & 0xff) as u8;
        uni >>= 8;
    }

    Ok(())
}
