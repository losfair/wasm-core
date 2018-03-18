use prelude::intrinsics;
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
    Value::I32(a.wrapping_shl((b as u32) & 31))
}

#[inline]
pub fn i32_shr_u(a: i32, b: i32) -> Value {
    Value::I32(((a as u32).wrapping_shr((b as u32) & 31)) as i32)
}

#[inline]
pub fn i32_shr_s(a: i32, b: i32) -> Value {
    Value::I32(a.wrapping_shr((b as u32) & 31))
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
pub fn i32_wrap_i64(a: i64) -> Value {
    Value::I32(a as i32)
}

unsafe trait LoadStore: Copy + Sized {}
unsafe impl LoadStore for i32 {}
unsafe impl LoadStore for i64 {}

#[inline]
fn load_from_mem<T: LoadStore>(index: u32, m: &Memarg, storage: &mut Memory, n: u32) -> ExecuteResult<T> {
    let n = n as usize;

    let t_size = ::prelude::mem::size_of::<T>();
    if n > t_size {
        return Err(ExecuteError::InvalidMemoryOperation);
    }

    let data: &[u8] = storage.data.as_slice();

    let ea = (index + m.offset) as usize;
    if ea + n > data.len() {
        return Err(ExecuteError::AddrOutOfBound(ea as u32, n as u32));
    }

    // n <= sizeof(T) holds here so we can copy safely.
    unsafe {
        let mut result: T = ::prelude::mem::zeroed();
        ::prelude::ptr::copy(
            &data[ea] as *const u8,
            &mut result as *mut T as *mut u8,
            n
        );

        Ok(result)
    }
}

#[inline]
fn store_to_mem<T: LoadStore>(index: u32, val: T, m: &Memarg, storage: &mut Memory, n: u32) -> ExecuteResult<()> {
    let n = n as usize;

    let t_size = ::prelude::mem::size_of::<T>();
    if n > t_size {
        return Err(ExecuteError::InvalidMemoryOperation);
    }

    let data: &mut [u8] = storage.data.as_mut_slice();

    let ea = (index + m.offset) as usize;

    // this will not overflow because all of index, m.offset
    // and n is in the range of u32.
    if ea + n > data.len() {
        return Err(ExecuteError::AddrOutOfBound(ea as u32, n as u32));
    }

    // ea + n <= data.len() && n <= sizeof(T) holds here so we can copy safely.
    unsafe {
        ::prelude::ptr::copy(
            &val as *const T as *const u8,
            &mut data[ea] as *mut u8,
            n
        );
    }

    Ok(())
}

#[inline]
pub fn i32_load(index: u32, m: &Memarg, storage: &mut Memory, n: u32) -> ExecuteResult<Value> {
    Ok(Value::I32(load_from_mem(index, m, storage, n)?))
}

#[inline]
pub fn i32_store(index: u32, val: Value, m: &Memarg, storage: &mut Memory, n: u32) -> ExecuteResult<()> {
    store_to_mem(index, val.get_i32()?, m, storage, n)
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
    Value::I64(((a as u64).wrapping_shr(b as u32)) as i64)
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
        Value::I32(1)
    } else {
        Value::I32(0)
    }
}

#[inline]
pub fn i64_eq(a: i64, b: i64) -> Value {
    if a == b {
        Value::I32(1)
    } else {
        Value::I32(0)
    }
}

#[inline]
pub fn i64_ne(a: i64, b: i64) -> Value {
    if a == b {
        Value::I32(0)
    } else {
        Value::I32(1)
    }
}

#[inline]
pub fn i64_lt_u(a: i64, b: i64) -> Value {
    if (a as u32) < (b as u32) {
        Value::I32(1)
    } else {
        Value::I32(0)
    }
}

#[inline]
pub fn i64_lt_s(a: i64, b: i64) -> Value {
    if a < b {
        Value::I32(1)
    } else {
        Value::I32(0)
    }
}

#[inline]
pub fn i64_le_u(a: i64, b: i64) -> Value {
    if (a as u32) <= (b as u32) {
        Value::I32(1)
    } else {
        Value::I32(0)
    }
}

#[inline]
pub fn i64_le_s(a: i64, b: i64) -> Value {
    if a <= b {
        Value::I32(1)
    } else {
        Value::I32(0)
    }
}

#[inline]
pub fn i64_gt_u(a: i64, b: i64) -> Value {
    if (a as u32) > (b as u32) {
        Value::I32(1)
    } else {
        Value::I32(0)
    }
}

#[inline]
pub fn i64_gt_s(a: i64, b: i64) -> Value {
    if a > b {
        Value::I32(1)
    } else {
        Value::I32(0)
    }
}

#[inline]
pub fn i64_ge_u(a: i64, b: i64) -> Value {
    if (a as u32) >= (b as u32) {
        Value::I32(1)
    } else {
        Value::I32(0)
    }
}

#[inline]
pub fn i64_ge_s(a: i64, b: i64) -> Value {
    if a >= b {
        Value::I32(1)
    } else {
        Value::I32(0)
    }
}

#[inline]
pub fn i64_extend_i32_u(v: i32) -> Value {
    // FIXME: Is this correct?
    Value::I64((v as i64) & 0x00000000ffffffffi64)
}

#[inline]
pub fn i64_extend_i32_s(v: i32) -> Value {
    Value::I64(v as i64)
}

#[inline]
pub fn i64_load(index: u32, m: &Memarg, storage: &mut Memory, n: u32) -> ExecuteResult<Value> {
    Ok(Value::I64(load_from_mem(index, m, storage, n)?))
}

#[inline]
pub fn i64_store(index: u32, val: Value, m: &Memarg, storage: &mut Memory, n: u32) -> ExecuteResult<()> {
    store_to_mem(index, val.get_i64()?, m, storage, n)
}
