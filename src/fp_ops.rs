use executor::{ExecuteResult, ExecuteError};

#[inline]
pub fn i32_reinterpret_f32(v: i32) -> f32 {
    unsafe {
        ::std::mem::transmute(v)
    }
}

#[inline]
pub fn i64_reinterpret_f64(v: i64) -> f64 {
    unsafe {
        ::std::mem::transmute(v)
    }
}

#[inline]
pub fn f32_reinterpret_i32(v: f32) -> i32 {
    unsafe {
        ::std::mem::transmute(v)
    }
}

#[inline]
pub fn f64_reinterpret_i64(v: f64) -> i64 {
    unsafe {
        ::std::mem::transmute(v)
    }
}


#[inline]
pub fn f32_convert_i32_s(v: f32) -> ExecuteResult<i32> {
    if v.is_nan() || v.is_infinite() {
        Err(ExecuteError::FloatingPointException)
    } else {
        Ok(v as i32)
    }
}

#[inline]
pub fn f32_convert_i64_s(v: f32) -> ExecuteResult<i64> {
    if v.is_nan() || v.is_infinite() {
        Err(ExecuteError::FloatingPointException)
    } else {
        Ok(v as i64)
    }
}

#[inline]
pub fn f64_convert_i32_s(v: f64) -> ExecuteResult<i32> {
    if v.is_nan() || v.is_infinite() {
        Err(ExecuteError::FloatingPointException)
    } else {
        Ok(v as i32)
    }
}

#[inline]
pub fn f64_convert_i64_s(v: f64) -> ExecuteResult<i64> {
    if v.is_nan() || v.is_infinite() {
        Err(ExecuteError::FloatingPointException)
    } else {
        Ok(v as i64)
    }
}
