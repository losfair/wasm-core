use executor::{ExecuteResult, ExecuteError};
use module::ValType;
use fp_ops;

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum Value {
    Undef,
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64)
}

impl Default for Value {
    fn default() -> Value {
        Value::Undef
    }
}

impl Value {
    pub fn get_i32(&self) -> ExecuteResult<i32> {
        match *self {
            Value::Undef => Ok(0),
            Value::I32(v) => Ok(v),
            _ => {
                //panic!();
                Err(ExecuteError::ValueTypeMismatch)
            }
        }
    }

    pub fn get_i64(&self) -> ExecuteResult<i64> {
        match *self {
            Value::Undef => Ok(0),
            Value::I64(v) => Ok(v),
            _ => {
                //panic!();
                Err(ExecuteError::ValueTypeMismatch)
            }
        }
    }

    pub fn cast_to_i64(&self) -> i64 {
        match *self {
            Value::Undef => 0,
            Value::I32(v) => v as i64,
            Value::I64(v) => v,
            Value::F32(v) => v as i64,
            Value::F64(v) => v as i64
        }
    }

    pub fn reinterpret_as_i64(&self) -> i64 {
        unsafe {
            match *self {
                Value::Undef => 0,
                Value::I32(v) => ::prelude::mem::transmute(v as u32 as u64),
                Value::I64(v) => v,
                Value::F32(v) => ::prelude::mem::transmute(v as f64),
                Value::F64(v) => ::prelude::mem::transmute(v)
            }
        }
    }

    pub fn reinterpret_from_i64(v: i64, ty: ValType) -> Value {
        unsafe {
            match ty {
                ValType::I32 => Value::I32(v as i32),
                ValType::I64 => Value::I64(v),
                ValType::F32 => Value::F32(::std::mem::transmute(v as u64 as u32)),
                ValType::F64 => Value::F64(::std::mem::transmute(v))
            }
        }
    }
}
