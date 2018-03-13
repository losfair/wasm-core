use executor::{ExecuteResult, ExecuteError};

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum Value {
    Undef,
    I32(i32),
    I64(i64)
}

impl Default for Value {
    fn default() -> Value {
        Value::Undef
    }
}

impl Value {
    pub fn get_i32(&self) -> ExecuteResult<i32> {
        match *self {
            Value::I32(v) => Ok(v),
            _ => Err(ExecuteError::TypeMismatch)
        }
    }

    pub fn get_i64(&self) -> ExecuteResult<i64> {
        match *self {
            Value::I64(v) => Ok(v),
            _ => Err(ExecuteError::TypeMismatch)
        }
    }
}
