use prelude::{Vec, String, BTreeMap};

use value::Value;

use opcode::Opcode;
use bincode;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Module {
    pub types: Vec<Type>,
    pub functions: Vec<Function>,
    pub data_segments: Vec<DataSegment>,
    pub exports: BTreeMap<String, Export>,
    pub tables: Vec<Table>,
    pub globals: Vec<Global>,
    pub natives: Vec<Native>
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Native {
    pub module: String,
    pub field: String,
    pub typeidx: u32
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Global {
    pub value: Value
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Table {
    pub min: u32,
    pub max: Option<u32>,
    pub elements: Vec<Option<u32>>
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Export {
    Function(u32)
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub enum Type {
    Func(Vec<ValType>, Vec<ValType>) // (args...) -> (ret)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Function {
    pub name: Option<String>,
    pub typeidx: u32,
    pub locals: Vec<ValType>,
    pub body: FunctionBody
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FunctionBody {
    pub opcodes: Vec<Opcode>
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataSegment {
    pub offset: u32,
    pub data: Vec<u8>
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub enum ValType {
    I32,
    I64,
    F32,
    F64
}

impl Module {
    pub fn std_serialize(&self) -> Result<Vec<u8>, String> {
        match bincode::serialize(self) {
            Ok(v) => Ok(v),
            Err(e) => Err(format!("{:?}", e))
        }
    }

    pub fn std_deserialize(data: &[u8]) -> Result<Module, String> {
        match bincode::deserialize(data) {
            Ok(v) => Ok(v),
            Err(e) => Err(format!("{:?}", e))
        }
    }
}
