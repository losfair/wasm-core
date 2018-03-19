#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct ModuleConfig {
    pub emscripten: Option<bool>
}
