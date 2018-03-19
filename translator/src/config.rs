#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct ModuleConfig {
    pub emscripten: Option<bool>
}

impl ModuleConfig {
    pub fn with_emscripten(mut self) -> Self {
        self.emscripten = Some(true);
        self
    }
}
