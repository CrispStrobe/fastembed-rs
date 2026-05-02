use super::DEFAULT_MAX_LENGTH;
use crate::{
    common::OnnxSource,
    init::{HasMaxLength, InitOptionsWithLength},
    RerankerModel, TokenizerFiles,
};
use ort::{execution_providers::ExecutionProviderDispatch, session::Session};
use tokenizers::Tokenizer;

#[derive(Debug)]
pub struct TextRerank {
    pub tokenizer: Tokenizer,
    pub(crate) session: Session,
    pub(crate) need_token_type_ids: bool,
    /// Maximum sequence length used for token-budget calculation when a
    /// prompt template is in use.
    pub(crate) max_length: usize,
    /// Optional chat-template prompt for generative-style rerankers
    /// (e.g. Zerank-1-small).  When `Some`, `{query}` and `{doc}`
    /// placeholders are substituted before tokenisation.
    pub(crate) prompt_template: Option<String>,
}

impl HasMaxLength for RerankerModel {
    const MAX_LENGTH: usize = DEFAULT_MAX_LENGTH;
}

/// Options for initializing the reranking models
pub type RerankInitOptions = InitOptionsWithLength<RerankerModel>;

/// Options for initializing UserDefinedRerankerModel
///
/// Model files are held by the UserDefinedRerankerModel struct
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RerankInitOptionsUserDefined {
    pub execution_providers: Vec<ExecutionProviderDispatch>,
    pub max_length: usize,
    /// Optional chat-template prompt with `{query}` / `{doc}` placeholders,
    /// used for generative-style rerankers such as Zerank-1-small.
    pub prompt_template: Option<String>,
}

impl Default for RerankInitOptionsUserDefined {
    fn default() -> Self {
        Self {
            execution_providers: Default::default(),
            max_length: DEFAULT_MAX_LENGTH,
            prompt_template: None,
        }
    }
}

/// Convert RerankInitOptions to RerankInitOptionsUserDefined
///
/// This is useful for when the user wants to use the same options for both the default and user-defined models
impl From<RerankInitOptions> for RerankInitOptionsUserDefined {
    fn from(options: RerankInitOptions) -> Self {
        RerankInitOptionsUserDefined {
            execution_providers: options.execution_providers,
            max_length: options.max_length,
            prompt_template: None,
        }
    }
}

/// Struct for "bring your own" reranking models
///
/// The onnx_file and tokenizer_files are expecting the files' bytes
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct UserDefinedRerankingModel {
    pub onnx_source: OnnxSource,
    pub tokenizer_files: TokenizerFiles,
}

impl UserDefinedRerankingModel {
    pub fn new(onnx_source: impl Into<OnnxSource>, tokenizer_files: TokenizerFiles) -> Self {
        Self {
            onnx_source: onnx_source.into(),
            tokenizer_files,
        }
    }
}

/// Rerank result.
#[derive(Debug, PartialEq, Clone)]
pub struct RerankResult {
    pub document: Option<String>,
    pub score: f32,
    pub index: usize,
}
