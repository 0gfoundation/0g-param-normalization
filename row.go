package paramnorm

// Row is what the library needs to know about one upstream row (model ×
// upstream identity). It carries only the declaration fields the translation
// functions read; callers map their own config onto it (the broker from
// config.ModelInfo, the router and gateway from the rules data).
//
// A nil *Row means "no declaration": every function treats it as advertising
// nothing and leaves the request unchanged.
type Row struct {
	// SupportedParameters lists the request parameters the row accepts, e.g.
	// "reasoning_effort", "chat_template_kwargs", "thinking".
	SupportedParameters []string
	// SupportedFormats lists the request surfaces the row accepts
	// (APIFormatOpenAI, APIFormatAnthropic). Empty means not declared.
	SupportedFormats []string
	// ReasoningEffortLevels lists the reasoning_effort levels the row's chat
	// template accepts under chat_template_kwargs, if it declares any.
	ReasoningEffortLevels []string
	// ThinkingTypes lists the `thinking.type` values the row accepts on the
	// OpenAI surface, if it declares any.
	ThinkingTypes []string
	// MaxCompletionTokens is the row's advertised output-token limit; 0 means
	// none. Options.CapOutput clamps requests to it.
	MaxCompletionTokens int
	// ContextLength is the row's context window in tokens; 0 means unknown.
	ContextLength int
	// RequestProfile names the upstream engine's request schema (e.g.
	// RequestProfileSGLang), whose unaccepted optional values are dropped.
	// Empty means none.
	RequestProfile string
}
