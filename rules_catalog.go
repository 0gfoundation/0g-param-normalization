package paramnorm

import "slices"

// Category is how a parameter is handled when a row cannot serve a value of it
// as sent (see the package documentation).
type Category string

const (
	// CategoryTranslate: an equivalent spelling exists; rewrite it.
	CategoryTranslate Category = "translate"
	// CategoryReject: dropping it would change cost, output structure or response
	// structure; route around rows that cannot serve it, fail if none can.
	CategoryReject Category = "reject"
	// CategoryDrop: dropping it leaves the request working; drop it and report it.
	CategoryDrop Category = "drop"
	// CategoryPassthrough: unrelated to the upstream's capabilities; forwarded as
	// sent and never judged.
	CategoryPassthrough Category = "passthrough"
	// CategoryStructural: the request itself (model, messages, input, stream,
	// ...), not a parameter of it; never judged.
	CategoryStructural Category = "structural"
	// CategoryContainer: an object whose members are judged one by one under
	// dotted names (see IsContainer). A container field whose value is not a
	// JSON object (a string, array, true) has no members to judge and is
	// unknown: refused, never forwarded unjudged. null counts as not sent.
	CategoryContainer Category = "container"
)

// TranslationLevelPrefix starts the name of a translation to another level of
// ReasoningEffortLadder ("level:high").
const TranslationLevelPrefix = "level:"

// structural are the request fields that carry the request itself. They are
// not in the catalog (rules can never name them, D6), but CategoryOf reports
// them so a caller walking a request body does not treat them as unknown.
var structural = map[string]bool{
	"model": true, "messages": true, "input": true, "instructions": true, "system": true,
	"stream": true, "stream_options": true,
	// The router's own extensions, read and removed before dispatch.
	"provider": true, "plugins": true, "attachments": true,
}

// ClassPresent is the value class of a parameter that rules judge by presence
// alone (stop, logprobs, the sampling parameters, ...).
const ClassPresent = "present"

// paramSpec is what the library knows about one parameter rules may name.
type paramSpec struct {
	category Category
	// classCategory, when set for a value class, replaces category for it: a
	// parameter can be translated as a whole yet have values that must be
	// avoided when a row cannot serve them (design section 3: thinking off on a
	// model that can only think).
	classCategory map[string]Category
	// classes are the value classes rules may give a status to. A parameter
	// judged by presence alone has exactly ClassPresent; a passthrough
	// parameter has none (rules may not name it).
	classes []string
	// translations are the names of the code-defined translations a rule may
	// pick for this parameter ("translate:<name>"). Rules can never name a target
	// field themselves (D6).
	translations []string
	// classTranslations, when set for a class, replaces translations for it.
	classTranslations map[string][]string
	// limits: rules may give this parameter a numeric range (min / max).
	limits bool
}

var present = []string{ClassPresent}

// reasoningEffortClasses: "none" plus every level of ReasoningEffortLadder.
var reasoningEffortClasses = append([]string{"none"}, ReasoningEffortLadder...)

// reasoningEffortTranslations: for a thinking level, the native thinking
// controls it can be rewritten to and another level of the ladder
// ("level:<level>"). D9: levels approximate each other (in either direction:
// the design maps medium to high where an upstream has no medium), but "none"
// (thinking off) may only become a native off switch, never a level that
// thinks. Whether a mapping is the nearest one is for the rule's reviewer.
var reasoningEffortTranslations = func() []string {
	t := []string{"enable_thinking", "thinking", "chat_template_kwargs", "output_config.effort", "thinking.budget_tokens"}
	return append(t, levelTranslations()...)
}()

func levelTranslations() []string {
	var t []string
	for _, l := range ReasoningEffortLadder {
		t = append(t, TranslationLevelPrefix+l)
	}
	return t
}

func without(list []string, drop string) []string {
	return slices.DeleteFunc(slices.Clone(list), func(v string) bool { return v == drop })
}

var thinkingOff = []string{"enable_thinking", "thinking", "chat_template_kwargs"}

var reasoningEffortClassTranslations = map[string][]string{"none": thinkingOff}

// Turning thinking off is avoided, not translated away, when a row cannot do
// it (design section 3, D9): a row that thinks anyway charges for thinking the
// caller turned off.
var thinkingOffAvoided = map[string]Category{"none": CategoryReject}

// Native thinking switches a client may send itself (Qwen / vLLM / SGLang
// chat_template_kwargs, Aliyun's top-level enable_thinking): "false" is
// thinking off, avoided when unsupported and translated only to another off
// switch (D9).
var enableThinkingClasses = []string{"true", "false"}

func nativeSwitch(translations []string) paramSpec {
	return paramSpec{
		category:      CategoryTranslate,
		classCategory: map[string]Category{"false": CategoryReject},
		classes:       enableThinkingClasses,
		translations:  translations,
	}
}

var reasoningEffort = paramSpec{
	category:          CategoryTranslate,
	classCategory:     thinkingOffAvoided,
	classes:           reasoningEffortClasses,
	translations:      reasoningEffortTranslations,
	classTranslations: reasoningEffortClassTranslations,
}

// Output formats: JSON mode and strict schemas are judged apart (D5, D12).
var outputFormatClasses = []string{"text", "json_object", "json_schema", "json_schema_strict"}

// catalog is the static parameter table (design section 3): the only
// parameters, value classes and translations rule data may name. Changing it
// changes what signed rules can express, so it lives in code and is reviewed.
// Fields that rules must never touch (messages, model, media, the broker's
// protected inject fields, engine controls) are absent, so data naming them
// fails to load.
//
// Parameters are keyed by their name on the wire, nested ones with dots:
// Chat Completions names, Anthropic Messages names (stop_sequences, thinking,
// thinking.display, thinking.budget_tokens, output_config.effort) and
// Responses names (reasoning.effort, text.format, ...). Callers look up the
// name the request carries.
//
// Value classes are lower-case and fixed per parameter (ValueClasses): callers
// classify the request's value before a lookup ("High" is not a class).
// A request whose tools are sealed (an encrypted request: the routing preview
// only says tools are present, D5) is looked up as class "function", the only
// kind Chat Completions carries.
//
// Nested parameters are named with dots; IsContainer (and CategoryOf, which
// reports CategoryContainer) says which request objects hold them (reasoning,
// text, output_config, thinking, chat_template_kwargs). A caller walking a
// body descends into those, judges each member as "<object>.<member>", and
// judges every other top-level key as itself. Anthropic thinking's class is
// thinking.type.
//
// tools classes: "function" (an Anthropic client tool, with no type or type
// "custom", is "function"); OpenAI "custom" (freeform grammar tools);
// Responses "local_shell";
// every provider-run tool ("hosted": Responses web_search, file_search,
// code_interpreter, computer_use_preview, image_generation, mcp; Anthropic
// server tools such as web_search_*, code_execution_*).
//
// Anthropic tool_choice.disable_parallel_tool_use is judged as
// parallel_tool_calls (true there is parallel_tool_calls false).
//
// service_tier in other spellings: Anthropic "auto" is "auto" and
// "standard_only" is "default".
//
// tool_choice classes in other spellings: Anthropic "any" is "required" and
// {"type":"tool"} is "function"; Responses "none" / "auto" / "required" are
// themselves and {"type":"function"} is "function". Any other form (Responses
// allowed_tools, a hosted-tool choice) has no class: unknown, so refused.
//
// "minimal" is the lowest thinking level, not thinking off (11-25 decision,
// as OpenAI and OpenRouter define it): only "none" (and thinking "disabled")
// turn thinking off.
//
// What a status means for a value, by its category (CategoryOfValue). Plan
// (a later change) implements this; it is stated here so rule authors know
// what they are writing:
//
//	category         supported  translate  ignored          rejected         unknown
//	translate        forward    rewrite+   avoid the row    avoid the row    avoid the row
//	reject / avoid   forward    rewrite+   avoid the row    avoid the row    avoid the row
//	drop             forward    rewrite+   drop and notify  drop and notify  forward as sent
//
// (rewrite+: rewrite and notify, D15.)
//
// Under require_parameters every drop-class cell other than supported avoids
// the row instead (D7). "ignored" on a value that is not dropped means the
// caller would pay for something that silently does not happen, so the row is
// avoided; a row that already behaves as asked (thinking on, sent to a row
// that always thinks) is "supported", not "ignored". An unknown drop-class
// value is forwarded as sent, as before rules existed, since rules are
// collected row by row. A translation is applied once: its result is not
// looked up again. Under require_parameters a translation to another level
// ("level:...", an approximation, D9) avoids the row; a translation that only
// respells the same value still applies. A field sent as null counts as not
// sent and is removed before dispatch.
//
// Families hold for every endpoint and every address; only row and address
// entries may name an endpoint. A supply line's conversion layer (new-api dropping
// response_format on chat for one channel type, serving tools on chat through
// responses for another) combines with the vendor's capability in the
// generator that writes the rules, which emits one entry per row and
// endpoint; the bundle carries the result, not the two facts to be merged.
// A lower layer's answer is inherited and cannot be cleared (there is no
// "unknown" status to write, and a cap cannot be removed), so when several
// rows share (identity, model) (a supply line reusing a broker's identity),
// each row's final answer goes on its own address entries, and the shared
// identity-level entries and families hold only what every such row may
// inherit.
//
// Use CategoryOfValue, not CategoryOf, to decide: several parameters handle
// some values apart from the parameter (thinking.display, tool_choice,
// reasoning_effort none, the native switches' false).
//
// The output cap (Ruleset.Limits) is one per row and is written on max_tokens
// only, even for a row that rejects the max_tokens spelling itself (its status
// says how the parameter is spelled; its limits are the row's cap).
// max_completion_tokens and max_output_tokens are capped by it. D10 applies
// across candidates: rows whose cap is below the request are avoided first;
// only when no row fits is the request truncated: the candidate with the
// largest cap goes first and the request is cut to that cap (D10), and a
// fallback is cut to its own cap (so a fallback never receives more than its cap; refused
// under require_parameters); a request below a row's min avoids that row.
// A request that sends no cap gets the row's cap injected (D6), as the
// broker's output-cap truncation does today.
var catalog = map[string]paramSpec{
	// translate
	"reasoning_effort": reasoningEffort,
	"reasoning.effort": reasoningEffort, // Responses
	// Anthropic thinking: the value class is its type.
	"thinking.type": {
		category: CategoryTranslate,
		// "between_tools": thinking off except between tool calls (the only
		// off switch some Anthropic models take); judged like "disabled".
		classCategory: map[string]Category{"disabled": CategoryReject, "between_tools": CategoryReject},
		classes:       []string{"enabled", "adaptive", "disabled", "between_tools"},
		// Each translation writes the target's own spelling of the same state:
		// "disabled" becomes reasoning_effort none or a native off switch, never
		// a level that thinks (D9); the code that performs it keeps that.
		translations:      []string{"reasoning_effort", "enable_thinking", "chat_template_kwargs"},
		classTranslations: map[string][]string{"between_tools": {"thinking"}},
	},
	"enable_thinking":                      nativeSwitch([]string{"reasoning_effort", "thinking", "chat_template_kwargs"}),
	"chat_template_kwargs.enable_thinking": nativeSwitch([]string{"reasoning_effort", "thinking", "enable_thinking"}),
	"chat_template_kwargs.thinking":        nativeSwitch([]string{"reasoning_effort", "thinking", "enable_thinking", "chat_template_kwargs"}), // DeepSeek V3.1+
	"chat_template_kwargs.thinking_budget": {category: CategoryTranslate, classes: present, translations: []string{"reasoning_effort", "thinking.budget_tokens"}},
	"thinking_budget":                      {category: CategoryTranslate, classes: present, translations: []string{"reasoning_effort", "thinking.budget_tokens", "chat_template_kwargs"}}, // Aliyun DashScope, beside enable_thinking
	// GLM-style templates. Its "chat_template_kwargs" translation writes the
	// template's own enable_thinking switch, as for reasoning_effort.
	"chat_template_kwargs.reasoning_effort": reasoningEffort,
	"thinking.budget_tokens":                {category: CategoryTranslate, classes: present, translations: []string{"reasoning_effort", "output_config.effort"}, limits: true},
	// thinking.display: "omitted" changes the response (no thinking text), so a
	// row that cannot serve it is avoided; the other values only shape what is
	// shown and are dropped (as the SGLang request profile does). Claude Code
	// also sends updates / highlights. No translation is defined yet.
	"thinking.display": {
		category:      CategoryTranslate,
		classCategory: map[string]Category{"omitted": CategoryReject, "summarized": CategoryDrop, "updates": CategoryDrop, "highlights": CategoryDrop},
		classes:       []string{"summarized", "omitted", "updates", "highlights"},
	},
	"output_config.effort": {category: CategoryTranslate, classes: slices.Clone(ReasoningEffortLadder), translations: append([]string{"reasoning_effort", "thinking.budget_tokens"}, levelTranslations()...)},
	// One output cap per row, whatever the spelling (D10, hongji A2): only
	// max_tokens carries limits in rule data, and Limits answers the other
	// spellings from it. Below a row's min is avoided; above its max, see the D10 note above the catalog.
	"max_tokens":            {category: CategoryTranslate, classes: present, translations: []string{"max_completion_tokens"}, limits: true},
	"max_completion_tokens": {category: CategoryTranslate, classes: present, translations: []string{"max_tokens"}},
	"max_output_tokens":     {category: CategoryTranslate, classes: present, translations: []string{"max_tokens", "max_completion_tokens"}}, // Responses

	// reject / avoid
	"tools": {category: CategoryReject, classes: []string{"function", "custom", "local_shell", "hosted"}},
	// Only the forcing forms and none change what comes back; a row that
	// ignores "auto" behaves as if it were sent.
	"tool_choice":          {category: CategoryReject, classCategory: map[string]Category{"auto": CategoryDrop}, classes: []string{"none", "auto", "required", "function"}},
	"response_format":      {category: CategoryReject, classes: outputFormatClasses},
	"text.format":          {category: CategoryReject, classes: outputFormatClasses},                                                                            // Responses
	"output_config.format": {category: CategoryReject, classes: outputFormatClasses},                                                                            // Anthropic structured outputs
	"functions":            {category: CategoryReject, classes: present},                                                                                        // legacy Chat spelling of tools
	"function_call":        {category: CategoryReject, classCategory: map[string]Category{"auto": CategoryDrop}, classes: []string{"none", "auto", "function"}}, // legacy tool_choice
	// Provider-run tools in other spellings, avoided like tools "hosted".
	"web_search_options": {category: CategoryReject, classes: present}, // Chat
	"enable_search":      {category: CategoryReject, classes: present}, // Aliyun
	"search_options":     {category: CategoryReject, classes: present}, // Aliyun
	"mcp_servers":        {category: CategoryReject, classes: present}, // Anthropic
	// MiniMax: moves thinking from inline text to reasoning_content, which
	// changes the response structure (as thinking.display omitted does).
	"reasoning_split":    {category: CategoryReject, classes: present},
	"container":          {category: CategoryReject, classes: present}, // Anthropic code execution
	"stop":               {category: CategoryReject, classes: present},
	"stop_sequences":     {category: CategoryReject, classes: present}, // Anthropic (D8)
	"logprobs":           {category: CategoryReject, classes: present},
	"top_logprobs":       {category: CategoryReject, classes: present},
	"n":                  {category: CategoryReject, classes: present},
	"service_tier":       {category: CategoryReject, classes: []string{"auto", "default", "flex", "priority"}},
	"context_management": {category: CategoryReject, classes: present},

	// drop and notify
	"temperature":                {category: CategoryDrop, classes: present},
	"top_p":                      {category: CategoryDrop, classes: present},
	"top_k":                      {category: CategoryDrop, classes: present},
	"frequency_penalty":          {category: CategoryDrop, classes: present},
	"presence_penalty":           {category: CategoryDrop, classes: present},
	"seed":                       {category: CategoryDrop, classes: present},
	"parallel_tool_calls":        {category: CategoryDrop, classes: present},
	"include":                    {category: CategoryDrop, classes: present}, // Responses
	"reasoning.summary":          {category: CategoryDrop, classes: present}, // Responses
	"reasoning.generate_summary": {category: CategoryDrop, classes: present}, // deprecated spelling of reasoning.summary
	"truncation":                 {category: CategoryDrop, classes: present}, // Responses
	"text.verbosity":             {category: CategoryDrop, classes: present}, // Responses
	"verbosity":                  {category: CategoryDrop, classes: present}, // Chat spelling of text.verbosity
	"thinking.clear_thinking":    {category: CategoryDrop, classes: present}, // Z.ai GLM
	"repetition_penalty":         {category: CategoryDrop, classes: present},
	"min_p":                      {category: CategoryDrop, classes: present},
	"logit_bias":                 {category: CategoryDrop, classes: present},
	// GLM template switches for earlier turns' thinking in the prompt.
	"chat_template_kwargs.clear_thinking":    {category: CategoryDrop, classes: present},
	"chat_template_kwargs.preserve_thinking": {category: CategoryDrop, classes: present},
	"preserve_thinking":                      {category: CategoryDrop, classes: present}, // top-level (DeepSeek-v4 / Qwen3.x on DashScope)

	// passthrough: forwarded as sent, never judged; rules may not name them.
	"user":              {category: CategoryPassthrough},
	"store":             {category: CategoryPassthrough},
	"metadata":          {category: CategoryPassthrough},
	"prompt_cache_key":  {category: CategoryPassthrough},
	"safety_identifier": {category: CategoryPassthrough},
	"cache_control":     {category: CategoryPassthrough}, // Anthropic top-level prompt caching hint
}

// containers are the request objects whose members the catalog names with dots.
var containers = map[string]bool{"reasoning": true, "text": true, "output_config": true, "thinking": true, "chat_template_kwargs": true}

// IsContainer reports whether a request field is an object whose members are
// judged one by one (as "<field>.<member>"); see the catalog's comment.
func IsContainer(field string) bool { return containers[field] }

// CategoryOf returns the handling category of a parameter the library knows,
// and false for any other parameter (which callers treat as reject / avoid).
// Some values of a parameter are handled differently from the parameter as a
// whole; use CategoryOfValue for a value.
func CategoryOf(param string) (Category, bool) {
	if containers[param] {
		return CategoryContainer, true
	}
	if structural[param] {
		return CategoryStructural, true
	}
	s, ok := catalog[param]
	return s.category, ok
}

// CategoryOfValue returns how one value class of a parameter is handled when a
// row cannot serve it: reasoning_effort is translated, but its "none" is
// avoided; tool_choice is avoided, but its "auto" may be dropped. It returns
// false for an unknown parameter or a class the parameter does not have
// (callers treat both as reject / avoid). Passthrough and structural
// parameters report their category for any value.
func CategoryOfValue(param, class string) (Category, bool) {
	if c, ok := CategoryOf(param); ok && (c == CategoryPassthrough || c == CategoryStructural || c == CategoryContainer) {
		return c, true
	}
	s, ok := catalog[param]
	if !ok || !s.hasClass(class) {
		return "", false
	}
	if c, ok := s.classCategory[class]; ok {
		return c, true
	}
	return s.category, true
}

// ValueClasses returns the value classes rules may give a parameter (nil for
// an unknown or passthrough parameter), for callers that classify request
// values or publish per-value capabilities (D12).
func ValueClasses(param string) []string {
	return slices.Clone(catalog[param].classes)
}

func (s paramSpec) hasClass(c string) bool { return slices.Contains(s.classes, c) }

// translationsFor returns the translations a rule may give one value class.
func (s paramSpec) translationsFor(class string) []string {
	if t, ok := s.classTranslations[class]; ok {
		return t
	}
	return s.translations
}
