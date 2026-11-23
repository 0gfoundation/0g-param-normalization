package paramnorm

// Shared vocabulary: parameter names, API surfaces and value ladders that both
// the translation functions and callers' config validation key off.

import (
	"slices"
	"strings"
)

// API surface format identifiers used in ModelInfo.SupportedFormats. They name
// the request surface a chat model accepts: APIFormatOpenAI is the OpenAI
// /chat/completions shape, APIFormatAnthropic is the Anthropic /v1/messages
// shape. The request path enforces these against the resolved model (the broker's
// ctrl.enforceRequestFormat) so a client can't hit a surface whose usage the
// upstream doesn't report (which would silently misbill).
const (
	APIFormatOpenAI    = "openai"
	APIFormatAnthropic = "anthropic"
)

// ParamChatTemplateKwargs is the supportedParameters entry naming the container
// vLLM/SGLang models accept for chat-template variables. Both the request-path
// translation (ctrl.nativeParamChatTemplateKwargs) and ModelInfo.Validate key off
// it, so it lives here as the single source of truth.
const ParamChatTemplateKwargs = "chat_template_kwargs"

// ParamThinking is the supportedParameters entry naming the OpenAI-surface
// `thinking` object (MiniMax, Zhipu GLM): {"type": <ThinkingType…>}. Like
// ParamChatTemplateKwargs it is shared by the request-path translation
// (ctrl.nativeParamThinking) and ModelInfo.Validate.
const ParamThinking = "thinking"

// The values a `thinking` object's "type" may take on the OpenAI surface. Which
// of them an upstream accepts differs per model, even within one vendor — see
// ModelInfo.ThinkingTypes.
const (
	ThinkingTypeEnabled  = "enabled"
	ThinkingTypeAdaptive = "adaptive"
	ThinkingTypeDisabled = "disabled"
)

// thinkingTypeValues lists the ThinkingType constants, for validation messages.
var thinkingTypeValues = []string{ThinkingTypeEnabled, ThinkingTypeAdaptive, ThinkingTypeDisabled}

// NormalizeThinkingType returns the canonical spelling of a thinking.type value
// (trimmed, lower-cased) and whether it is one the broker knows. The request
// path writes the canonical spelling, so a "Adaptive" in config still reaches
// the upstream as "adaptive".
func NormalizeThinkingType(t string) (string, bool) {
	n := strings.ToLower(strings.TrimSpace(t))
	return n, slices.Contains(thinkingTypeValues, n)
}

// ReasoningEffortLadder is the reasoning_effort ladder in increasing depth, as
// accepted by vLLM's ChatCompletionRequest. "none" is deliberately absent: it
// means "no reasoning at all" rather than a depth, and the broker expresses off
// through the dialect's own toggle (e.g. enable_thinking: false), never as a
// level. Order is significant — ReasoningEffortRank returns an index into it, and
// the request path resolves a client's request to the deepest declared level that
// does not exceed it.
var ReasoningEffortLadder = []string{"minimal", "low", "medium", "high", "xhigh", "max"}

// ReasoningEffortRank returns level's position in ReasoningEffortLadder, and
// whether it is a known level. Matching is case-insensitive and
// whitespace-trimmed, mirroring how the request path normalizes an inbound
// reasoning_effort.
func ReasoningEffortRank(level string) (int, bool) {
	normalized := strings.ToLower(strings.TrimSpace(level))
	for i, l := range ReasoningEffortLadder {
		if l == normalized {
			return i, true
		}
	}
	return 0, false
}

// IsReasoningEffortLevel reports whether level names a depth on the
// reasoning_effort ladder.
func IsReasoningEffortLevel(level string) bool {
	_, ok := ReasoningEffortRank(level)
	return ok
}

// RequestProfileSGLang is the Row.RequestProfile of an upstream served by
// SGLang; see requestProfiles for what it drops.
const RequestProfileSGLang = "sglang"
