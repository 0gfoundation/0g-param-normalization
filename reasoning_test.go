package paramnorm

import (
	"bytes"
	"encoding/json"
	"testing"
)

// newTestRow builds a row that advertises the given supportedParameters.
func newTestRow(supportedParams ...string) *Row {
	return &Row{SupportedParameters: supportedParams}
}

// newTestRowWithLevels is newTestRow for a row that also declares the
// reasoning_effort levels its chat template accepts.
func newTestRowWithLevels(levels []string, supportedParams ...string) *Row {
	r := newTestRow(supportedParams...)
	r.ReasoningEffortLevels = levels
	return r
}

func TestNormalizeReasoningEffort(t *testing.T) {
	tests := []struct {
		effort string
		want   reasoningIntent
	}{
		{"", reasoningUnset},
		{"none", reasoningOff},
		{"minimal", reasoningOff},
		{"NONE", reasoningOff},
		{"  minimal  ", reasoningOff},
		{"low", reasoningOn},
		{"medium", reasoningOn},
		{"high", reasoningOn},
		{"High", reasoningOn},
	}
	for _, tt := range tests {
		if got := normalizeReasoningEffort(tt.effort); got != tt.want {
			t.Errorf("normalizeReasoningEffort(%q) = %v, want %v", tt.effort, got, tt.want)
		}
	}
}

func TestNativeReasoningParam(t *testing.T) {
	tests := []struct {
		name   string
		params []string
		want   string
	}{
		{"detects chat_template_kwargs", []string{"temperature", "reasoning_effort", "chat_template_kwargs"}, "chat_template_kwargs"},
		{"reasoning_effort is not a target", []string{"temperature", "reasoning_effort"}, ""},
		{"no reasoning params", []string{"temperature", "top_p"}, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			row := newTestRow(tt.params...)
			if got := NativeReasoningParam(row); got != tt.want {
				t.Errorf("nativeReasoningParam() = %q, want %q", got, tt.want)
			}
		})
	}
}

// decodeEnableThinking extracts chat_template_kwargs.enable_thinking, returning
// (value, present).
func decodeEnableThinking(t *testing.T, body []byte) (bool, bool) {
	t.Helper()
	var out map[string]interface{}
	if err := json.Unmarshal(body, &out); err != nil {
		t.Fatalf("invalid json: %v", err)
	}
	kw, ok := out["chat_template_kwargs"].(map[string]interface{})
	if !ok {
		return false, false
	}
	v, present := kw["enable_thinking"]
	if !present {
		return false, false
	}
	b, _ := v.(bool)
	return b, true
}

func TestTranslateReasoning_EffortOn(t *testing.T) {
	row := newTestRow("reasoning_effort", "chat_template_kwargs")
	body := []byte(`{"model":"qwen3","reasoning_effort":"high","messages":[]}`)

	got, err := TranslateReasoning(body, row)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	on, present := decodeEnableThinking(t, got)
	if !present || !on {
		t.Errorf("enable_thinking = (%v,%v), want (true,true)", on, present)
	}
	// reasoning_effort consumed and dropped.
	var out map[string]interface{}
	_ = json.Unmarshal(got, &out)
	if _, ok := out[reasoningEffortKey]; ok {
		t.Errorf("reasoning_effort should be removed from outgoing body")
	}
}

func TestTranslateReasoning_EffortOff(t *testing.T) {
	row := newTestRow("reasoning_effort", "chat_template_kwargs")
	body := []byte(`{"model":"qwen3","reasoning_effort":"none","messages":[]}`)

	got, err := TranslateReasoning(body, row)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	on, present := decodeEnableThinking(t, got)
	if !present || on {
		t.Errorf("enable_thinking = (%v,%v), want (false,true)", on, present)
	}
}

func TestTranslateReasoning_Unset(t *testing.T) {
	row := newTestRow("reasoning_effort", "chat_template_kwargs")
	body := []byte(`{"model":"qwen3","messages":[]}`)

	got, err := TranslateReasoning(body, row)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, present := decodeEnableThinking(t, got); present {
		t.Errorf("enable_thinking should be absent when no reasoning_effort is sent")
	}
}

func TestTranslateReasoning_ExplicitNativeWins(t *testing.T) {
	row := newTestRow("reasoning_effort", "chat_template_kwargs")
	// Client sets enable_thinking=false directly AND a high effort; the explicit
	// native value must survive and reasoning_effort must not override it.
	body := []byte(`{"model":"qwen3","reasoning_effort":"high","chat_template_kwargs":{"enable_thinking":false},"messages":[]}`)

	got, err := TranslateReasoning(body, row)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	on, present := decodeEnableThinking(t, got)
	if !present || on {
		t.Errorf("enable_thinking = (%v,%v), want (false,true) — explicit native must win", on, present)
	}
	// Body returned untouched, so reasoning_effort is preserved (not consumed).
	var out map[string]interface{}
	_ = json.Unmarshal(got, &out)
	if out[reasoningEffortKey] != "high" {
		t.Errorf("reasoning_effort should be preserved when native param wins")
	}
}

func TestTranslateReasoning_NoNativeParamPassthrough(t *testing.T) {
	// Model advertises only reasoning_effort (genuine OpenAI surface): no
	// translation, body unchanged including reasoning_effort.
	row := newTestRow("reasoning_effort")
	body := []byte(`{"model":"qwen3","reasoning_effort":"high","messages":[]}`)

	got, err := TranslateReasoning(body, row)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, present := decodeEnableThinking(t, got); present {
		t.Errorf("enable_thinking must not be injected when model advertises no native param")
	}
	var out map[string]interface{}
	_ = json.Unmarshal(got, &out)
	if out[reasoningEffortKey] != "high" {
		t.Errorf("reasoning_effort should pass through untouched")
	}
}

func TestTranslateReasoning_PreservesExistingChatTemplateKwargs(t *testing.T) {
	row := newTestRow("reasoning_effort", "chat_template_kwargs")
	body := []byte(`{"model":"qwen3","reasoning_effort":"low","chat_template_kwargs":{"foo":"bar"},"messages":[]}`)

	got, err := TranslateReasoning(body, row)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var out map[string]interface{}
	if err := json.Unmarshal(got, &out); err != nil {
		t.Fatalf("invalid json: %v", err)
	}
	kw, _ := out["chat_template_kwargs"].(map[string]interface{})
	if kw["foo"] != "bar" {
		t.Errorf("existing chat_template_kwargs entries must be preserved, got %v", kw)
	}
	if kw["enable_thinking"] != true {
		t.Errorf("enable_thinking = %v, want true", kw["enable_thinking"])
	}
}

// TestTranslateReasoning_ChatTemplateEffortIsExplicit verifies that a client's
// nested chat_template_kwargs.reasoning_effort — the graded depth key
// GLM-5.2/5.3-style templates read — counts as an explicit native set just like
// enable_thinking. Otherwise the broker writes enable_thinking into the very same
// map, the template reads both keys, and the client's explicit depth is silently
// overridden (see docs/design/reasoning-translation.md).
func TestTranslateReasoning_ChatTemplateEffortIsExplicit(t *testing.T) {
	row := newTestRow("reasoning_effort", "chat_template_kwargs")
	tests := []struct {
		name   string
		effort string
		nested string
	}{
		// A portable "none" must not turn off a client that explicitly asked for
		// deep thinking through the template's own key.
		{"off effort over nested high", "none", "high"},
		// ...nor may an "on" effort layer enable_thinking next to a nested "low".
		{"on effort over nested low", "high", "low"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := []byte(`{"model":"glm-5.3","reasoning_effort":"` + tt.effort +
				`","chat_template_kwargs":{"reasoning_effort":"` + tt.nested + `"},"messages":[]}`)
			got, err := TranslateReasoning(body, row)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			var out map[string]interface{}
			if err := json.Unmarshal(got, &out); err != nil {
				t.Fatalf("invalid json: %v", err)
			}
			kw, _ := out["chat_template_kwargs"].(map[string]interface{})
			if _, ok := kw[enableThinkingKey]; ok {
				t.Errorf("must not derive enable_thinking alongside client's explicit nested reasoning_effort, got chat_template_kwargs=%v", kw)
			}
			if kw[chatTemplateEffortKey] != tt.nested {
				t.Errorf("nested reasoning_effort = %v, want %q", kw[chatTemplateEffortKey], tt.nested)
			}
			if out[reasoningEffortKey] != tt.effort {
				t.Errorf("reasoning_effort should be preserved when native param wins, got %v", out[reasoningEffortKey])
			}
		})
	}
}

func TestResolveChatTemplateEffort(t *testing.T) {
	// The three level sets actually seen in the wild, plus the no-declaration case.
	glm53 := []string{"low", "high", "max"}
	glm52 := []string{"high", "max"}
	qwen38 := []string{"low", "medium", "xhigh"}
	tests := []struct {
		name     string
		effort   string
		declared []string
		want     string
	}{
		{"no declared levels writes nothing", "low", nil, ""},
		{"off-ladder value writes nothing", "turbo", glm53, ""},
		{"exact match", "low", glm53, "low"},
		{"exact match deep", "max", glm53, "max"},
		// medium is not in GLM-5.3's set: resolves DOWN to low, never up to high —
		// verbatim forwarding would have meant max, i.e. more than "high".
		{"unrepresentable resolves down", "medium", glm53, "low"},
		{"xhigh resolves down to high", "xhigh", glm53, "high"},
		// Shallower than everything declared: the shallowest declared level.
		{"below all declared takes shallowest", "low", glm52, "high"},
		{"minimal below all declared", "minimal", glm52, "high"},
		{"qwen-style medium is exact", "medium", qwen38, "medium"},
		{"qwen-style high resolves down", "high", qwen38, "medium"},
		{"qwen-style max resolves down", "max", qwen38, "xhigh"},
		// Config spelling is normalized to the canonical ladder value.
		{"canonicalizes declared spelling", "high", []string{" High ", "MAX"}, "high"},
		{"canonicalizes client spelling", " HIGH ", glm53, "high"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := resolveChatTemplateEffort(tt.effort, tt.declared); got != tt.want {
				t.Errorf("resolveChatTemplateEffort(%q, %v) = %q, want %q", tt.effort, tt.declared, got, tt.want)
			}
		})
	}
}

// TestResolveChatTemplateEffort_Monotone guards the property the resolution rule
// exists for: walking the ladder from shallow to deep must never resolve to a
// shallower level than a shallower request did. Verbatim forwarding violates this
// (a template maps an unrecognized value to its maximum depth).
func TestResolveChatTemplateEffort_Monotone(t *testing.T) {
	for _, declared := range [][]string{{"low", "high", "max"}, {"high", "max"}, {"low", "medium", "xhigh"}, {"medium"}} {
		prev := -1
		for _, effort := range ReasoningEffortLadder {
			got := resolveChatTemplateEffort(effort, declared)
			rank, ok := ReasoningEffortRank(got)
			if !ok {
				t.Fatalf("declared=%v effort=%q resolved to %q, which is not a ladder level", declared, effort, got)
			}
			if rank < prev {
				t.Errorf("declared=%v: %q resolved to %q, shallower than the previous request's level", declared, effort, got)
			}
			prev = rank
		}
	}
}

// TestTranslateReasoning_GradedChatTemplateEffort covers the end-to-end graded
// path: a model declaring its levels gets both the enable_thinking gate and the
// resolved nested depth, and the portable field is still consumed.
func TestTranslateReasoning_GradedChatTemplateEffort(t *testing.T) {
	row := newTestRowWithLevels([]string{"low", "high", "max"},
		"reasoning_effort", "chat_template_kwargs")
	tests := []struct {
		name       string
		effort     string
		wantOn     bool
		wantNested interface{}
	}{
		{"low is honored", "low", true, "low"},
		{"high is honored", "high", true, "high"},
		{"medium resolves down", "medium", true, "low"},
		// Off is fully expressed by the gate; a depth would contradict it.
		{"none writes no depth", "none", false, nil},
		{"minimal writes no depth", "minimal", false, nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := []byte(`{"model":"glm-5.3","reasoning_effort":"` + tt.effort + `","messages":[]}`)
			got, err := TranslateReasoning(body, row)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			var out map[string]interface{}
			if err := json.Unmarshal(got, &out); err != nil {
				t.Fatalf("invalid json: %v", err)
			}
			kw, _ := out["chat_template_kwargs"].(map[string]interface{})
			if kw[enableThinkingKey] != tt.wantOn {
				t.Errorf("enable_thinking = %v, want %v", kw[enableThinkingKey], tt.wantOn)
			}
			if kw[chatTemplateEffortKey] != tt.wantNested {
				t.Errorf("nested reasoning_effort = %v, want %v", kw[chatTemplateEffortKey], tt.wantNested)
			}
			if _, ok := out[reasoningEffortKey]; ok {
				t.Errorf("reasoning_effort should be removed from outgoing body")
			}
		})
	}
}

// TestTranslateReasoning_NoDeclaredLevelsStaysBinary pins the compatibility
// promise: a model that declares no levels is translated exactly as it was before
// the graded path existed — the bool alone, no nested depth key invented for it.
func TestTranslateReasoning_NoDeclaredLevelsStaysBinary(t *testing.T) {
	row := newTestRow("reasoning_effort", "chat_template_kwargs")
	for _, effort := range []string{"low", "medium", "high", "max"} {
		t.Run(effort, func(t *testing.T) {
			body := []byte(`{"model":"qwen3","reasoning_effort":"` + effort + `","messages":[]}`)
			got, err := TranslateReasoning(body, row)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			var out map[string]interface{}
			if err := json.Unmarshal(got, &out); err != nil {
				t.Fatalf("invalid json: %v", err)
			}
			kw, _ := out["chat_template_kwargs"].(map[string]interface{})
			if kw[enableThinkingKey] != true {
				t.Errorf("enable_thinking = %v, want true", kw[enableThinkingKey])
			}
			if _, ok := kw[chatTemplateEffortKey]; ok {
				t.Errorf("must not write a nested depth for a model declaring no levels, got %v", kw)
			}
		})
	}
}

// TestTranslateReasoning_GradedRespectsExplicitNested keeps the graded path behind
// the precedence rule: a client that wrote its own nested depth still wins, so the
// broker adds neither the bool nor a resolved level next to it.
func TestTranslateReasoning_GradedRespectsExplicitNested(t *testing.T) {
	row := newTestRowWithLevels([]string{"low", "high", "max"},
		"reasoning_effort", "chat_template_kwargs")
	body := []byte(`{"model":"glm-5.3","reasoning_effort":"none","chat_template_kwargs":{"reasoning_effort":"max"},"messages":[]}`)

	got, err := TranslateReasoning(body, row)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var out map[string]interface{}
	if err := json.Unmarshal(got, &out); err != nil {
		t.Fatalf("invalid json: %v", err)
	}
	kw, _ := out["chat_template_kwargs"].(map[string]interface{})
	if _, ok := kw[enableThinkingKey]; ok {
		t.Errorf("must not derive enable_thinking over a client's explicit nested depth, got %v", kw)
	}
	if kw[chatTemplateEffortKey] != "max" {
		t.Errorf("nested reasoning_effort = %v, want \"max\" (client's own value untouched)", kw[chatTemplateEffortKey])
	}
}

func TestNativeReasoningParam_PreserveThinkingNotAToggle(t *testing.T) {
	// preserve_thinking is a multi-turn context flag, not an on/off toggle, so it
	// must not be picked as a translation target.
	row := newTestRow("temperature", "preserve_thinking")
	if got := NativeReasoningParam(row); got != "" {
		t.Errorf("nativeReasoningParam() = %q, want \"\" (preserve_thinking is not a toggle)", got)
	}
}

func TestNativeReasoningParam_AnthropicThinkingExcluded(t *testing.T) {
	// Genuine Anthropic /v1/messages "thinking" requires a mandatory budget_tokens
	// sub-field the broker does not compute — distinct from MiniMax/Zhipu's
	// type-only "thinking" on the OpenAI surface — so it must not be picked as a
	// reasoning_effort translation target.
	row := &Row{
		SupportedParameters: []string{"max_tokens", "thinking"},
		SupportedFormats:    []string{"anthropic"},
	}
	if got := NativeReasoningParam(row); got != "" {
		t.Errorf("nativeReasoningParam() = %q, want \"\" (Anthropic-surface thinking requires budget_tokens)", got)
	}
}

// TestRequiresAnthropicBudgetTokens_TrimsAndFoldsCase verifies the match is
// case-insensitive and whitespace-tolerant, mirroring proxy.go's formatAllowed
// (ModelInfo.Validate accepts a raw config value like " Anthropic" without
// normalizing the stored string, so the two checks must tolerate it the same way).
func TestRequiresAnthropicBudgetTokens_TrimsAndFoldsCase(t *testing.T) {
	tests := []struct {
		name    string
		formats []string
		want    bool
	}{
		{"exact match", []string{"anthropic"}, true},
		{"case drift", []string{"Anthropic"}, true},
		{"leading whitespace", []string{" anthropic"}, true},
		{"trailing whitespace", []string{"anthropic "}, true},
		{"case and whitespace drift", []string{" Anthropic "}, true},
		{"openai only", []string{"openai"}, false},
		{"nil formats", nil, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := requiresAnthropicBudgetTokens(tt.formats); got != tt.want {
				t.Errorf("requiresAnthropicBudgetTokens(%v) = %v, want %v", tt.formats, got, tt.want)
			}
		})
	}
}

func TestTranslateReasoning_AnthropicThinkingPassthrough(t *testing.T) {
	row := &Row{
		SupportedParameters: []string{"max_tokens", "thinking"},
		SupportedFormats:    []string{"anthropic"},
	}
	body := []byte(`{"model":"claude-opus-4-8","reasoning_effort":"high","messages":[]}`)
	got, err := TranslateReasoning(body, row)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var out map[string]interface{}
	if err := json.Unmarshal(got, &out); err != nil {
		t.Fatalf("invalid json: %v", err)
	}
	if _, ok := out["thinking"]; ok {
		t.Errorf("must not synthesize an incomplete thinking object (missing required budget_tokens), got %v", out["thinking"])
	}
	if out[reasoningEffortKey] != "high" {
		t.Errorf("reasoning_effort should pass through untouched, got %v", out[reasoningEffortKey])
	}
}

func TestTranslateReasoning_TopLevelEnableThinking(t *testing.T) {
	// DashScope dialect: top-level enable_thinking bool (not nested).
	row := newTestRow("reasoning_effort", "enable_thinking")
	body := []byte(`{"model":"qwen3","reasoning_effort":"high","messages":[]}`)

	got, err := TranslateReasoning(body, row)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var out map[string]interface{}
	if err := json.Unmarshal(got, &out); err != nil {
		t.Fatalf("invalid json: %v", err)
	}
	if out["enable_thinking"] != true {
		t.Errorf("top-level enable_thinking = %v, want true", out["enable_thinking"])
	}
	if _, ok := out["chat_template_kwargs"]; ok {
		t.Errorf("top-level dialect must not create chat_template_kwargs")
	}
}

// TestTranslateReasoning_MiniMaxThinkingObject covers a `thinking` model that
// declares no thinkingTypes: it keeps the enabled/disabled pair the broker has
// always written (MiniMax-M2.x accepts it). MiniMax-M3, which rejects "enabled",
// must declare its types — see TestTranslateReasoning_DeclaredThinkingTypes.
func TestTranslateReasoning_MiniMaxThinkingObject(t *testing.T) {
	row := newTestRow("reasoning_effort", "thinking", "reasoning_split")
	tests := []struct {
		effort   string
		wantType string
	}{
		{"high", "enabled"},
		{"none", "disabled"},
	}
	for _, tt := range tests {
		t.Run(tt.effort, func(t *testing.T) {
			body := []byte(`{"model":"MiniMax-M2.7","reasoning_effort":"` + tt.effort + `","messages":[]}`)
			got, err := TranslateReasoning(body, row)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			var out map[string]interface{}
			if err := json.Unmarshal(got, &out); err != nil {
				t.Fatalf("invalid json: %v", err)
			}
			th, ok := out["thinking"].(map[string]interface{})
			if !ok {
				t.Fatalf("thinking = %v, want object", out["thinking"])
			}
			if th["type"] != tt.wantType {
				t.Errorf("thinking.type = %v, want %q", th["type"], tt.wantType)
			}
		})
	}
}

func TestTranslateReasoning_OpenRouterReasoningObject(t *testing.T) {
	row := newTestRow("reasoning_effort", "reasoning")
	tests := []struct {
		effort string
		want   bool
	}{
		{"high", true},
		{"none", false},
	}
	for _, tt := range tests {
		t.Run(tt.effort, func(t *testing.T) {
			body := []byte(`{"model":"deepseek-v4-flash","reasoning_effort":"` + tt.effort + `","messages":[]}`)
			got, err := TranslateReasoning(body, row)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			var out map[string]interface{}
			if err := json.Unmarshal(got, &out); err != nil {
				t.Fatalf("invalid json: %v", err)
			}
			r, ok := out["reasoning"].(map[string]interface{})
			if !ok {
				t.Fatalf("reasoning = %v, want object", out["reasoning"])
			}
			if r["enabled"] != tt.want {
				t.Errorf("reasoning.enabled = %v, want %v", r["enabled"], tt.want)
			}
			if _, ok := out[reasoningEffortKey]; ok {
				t.Errorf("reasoning_effort should be removed from outgoing body")
			}
		})
	}
}

func TestTranslateReasoning_OpenRouterPreservesExistingReasoningFields(t *testing.T) {
	row := newTestRow("reasoning_effort", "reasoning")
	// "exclude" (response-visibility control) does not itself address the on/off
	// concept, so it must not count as an explicit native set — translation still
	// applies and the field is preserved through the merge.
	body := []byte(`{"model":"deepseek-v4-flash","reasoning_effort":"high","reasoning":{"exclude":true},"messages":[]}`)

	got, err := TranslateReasoning(body, row)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var out map[string]interface{}
	if err := json.Unmarshal(got, &out); err != nil {
		t.Fatalf("invalid json: %v", err)
	}
	r, _ := out["reasoning"].(map[string]interface{})
	if r["exclude"] != true {
		t.Errorf("existing reasoning.exclude must be preserved, got %v", r)
	}
	if r["enabled"] != true {
		t.Errorf("reasoning.enabled = %v, want true", r["enabled"])
	}
	if _, ok := out[reasoningEffortKey]; ok {
		t.Errorf("reasoning_effort should be removed from outgoing body")
	}
}

// TestTranslateReasoning_OpenRouterEffortAndMaxTokensAreExplicit verifies that
// "effort" and "max_tokens" — OpenRouter's own on/off-implying sub-fields — are
// treated as an explicit native set just like "enabled", so the broker never
// layers a derived enabled:false next to a client's own effort:"high" (or vice
// versa): that would send OpenRouter contradictory reasoning hints.
func TestTranslateReasoning_OpenRouterEffortAndMaxTokensAreExplicit(t *testing.T) {
	row := newTestRow("reasoning_effort", "reasoning")
	tests := []struct {
		name string
		sub  string
	}{
		{"effort", `"effort":"high"`},
		{"max_tokens", `"max_tokens":2000`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// A stale/conflicting reasoning_effort must not override the client's
			// own explicit OpenRouter-native control.
			body := []byte(`{"model":"deepseek-v4-flash","reasoning_effort":"none","reasoning":{` + tt.sub + `},"messages":[]}`)
			got, err := TranslateReasoning(body, row)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			var out map[string]interface{}
			if err := json.Unmarshal(got, &out); err != nil {
				t.Fatalf("invalid json: %v", err)
			}
			r, _ := out["reasoning"].(map[string]interface{})
			if _, ok := r["enabled"]; ok {
				t.Errorf("must not derive enabled alongside client's explicit %s, got reasoning=%v", tt.name, r)
			}
			if out[reasoningEffortKey] != "none" {
				t.Errorf("reasoning_effort should be preserved when native param wins, got %v", out[reasoningEffortKey])
			}
		})
	}
}

func TestTranslateReasoning_OpenRouterExplicitNativeWins(t *testing.T) {
	row := newTestRow("reasoning_effort", "reasoning")
	// Client sets reasoning directly AND a high effort; explicit native must win.
	body := []byte(`{"model":"deepseek-v4-flash","reasoning_effort":"high","reasoning":{"enabled":false},"messages":[]}`)

	got, err := TranslateReasoning(body, row)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var out map[string]interface{}
	if err := json.Unmarshal(got, &out); err != nil {
		t.Fatalf("invalid json: %v", err)
	}
	r, _ := out["reasoning"].(map[string]interface{})
	if r["enabled"] != false {
		t.Errorf("reasoning.enabled = %v, want false — explicit native must win", r["enabled"])
	}
	if out[reasoningEffortKey] != "high" {
		t.Errorf("reasoning_effort should be preserved when native param wins")
	}
}

func TestAdvertisedSupportedParameters(t *testing.T) {
	tests := []struct {
		name    string
		in      []string
		formats []string
		want    []string
	}{
		{
			name: "native toggle present, reasoning_effort appended",
			in:   []string{"temperature", "enable_thinking"},
			want: []string{"temperature", "enable_thinking", "reasoning_effort"},
		},
		{
			name: "nested container toggle also triggers",
			in:   []string{"chat_template_kwargs"},
			want: []string{"chat_template_kwargs", "reasoning_effort"},
		},
		{
			name: "OpenRouter object toggle also triggers",
			in:   []string{"reasoning"},
			want: []string{"reasoning", "reasoning_effort"},
		},
		{
			name: "reasoning_effort already advertised, unchanged",
			in:   []string{"reasoning_effort", "thinking"},
			want: []string{"reasoning_effort", "thinking"},
		},
		{
			name: "no native toggle (preserve_thinking only), unchanged",
			in:   []string{"temperature", "preserve_thinking"},
			want: []string{"temperature", "preserve_thinking"},
		},
		{
			name: "no reasoning params, unchanged",
			in:   []string{"temperature", "top_p"},
			want: []string{"temperature", "top_p"},
		},
		{
			name:    "MiniMax/Zhipu thinking on the OpenAI surface, reasoning_effort appended",
			in:      []string{"temperature", "thinking"},
			formats: []string{"openai"},
			want:    []string{"temperature", "thinking", "reasoning_effort"},
		},
		{
			name:    "Anthropic-surface thinking (requires budget_tokens) NOT treated as a target, unchanged",
			in:      []string{"max_tokens", "thinking"},
			formats: []string{"anthropic"},
			want:    []string{"max_tokens", "thinking"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := AdvertisedSupportedParameters(tt.in, tt.formats)
			if len(got) != len(tt.want) {
				t.Fatalf("AdvertisedSupportedParameters(%v, %v) = %v, want %v", tt.in, tt.formats, got, tt.want)
			}
			for i := range tt.want {
				if got[i] != tt.want[i] {
					t.Fatalf("AdvertisedSupportedParameters(%v, %v) = %v, want %v", tt.in, tt.formats, got, tt.want)
				}
			}
		})
	}
}

func TestAdvertisedSupportedParameters_DoesNotMutateInput(t *testing.T) {
	in := []string{"temperature", "enable_thinking"}
	_ = AdvertisedSupportedParameters(in, nil)
	if len(in) != 2 {
		t.Errorf("input slice was mutated: %v", in)
	}
}

func TestTranslateReasoning_PreservesLargeIntegers(t *testing.T) {
	// A seed above 2^53 must survive the decode→encode round-trip exactly; a plain
	// json.Unmarshal into interface{} would round it through float64 and corrupt it.
	row := newTestRow("reasoning_effort", "chat_template_kwargs")
	body := []byte(`{"model":"qwen3","reasoning_effort":"high","seed":9007199254740993,"messages":[]}`)

	got, err := TranslateReasoning(body, row)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !bytes.Contains(got, []byte(`9007199254740993`)) {
		t.Errorf("seed was corrupted in translation; got %s", got)
	}
	// And the translation still happened.
	if on, present := decodeEnableThinking(t, got); !present || !on {
		t.Errorf("enable_thinking = (%v,%v), want (true,true)", on, present)
	}
}

func TestTranslateReasoning_EmptyBody(t *testing.T) {
	row := newTestRow("reasoning_effort", "chat_template_kwargs")
	got, err := TranslateReasoning(nil, row)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("empty body should be returned unchanged, got %q", got)
	}
}

func TestNativeReasoningParam_NilModelInfo(t *testing.T) {
	// A nil row advertises nothing, so no native reasoning param can be resolved.
	var row *Row
	if got := NativeReasoningParam(row); got != "" {
		t.Errorf("nativeReasoningParam() = %q, want \"\" when row is nil", got)
	}
}

func TestTranslateReasoning_NonJSONUnchanged(t *testing.T) {
	row := newTestRow("reasoning_effort", "chat_template_kwargs")
	body := []byte(`not json`)
	got, err := TranslateReasoning(body, row)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if string(got) != "not json" {
		t.Errorf("non-JSON body should be returned unchanged, got %q", got)
	}
}

func TestResolveThinkingType(t *testing.T) {
	tests := []struct {
		name     string
		on       bool
		declared []string
		want     string
	}{
		// Undeclared: the pair the broker has always written.
		{"undeclared on", true, nil, "enabled"},
		{"undeclared off", false, nil, "disabled"},
		// MiniMax-M3 takes adaptive/disabled and rejects "enabled".
		{"m3 on", true, []string{"adaptive", "disabled"}, "adaptive"},
		{"m3 off", false, []string{"adaptive", "disabled"}, "disabled"},
		// MiniMax-M3.1-Flash cannot turn thinking off.
		{"no disabled on", true, []string{"adaptive", "enabled"}, "adaptive"},
		{"no disabled off", false, []string{"adaptive", "enabled"}, ""},
		// Declared enabled only for on.
		{"enabled on", true, []string{"enabled", "disabled"}, "enabled"},
		// Canonical spelling is written whatever the config spelling.
		{"case folded", true, []string{" Adaptive ", "DISABLED"}, "adaptive"},
		{"case folded off", false, []string{" Adaptive ", "DISABLED"}, "disabled"},
		// No on value declared (Validate rejects this; resolution stays safe).
		{"disabled only on", true, []string{"disabled"}, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := resolveThinkingType(tt.on, tt.declared); got != tt.want {
				t.Errorf("resolveThinkingType(%v, %v) = %q, want %q", tt.on, tt.declared, got, tt.want)
			}
		})
	}
}

// TestTranslateReasoning_DeclaredThinkingTypes is issue 0g-router#1229: MiniMax-M3
// answers thinking.type "enabled" with HTTP 400 (allowed: adaptive, disabled), so
// every reasoning_effort above "minimal" failed once the broker translated it.
func TestTranslateReasoning_DeclaredThinkingTypes(t *testing.T) {
	newRow := func(types ...string) *Row {
		row := newTestRow("reasoning_effort", "thinking", "reasoning_split")
		row.ThinkingTypes = types
		return row
	}
	tests := []struct {
		name     string
		types    []string
		effort   string
		wantType string // "" = body forwarded unchanged
	}{
		{"m3 high", []string{"adaptive", "disabled"}, "high", "adaptive"},
		{"m3 low", []string{"adaptive", "disabled"}, "low", "adaptive"},
		{"m3 medium", []string{"adaptive", "disabled"}, "medium", "adaptive"},
		{"m3 none", []string{"adaptive", "disabled"}, "none", "disabled"},
		{"m3 minimal", []string{"adaptive", "disabled"}, "minimal", "disabled"},
		{"cannot disable high", []string{"adaptive", "enabled"}, "high", "adaptive"},
		{"cannot disable none", []string{"adaptive", "enabled"}, "none", ""},
		{"enabled declared", []string{"enabled", "disabled"}, "high", "enabled"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := []byte(`{"model":"MiniMax-M3","reasoning_effort":"` + tt.effort + `","messages":[]}`)
			got, err := TranslateReasoning(body, newRow(tt.types...))
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if tt.wantType == "" {
				if !bytes.Equal(got, body) {
					t.Fatalf("body = %s, want it forwarded unchanged", got)
				}
				return
			}
			var out map[string]interface{}
			if err := json.Unmarshal(got, &out); err != nil {
				t.Fatalf("invalid json: %v", err)
			}
			th, ok := out["thinking"].(map[string]interface{})
			if !ok || th["type"] != tt.wantType {
				t.Errorf("thinking = %v, want {type: %q}", out["thinking"], tt.wantType)
			}
			if _, present := out["reasoning_effort"]; present {
				t.Errorf("reasoning_effort still present after translation: %v", out["reasoning_effort"])
			}
		})
	}
}

// TestTranslateReasoning_DeclaredThinkingTypesExplicitNativeWins keeps the
// precedence rule with a declaration: a client's own thinking object is
// forwarded as sent, even a value the declaration would not have chosen.
func TestTranslateReasoning_DeclaredThinkingTypesExplicitNativeWins(t *testing.T) {
	row := newTestRow("reasoning_effort", "thinking")
	row.ThinkingTypes = []string{"adaptive", "disabled"}
	body := []byte(`{"model":"MiniMax-M3","reasoning_effort":"high","thinking":{"type":"disabled"},"messages":[]}`)
	got, err := TranslateReasoning(body, row)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !bytes.Equal(got, body) {
		t.Errorf("body = %s, want it forwarded unchanged", got)
	}
}
