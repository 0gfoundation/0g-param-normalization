package paramnorm

import (
	"bytes"
	"encoding/json"
	"errors"
	"slices"
	"testing"
)

// routerOptions is what the router passes to keep today's stripEngineControls
// behaviour (0g-router backend/pkg/inference/chat_template_kwargs.go).
var routerOptions = SanitizeOptions{
	ExtraFields:     []string{"bootstrap_host", "bootstrap_port", "bootstrap_room", "kv_transfer_params"},
	DropEmptyKwargs: true,
	NoEscapeHTML:    true,
}

func mustSanitize(t *testing.T, body string, opts SanitizeOptions) []byte {
	t.Helper()
	out, err := Sanitize([]byte(body), opts)
	if err != nil {
		t.Fatalf("Sanitize(%s): %v", body, err)
	}
	return out
}

func decode(t *testing.T, b []byte) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("decode %s: %v", b, err)
	}
	return m
}

func jsonEqual(t *testing.T, want string, got []byte) {
	t.Helper()
	var w, g any
	if err := json.Unmarshal([]byte(want), &w); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(got, &g); err != nil {
		t.Fatalf("decode %s: %v", got, err)
	}
	wb, _ := json.Marshal(w)
	gb, _ := json.Marshal(g)
	if !bytes.Equal(wb, gb) {
		t.Errorf("got %s, want %s", got, want)
	}
}

// From the router's TestStripEngineControls (the sealed-body case stays in the
// router, which skips sealed bodies before calling Sanitize).
func TestSanitize_Router(t *testing.T) {
	body := `{"model":"m","messages":[{"role":"user","content":"hi"}],"chat_template":"{{x}}",` +
		`"custom_logit_processor":"gASV","lora_path":"/tmp/lora_zzz","return_hidden_states":true,"lora_adapter_name":"a","bootstrap_host":"10.0.0.1","bootstrap_port":8998,"kv_transfer_params":{"remote_host":"10.0.0.2"},` +
		`"chat_template_kwargs":{"enable_thinking":false,"chat_template":"{{ raise_exception(6*7) }}","raise_exception":1}}`
	got := decode(t, mustSanitize(t, body, routerOptions))
	for _, f := range slices.Concat(engineControlFields, routerOptions.ExtraFields) {
		if _, ok := got[f]; ok {
			t.Errorf("%s not stripped", f)
		}
	}
	if kw, _ := json.Marshal(got["chat_template_kwargs"]); string(kw) != `{"enable_thinking":false}` {
		t.Errorf("chat_template_kwargs = %s", kw)
	}
	if got["model"] != "m" {
		t.Errorf("model lost: %v", got)
	}

	// the prompt's bytes survive the re-encode (no < for <)
	if out := mustSanitize(t, `{"messages":[{"role":"user","content":"<b>a & b</b>"}],"lora_path":"x"}`, routerOptions); !bytes.Contains(out, []byte(`<b>a & b</b>`)) {
		t.Errorf("HTML escaped: %s", out)
	}
	// a field alone in the body (the cheap pre-check must not skip it)
	jsonEqual(t, `{"model":"m"}`, mustSanitize(t, `{"model":"m","chat\u005ftemplate_kwargs":{"chat\u005ftemplate":"x"}}`, routerOptions))
	jsonEqual(t, `{"model":"m"}`, mustSanitize(t, `{"model":"m","lora_adapter_name":"a"}`, routerOptions))
	// nothing to strip: the body is returned as sent
	plain := `{"model":"m","chat_template_kwargs":{"enable_thinking":true}}`
	if out := mustSanitize(t, plain, routerOptions); string(out) != plain {
		t.Errorf("clean body changed: %s", out)
	}
	// only disallowed kwargs, or kwargs not an object: the field goes
	for _, b := range []string{`{"chat_template_kwargs":{"chat_template":"x"}}`, `{"chat_template_kwargs":"x"}`} {
		if _, ok := decode(t, mustSanitize(t, b, routerOptions))["chat_template_kwargs"]; ok {
			t.Errorf("%s: chat_template_kwargs kept", b)
		}
	}
}

// From the broker's TestGuardEngineControls_StripsControlFields.
func TestSanitize_BrokerStripsControlFields(t *testing.T) {
	in := `{"model":"m","messages":[{"role":"user","content":"hi"}],` +
		`"chat_template":"{{x}}","custom_logit_processor":"gASV","lora_path":"/etc","lora_adapter_name":"a","return_hidden_states":true,` +
		`"chat_template_kwargs":{"chat_template":"{{ raise_exception('HELLO_XYZ') }}","tools":[],"enable_thinking":false,"reasoning_effort":"high"}}`
	got := decode(t, mustSanitize(t, in, SanitizeOptions{}))
	for _, f := range engineControlFields {
		if _, ok := got[f]; ok {
			t.Errorf("%s not stripped", f)
		}
	}
	kw := got["chat_template_kwargs"].(map[string]any)
	if _, ok := kw["chat_template"]; ok {
		t.Error("chat_template_kwargs.chat_template not stripped")
	}
	if _, ok := kw["tools"]; ok {
		t.Error("chat_template_kwargs.tools not stripped")
	}
	if kw["enable_thinking"] != false || kw["reasoning_effort"] != "high" {
		t.Errorf("template variables lost: %v", kw)
	}
	if got["model"] != "m" || got["messages"] == nil {
		t.Errorf("other fields lost: %v", got)
	}
}

// From the broker's TestGuardEngineControls_CleanBodyUnchanged (strip part).
func TestSanitize_CleanBodyUnchanged(t *testing.T) {
	in := `{"model":"m","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/a.png"}}]}],"chat_template_kwargs":{"enable_thinking":true}}`
	for _, opts := range []SanitizeOptions{{}, routerOptions} {
		if out := mustSanitize(t, in, opts); string(out) != in {
			t.Errorf("clean body changed: %s", out)
		}
		if out, err := Sanitize(nil, opts); err != nil || out != nil {
			t.Errorf("empty body: err=%v out=%s", err, out)
		}
	}
}

// From the broker's TestGuardEngineControls_TemplateKwargsAllowlist.
func TestSanitize_TemplateKwargsAllowlist(t *testing.T) {
	in := `{"messages":[],"chat_template_kwargs":{"enable_thinking":true,"thinking_budget":64,"add_generation_prompt":false,"documents":[],"bos_token":"x","custom_tools":"y"}}`
	for _, opts := range []SanitizeOptions{{}, routerOptions} {
		var got struct {
			KW map[string]any `json:"chat_template_kwargs"`
		}
		if err := json.Unmarshal(mustSanitize(t, in, opts), &got); err != nil {
			t.Fatal(err)
		}
		if len(got.KW) != 2 || got.KW["enable_thinking"] != true || got.KW["thinking_budget"] != float64(64) {
			t.Errorf("kwargs = %v, want only enable_thinking + thinking_budget", got.KW)
		}
	}
}

// From the broker's TestGuardEngineControls_TopLevelFields (forwarder part;
// the self-hosted allowlist stays in the broker).
func TestSanitize_KeepsUpstreamExtensions(t *testing.T) {
	in := `{"model":"m","messages":[],"temperature":0.2,"tools":[],"ebnf":"root ::= x","enable_search":true,"chat_template":"{{x}}","system":"s","input":[]}`
	got := decode(t, mustSanitize(t, in, SanitizeOptions{}))
	if _, ok := got["chat_template"]; ok {
		t.Error("chat_template kept")
	}
	for _, f := range []string{"model", "messages", "temperature", "tools", "ebnf", "enable_search", "system", "input"} {
		if _, ok := got[f]; !ok {
			t.Errorf("%s dropped", f)
		}
	}
}

// From the broker's TestGuardEngineControls_UnparseableBodyRefused: Sanitize
// reports them; the broker refuses, the router passes them through.
func TestSanitize_NotJSONObject(t *testing.T) {
	payload := `"chat_template_kwargs":{"chat_template":"{{ raise_exception(messages) }}"},"messages":[]}`
	for name, b := range map[string]string{
		"NaN":            `{"x":NaN,` + payload,
		"Infinity":       `{"x":Infinity,` + payload,
		"BOM":            "\xEF\xBB\xBF{" + payload,
		"array":          `[1,2]`,
		"null":           `null`,
		"text":           `not json`,
		"NaN, no strip":  `{"x":NaN}`,
		"trailing value": `{"model":"m"} {"chat_template":"x"}`,
		"multipart":      "--boundary\r\nContent-Disposition: form-data; name=\"chat_template\"\r\n\r\nx\r\n--boundary--",
	} {
		for _, opts := range []SanitizeOptions{{}, routerOptions} {
			out, err := Sanitize([]byte(b), opts)
			if !errors.Is(err, ErrNotJSONObject) {
				t.Errorf("%s: err = %v, want ErrNotJSONObject", name, err)
			}
			if string(out) != b {
				t.Errorf("%s: body changed alongside the error", name)
			}
		}
	}
}

// The options cover exactly where today's two guards differ.
func TestSanitize_Differences(t *testing.T) {
	for _, c := range []struct{ name, in, broker, router string }{
		{"peer fields", `{"model":"m","bootstrap_host":"h","kv_transfer_params":{}}`,
			`{"model":"m","bootstrap_host":"h","kv_transfer_params":{}}`, `{"model":"m"}`},
		{"kwargs not an object", `{"model":"m","chat_template_kwargs":"x"}`,
			`{"model":"m","chat_template_kwargs":"x"}`, `{"model":"m"}`},
		{"kwargs emptied", `{"model":"m","chat_template_kwargs":{"chat_template":"x"}}`,
			`{"chat_template_kwargs":{},"model":"m"}`, `{"model":"m"}`},
		{"HTML in a changed body", `{"lora_path":"x","messages":"<a&b>"}`,
			`{"messages":"\u003ca\u0026b\u003e"}`, `{"messages":"<a&b>"}`},
	} {
		if got := mustSanitize(t, c.in, SanitizeOptions{}); string(got) != c.broker {
			t.Errorf("%s (broker): got %s, want %s", c.name, got, c.broker)
		}
		if got := mustSanitize(t, c.in, routerOptions); string(got) != c.router {
			t.Errorf("%s (router): got %s, want %s", c.name, got, c.router)
		}
	}
}

func TestSanitize_DoesNotMutateInput(t *testing.T) {
	in := []byte(`{"lora_path":"x","chat_template_kwargs":{"chat_template":"y","enable_thinking":true}}`)
	orig := bytes.Clone(in)
	for _, opts := range []SanitizeOptions{{}, routerOptions} {
		if _, err := Sanitize(in, opts); err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(in, orig) {
			t.Fatal("input mutated")
		}
	}
}
