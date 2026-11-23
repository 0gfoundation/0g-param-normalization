package paramnorm

import (
	"bytes"
	"encoding/json"
	"fmt"
	"slices"
	"testing"
)

// Temporary: the two guards Sanitize replaces, copied verbatim so the port can
// be checked byte for byte. Remove once both callers are on Sanitize.

// routerStripEngineControls is 0g-router backend/pkg/inference/
// chat_template_kwargs.go stripEngineControls at a8cf90b1, minus the sealed
// check (which stays in the router).
func routerStripEngineControls(reqBody []byte) []byte {
	routerFields := []string{"chat_template", "custom_logit_processor", "lora_path", "lora_adapter_name", "return_hidden_states",
		"bootstrap_host", "bootstrap_port", "bootstrap_room", "kv_transfer_params"}
	if !bytes.Contains(reqBody, []byte(`\u`)) &&
		!bytes.Contains(reqBody, []byte("chat_template")) && !bytes.Contains(reqBody, []byte("logit_processor")) &&
		!bytes.Contains(reqBody, []byte("lora_")) && !bytes.Contains(reqBody, []byte("hidden_states")) &&
		!bytes.Contains(reqBody, []byte("bootstrap_")) && !bytes.Contains(reqBody, []byte("kv_transfer")) {
		return reqBody
	}
	var top map[string]json.RawMessage
	if json.Unmarshal(reqBody, &top) != nil {
		return reqBody
	}
	changed := false
	for _, f := range routerFields {
		if _, ok := top[f]; ok {
			delete(top, f)
			changed = true
		}
	}
	if raw, ok := top["chat_template_kwargs"]; ok {
		var kw map[string]json.RawMessage
		if json.Unmarshal(raw, &kw) != nil {
			delete(top, "chat_template_kwargs")
			changed = true
		} else {
			kept := map[string]json.RawMessage{}
			for k, v := range kw {
				if allowedChatTemplateKwargs[k] {
					kept[k] = v
				}
			}
			if len(kept) != len(kw) {
				changed = true
				if len(kept) == 0 {
					delete(top, "chat_template_kwargs")
				} else {
					top["chat_template_kwargs"], _ = json.Marshal(kept)
				}
			}
		}
	}
	if !changed {
		return reqBody
	}
	var out bytes.Buffer
	enc := json.NewEncoder(&out)
	enc.SetEscapeHTML(false)
	if enc.Encode(top) != nil {
		return reqBody
	}
	return bytes.TrimSuffix(out.Bytes(), []byte("\n"))
}

// brokerGuardEngineControls is 0g-serving-broker api/inference/internal/ctrl/
// engine_controls.go guardEngineControlsMedia at 04cb230e with selfHosted=false
// and without the media walk (both stay in the broker).
func brokerGuardEngineControls(body []byte) ([]byte, error) {
	if len(bytes.TrimSpace(body)) == 0 {
		return body, nil
	}
	var top map[string]json.RawMessage
	if err := json.Unmarshal(body, &top); err != nil || top == nil {
		return nil, fmt.Errorf("request body must be a JSON object")
	}
	changed := false
	for f := range top {
		if slices.Contains(engineControlFields, f) {
			delete(top, f)
			changed = true
		}
	}
	if raw, ok := top["chat_template_kwargs"]; ok {
		var kw map[string]json.RawMessage
		if json.Unmarshal(raw, &kw) == nil {
			stripped := false
			for k := range kw {
				if !allowedChatTemplateKwargs[k] {
					delete(kw, k)
					stripped = true
				}
			}
			if stripped {
				b, err := json.Marshal(kw)
				if err != nil {
					return nil, err
				}
				top["chat_template_kwargs"] = b
				changed = true
			}
		}
	}
	if !changed {
		return body, nil
	}
	return json.Marshal(top)
}

var parityBodies = []string{
	``, `   `, `{}`, `null`, `[1]`, `"s"`, `not json`, `{"x":NaN}`, "\xEF\xBB\xBF{}",
	`{"model":"m"} {"chat_template":"x"}`,
	"--b\r\nContent-Disposition: form-data; name=\"lora_path\"\r\n\r\nx\r\n--b--",
	`{"model":"m","messages":[{"role":"user","content":"<b>a & b</b>"}]}`,
	`{"model":"m","messages":[{"role":"user","content":"<b>a & b</b>"}],"lora_path":"x"}`,
	`{"model":"m","chat_template":"{{x}}"}`,
	`{"model":"m","custom_logit_processor":"gASV","return_hidden_states":true}`,
	`{"model":"m","lora_adapter_name":"a"}`,
	`{"model":"m","bootstrap_host":"10.0.0.1","bootstrap_port":8998,"bootstrap_room":1}`,
	`{"model":"m","kv_transfer_params":{"remote_host":"10.0.0.2"}}`,
	`{"model":"m","chat_template_kwargs":{"enable_thinking":true}}`,
	`{"model":"m","chat_template_kwargs":{"enable_thinking":true,"chat_template":"x","raise_exception":1}}`,
	`{"model":"m","chat_template_kwargs":{"chat_template":"x"}}`,
	`{"model":"m","chat_template_kwargs":{"thinking":"<a&b>","tools":[]}}`,
	`{"model":"m","chat_template_kwargs":"x"}`,
	`{"model":"m","chat_template_kwargs":null}`,
	`{"model":"m","chat_template_kwargs":{}}`,
	`{"model":"m","chat_template_kwargs":[1]}`,
	`{"model":"m","chat\u005ftemplate_kwargs":{"chat\u005ftemplate":"x"}}`,
	`{"model":"m","lora\u005fpath":"x"}`,
	`{"model":"m","note":"\u003c"}`,
	`{"model":"m","messages":[{"role":"user","content":"mentions chat_template and lora_path in text"}]}`,
	` {"model":"m","lora_path":"x"} `,
	`{"model":"m","lora_path":"a","lora_path":"b"}`,
	`{"model":"m","x":1e400,"lora_path":"x"}`,
	`{"model":"m","ebnf":"root ::= x","enable_search":true,"chat_template":"{{x}}"}`,
}

func TestSanitize_MatchesRouter(t *testing.T) {
	for _, b := range parityBodies {
		want := routerStripEngineControls([]byte(b))
		got, err := Sanitize([]byte(b), routerOptions)
		if err != nil {
			got = []byte(b) // the router passes a body Sanitize reports through
		}
		if !bytes.Equal(got, want) {
			t.Errorf("%q: Sanitize %s, router %s", b, got, want)
		}
	}
}

func TestSanitize_MatchesBroker(t *testing.T) {
	for _, b := range parityBodies {
		want, wantErr := brokerGuardEngineControls([]byte(b))
		got, err := Sanitize([]byte(b), SanitizeOptions{})
		if (err != nil) != (wantErr != nil) {
			t.Errorf("%q: Sanitize err %v, broker err %v", b, err, wantErr)
			continue
		}
		if err == nil && !bytes.Equal(got, want) {
			t.Errorf("%q: Sanitize %s, broker %s", b, got, want)
		}
	}
}
