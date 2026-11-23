package paramnorm

import (
	"bytes"
	"encoding/json"
	"errors"
)

// Engine-control sanitizing.
//
// On 2026-11-19 a stolen key drove a self-hosted sglang engine with
// chat_template_kwargs.chat_template: engines splat chat_template_kwargs into
// tokenizer.apply_chat_template, so the caller's Jinja replaced the model's
// template and read data back through raise_exception in the error text. The
// router (0g-router#1589) and the serving broker (0g-serving-broker#798) each
// wrote a guard that day. Sanitize is the part the two share; what differs
// between them is kept as SanitizeOptions so each caller behaves as before.
//
// Not here, because they depend on where the caller sits:
//   - sealed (E2EE) bodies: the router leaves them as sent (their cleartext
//     fields are bound into the seal); the broker sanitizes after decrypting;
//   - a body that is not a JSON object: Sanitize returns ErrNotJSONObject. The
//     router passes it through (the same path carries multipart uploads); the
//     broker refuses it (Python engines parse NaN, a BOM or UTF-16 that Go
//     cannot, so an unparsed body would reach them unchecked);
//   - the broker's self-hosted field allowlist, vllm_xargs check and media
//     link checks, which need the broker's own service config.

// engineControlFields are top-level fields an engine reads as server controls,
// not request parameters: a per-request chat template (vLLM with
// --trust-request-chat-template), a pickled logit processor (sglang dill-loads
// it: code execution), a LoRA adapter path or name, the model's hidden states.
var engineControlFields = []string{
	"chat_template", "custom_logit_processor", "lora_path", "lora_adapter_name", "return_hidden_states",
}

// allowedChatTemplateKwargs are the chat_template_kwargs keys a caller may
// send: the thinking switches open-weight templates read. An allowlist, not a
// denylist: engines merge these kwargs into their own with dict.update (vLLM
// CVE-2025-61620 / CVE-2025-62426), so any key the engine passes itself, or any
// template global, could otherwise be overridden.
var allowedChatTemplateKwargs = map[string]bool{
	"enable_thinking":   true, // Qwen3, GLM
	"thinking":          true, // DeepSeek V3.1+
	"clear_thinking":    true, // GLM-4.7+
	"preserve_thinking": true,
	"thinking_budget":   true,
	"reasoning_effort":  true, // gpt-oss
}

// ErrNotJSONObject is returned for a non-empty body that is not a single JSON
// object. The body is returned unchanged alongside it.
var ErrNotJSONObject = errors.New("request body must be a JSON object")

// SanitizeOptions carries the behaviour that differs between today's router and
// broker guards. The zero value is the broker's behaviour.
type SanitizeOptions struct {
	// ExtraFields are removed in addition to the shared engine-control fields.
	// The router passes the prefill/decode peer addresses (bootstrap_host,
	// bootstrap_port, bootstrap_room, kv_transfer_params).
	ExtraFields []string
	// DropEmptyKwargs removes chat_template_kwargs as a whole when it is not an
	// object or when no allowed key is left in it (router). Otherwise a
	// non-object value is left as sent and an emptied object stays as {}
	// (broker).
	DropEmptyKwargs bool
	// NoEscapeHTML re-encodes a changed body without escaping <, > and & in
	// string values (router). Otherwise json.Marshal escapes them (broker).
	// The JSON is the same either way; only the bytes upstream differ.
	NoEscapeHTML bool
}

// Sanitize removes engine-control fields and the chat_template_kwargs keys
// outside the thinking switches. A body with nothing to remove is returned as
// sent (same bytes); a changed body is re-encoded with its top-level keys in
// sorted order. An empty body is returned as is. A body that is not a JSON
// object yields ErrNotJSONObject (see the package notes above).
func Sanitize(body []byte, opts SanitizeOptions) ([]byte, error) {
	trimmed := bytes.TrimSpace(body)
	if len(trimmed) == 0 {
		return body, nil
	}
	if trimmed[0] != '{' {
		return body, ErrNotJSONObject
	}
	// Cheap exit when no field name to remove appears in the raw bytes, unless
	// a \u escape could spell one ("chat\u005ftemplate_kwargs" decodes to
	// chat_template_kwargs everywhere). "chat_template" also covers
	// chat_template_kwargs.
	if !bytes.Contains(body, []byte(`\u`)) && !mentionsAny(body, engineControlFields) && !mentionsAny(body, opts.ExtraFields) {
		if !json.Valid(body) {
			return body, ErrNotJSONObject
		}
		return body, nil
	}
	var top map[string]json.RawMessage
	if err := json.Unmarshal(body, &top); err != nil || top == nil {
		return body, ErrNotJSONObject
	}
	changed := false
	for _, fields := range [][]string{engineControlFields, opts.ExtraFields} {
		for _, f := range fields {
			if _, ok := top[f]; ok {
				delete(top, f)
				changed = true
			}
		}
	}
	if raw, ok := top["chat_template_kwargs"]; ok {
		var kw map[string]json.RawMessage
		if json.Unmarshal(raw, &kw) != nil {
			// Not an object: no allowed key can be in it.
			if opts.DropEmptyKwargs {
				delete(top, "chat_template_kwargs")
				changed = true
			}
		} else {
			kept := make(map[string]json.RawMessage, len(kw))
			for k, v := range kw {
				if allowedChatTemplateKwargs[k] {
					kept[k] = v
				}
			}
			if len(kept) != len(kw) {
				changed = true
				if len(kept) == 0 && opts.DropEmptyKwargs {
					delete(top, "chat_template_kwargs")
				} else {
					b, err := json.Marshal(kept)
					if err != nil {
						return body, err
					}
					top["chat_template_kwargs"] = b
				}
			}
		}
	}
	if !changed {
		return body, nil
	}
	if !opts.NoEscapeHTML {
		out, err := json.Marshal(top)
		if err != nil {
			return body, err
		}
		return out, nil
	}
	var out bytes.Buffer
	enc := json.NewEncoder(&out)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(top); err != nil {
		return body, err
	}
	return bytes.TrimSuffix(out.Bytes(), []byte("\n")), nil
}

func mentionsAny(body []byte, fields []string) bool {
	for _, f := range fields {
		if bytes.Contains(body, []byte(f)) {
			return true
		}
	}
	return false
}
