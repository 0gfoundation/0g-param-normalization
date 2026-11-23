package paramnorm

import (
	"bytes"
	"encoding/json"
	"testing"
)

func decodeProfileBody(t *testing.T, b []byte) map[string]interface{} {
	t.Helper()
	var m map[string]interface{}
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("invalid json %s: %v", b, err)
	}
	return m
}

// Claude Code's display values SGLang does not list are dropped; thinking itself,
// and so thinking on, stays.
func TestApplyRequestProfile_DropsUnknownThinkingDisplay(t *testing.T) {
	for _, display := range []string{"updates", "highlights"} {
		body := []byte(`{"model":"glm-5.3","max_tokens":64,"messages":[],"thinking":{"type":"adaptive","display":"` + display + `"}}`)
		got, _, err := applyRequestProfile(body, RequestProfileSGLang, APIFormatAnthropic)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		thinking, ok := decodeProfileBody(t, got)["thinking"].(map[string]interface{})
		if !ok {
			t.Fatalf("thinking dropped: %s", got)
		}
		if _, present := thinking["display"]; present {
			t.Errorf("display %q not dropped: %s", display, got)
		}
		if thinking["type"] != "adaptive" {
			t.Errorf("thinking.type = %v, want adaptive", thinking["type"])
		}
	}
}

// Accepted values, non-string values, absent fields and other surfaces leave the
// body byte-for-byte unchanged.
func TestApplyRequestProfile_KeepsWhatTheEngineAccepts(t *testing.T) {
	cases := []struct {
		name, surface, body string
	}{
		{"accepted display", APIFormatAnthropic, `{"thinking":{"type":"enabled","budget_tokens":2048,"display":"summarized"}}`},
		{"null display", APIFormatAnthropic, `{"thinking":{"type":"adaptive","display":null}}`},
		{"no thinking", APIFormatAnthropic, `{"messages":[]}`},
		{"thinking not an object", APIFormatAnthropic, `{"thinking":"on"}`},
		{"accepted effort", APIFormatAnthropic, `{"output_config":{"effort":"xhigh"}}`},
		{"numeric reasoning_effort", APIFormatOpenAI, `{"reasoning_effort":0.5}`},
		{"accepted reasoning_effort", APIFormatOpenAI, `{"reasoning_effort":"max"}`},
		{"numeric-string reasoning_effort", APIFormatOpenAI, `{"reasoning_effort":"0.5"}`},
		{"out-of-range numeric string is SGLang's to judge", APIFormatOpenAI, `{"reasoning_effort":"1.5"}`},
		{"underscored numeric string pydantic accepts", APIFormatOpenAI, `{"reasoning_effort":"0._5"}`},
		{"display on the openai surface", APIFormatOpenAI, `{"thinking":{"display":"updates"}}`},
		{"unrecognized surface", "", `{"thinking":{"display":"updates"}}`},
		{"not json", APIFormatAnthropic, `not json`},
	}
	for _, tc := range cases {
		got, _, err := applyRequestProfile([]byte(tc.body), RequestProfileSGLang, tc.surface)
		if err != nil {
			t.Fatalf("%s: unexpected error: %v", tc.name, err)
		}
		if string(got) != tc.body {
			t.Errorf("%s: body changed to %s", tc.name, got)
		}
	}
}

func TestApplyRequestProfile_DropsUnknownEffortTiers(t *testing.T) {

	got, _, err := applyRequestProfile([]byte(`{"output_config":{"effort":"ultra","task_budget":{"type":"tokens","total":1000}}}`), RequestProfileSGLang, APIFormatAnthropic)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	oc, _ := decodeProfileBody(t, got)["output_config"].(map[string]interface{})
	if _, present := oc["effort"]; present || oc["task_budget"] == nil {
		t.Errorf("want only output_config.effort dropped, got %s", got)
	}

	got, _, err = applyRequestProfile([]byte(`{"reasoning_effort":"ultra","temperature":0.7}`), RequestProfileSGLang, APIFormatOpenAI)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	m := decodeProfileBody(t, got)
	if _, present := m["reasoning_effort"]; present || m["temperature"] == nil {
		t.Errorf("want only reasoning_effort dropped, got %s", got)
	}
}

// Log-only mode (Options.Drop = DryRun) forwards the body untouched even when a
// rule matches, and reports what it would have dropped.
func TestApplyRequestProfile_LogOnlyLeavesBody(t *testing.T) {
	body := []byte(`{"thinking":{"type":"adaptive","display":"updates"}}`)
	res, err := Normalize(Request{Body: body, Format: APIFormatAnthropic},
		&Row{RequestProfile: RequestProfileSGLang}, Options{Drop: DryRun})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !bytes.Equal(res.Body, body) {
		t.Errorf("log-only mode changed the body to %s", res.Body)
	}
	if len(res.Skipped) != 1 || res.Skipped[0] != (Change{Param: "thinking.display", Action: ActionDropped, From: "updates"}) {
		t.Errorf("log-only mode reported %+v, want the would-be drop of thinking.display", res.Skipped)
	}
}
