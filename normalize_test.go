package paramnorm

import (
	"bytes"
	"errors"
	"reflect"
	"testing"
)

// brokerSequence runs the steps one after another in the order the broker's
// PrepareHTTPRequest calls them (cap, max_tokens, reasoning, request profile),
// each on the previous step's output: what Normalize with every category On
// must reproduce.
func brokerSequence(t *testing.T, body []byte, row *Row, format string) []byte {
	t.Helper()
	var err error
	if body, _, err = capMaxOutputTokens(body, row); err != nil {
		t.Fatal(err)
	}
	if body, _, err = translateMaxTokens(body, row); err != nil {
		t.Fatal(err)
	}
	if body, _, err = translateReasoning(body, row); err != nil {
		t.Fatal(err)
	}
	if row != nil {
		if body, _, err = applyRequestProfile(body, row.RequestProfile, format); err != nil {
			t.Fatal(err)
		}
	}
	return body
}

var allOn = Options{Translate: On, Drop: On, CapOutput: On}

// The rows and bodies the equivalence test crosses: each row shape the broker
// translates for, and bodies that touch every step and their interactions.
var (
	equivRows = map[string]*Row{
		"nil":            nil,
		"empty":          {},
		"qwen vllm":      {SupportedParameters: []string{"max_tokens", "reasoning_effort", "chat_template_kwargs"}, MaxCompletionTokens: 8192, ContextLength: 32768},
		"glm graded":     {SupportedParameters: []string{"max_tokens", "chat_template_kwargs"}, ReasoningEffortLevels: []string{"low", "high", "max"}, MaxCompletionTokens: 32768, ContextLength: 262144, RequestProfile: RequestProfileSGLang},
		"openai newer":   {SupportedParameters: []string{"max_completion_tokens", "reasoning_effort"}, MaxCompletionTokens: 4096},
		"minimax":        {SupportedParameters: []string{"max_tokens", "thinking"}, ThinkingTypes: []string{"adaptive", "disabled"}},
		"dashscope":      {SupportedParameters: []string{"max_completion_tokens", "enable_thinking"}},
		"openrouter":     {SupportedParameters: []string{"max_tokens", "max_completion_tokens", "reasoning"}},
		"anthropic only": {SupportedParameters: []string{"max_tokens", "thinking"}, SupportedFormats: []string{"anthropic"}, RequestProfile: RequestProfileSGLang},
	}
	equivBodies = []struct{ format, body string }{
		{APIFormatOpenAI, `{"model":"m","messages":[{"role":"user","content":"hi"}]}`},
		{APIFormatOpenAI, `{"model":"m","reasoning_effort":"high","max_completion_tokens":100000,"messages":[]}`},
		{APIFormatOpenAI, `{"model":"m","reasoning_effort":"none","max_tokens":10,"seed":9007199254740993}`},
		{APIFormatOpenAI, `{"model":"m","reasoning_effort":"medium","max_tokens":null,"max_completion_tokens":"lots"}`},
		{APIFormatOpenAI, `{"model":"m","reasoning_effort":"ultra","chat_template_kwargs":{"enable_thinking":true}}`},
		{APIFormatOpenAI, `{"model":"m","reasoning_effort":"low","reasoning":{"max_tokens":100}}`},
		{APIFormatAnthropic, `{"model":"m","max_tokens":64,"thinking":{"type":"adaptive","display":"updates"},"output_config":{"effort":"ultra"}}`},
		{APIFormatOpenAI, `not json`},
		{APIFormatOpenAI, ``},
	}
)

// With every category on, Normalize is the broker's sequence: same body for
// every row shape and every body.
func TestNormalize_AllOnMatchesBrokerSequence(t *testing.T) {
	for rowName, row := range equivRows {
		for _, b := range equivBodies {
			want := brokerSequence(t, []byte(b.body), row, b.format)
			res, err := Normalize(Request{Body: []byte(b.body), Format: b.format}, row, allOn)
			if err != nil {
				t.Fatalf("%s / %s: %v", rowName, b.body, err)
			}
			if !bytes.Equal(res.Body, want) {
				t.Errorf("%s / %s:\n got %s\nwant %s", rowName, b.body, res.Body, want)
			}
		}
	}
}

// The zero Options runs nothing: the body is returned as is and nothing is
// reported, whatever the row declares.
func TestNormalize_ZeroOptionsIsInert(t *testing.T) {
	for rowName, row := range equivRows {
		for _, b := range equivBodies {
			res, err := Normalize(Request{Body: []byte(b.body), Format: b.format}, row, Options{})
			if err != nil {
				t.Fatalf("%s / %s: %v", rowName, b.body, err)
			}
			if string(res.Body) != b.body || res.Applied != nil || res.Skipped != nil {
				t.Errorf("%s / %s: got %+v, want the body untouched and nothing reported", rowName, b.body, res)
			}
		}
	}
}

// DryRun reports exactly what On would apply, and leaves the body alone. Checked
// one category at a time: with several, a DryRun step sees the original body
// rather than the previous step's output.
func TestNormalize_DryRunReportsWhatOnApplies(t *testing.T) {
	for cat, set := range map[string]func(*Options, Mode){
		"translate":  func(o *Options, m Mode) { o.Translate = m },
		"drop":       func(o *Options, m Mode) { o.Drop = m },
		"cap output": func(o *Options, m Mode) { o.CapOutput = m },
	} {
		var on, dry Options
		set(&on, On)
		set(&dry, DryRun)
		for rowName, row := range equivRows {
			for _, b := range equivBodies {
				req := Request{Body: []byte(b.body), Format: b.format}
				applied, err := Normalize(req, row, on)
				if err != nil {
					t.Fatal(err)
				}
				skipped, err := Normalize(req, row, dry)
				if err != nil {
					t.Fatal(err)
				}
				if string(skipped.Body) != b.body || skipped.Applied != nil {
					t.Errorf("%s / %s / %s: DryRun changed the body to %s", cat, rowName, b.body, skipped.Body)
				}
				if !reflect.DeepEqual(skipped.Skipped, applied.Applied) {
					t.Errorf("%s / %s / %s: DryRun reported %+v, On applied %+v", cat, rowName, b.body, skipped.Skipped, applied.Applied)
				}
			}
		}
	}
}

// Order matters: the cap lowers what the client sent, then the rename moves it
// to the spelling the row accepts. Reversed, the rename would move the
// uncapped value and the cap would inject a second field.
func TestNormalize_CapRunsBeforeRename(t *testing.T) {
	row := &Row{SupportedParameters: []string{"max_completion_tokens"}, MaxCompletionTokens: 4096}
	res, err := Normalize(Request{Body: []byte(`{"max_tokens":100000}`), Format: APIFormatOpenAI}, row, allOn)
	if err != nil {
		t.Fatal(err)
	}
	if string(res.Body) != `{"max_completion_tokens":4096}` {
		t.Fatalf("body = %s, want the capped value under the accepted spelling", res.Body)
	}
	want := []Change{
		{Step: StepCapOutput, Param: "max_tokens", Action: ActionCapped, To: "4096"},
		{Step: StepMaxTokens, Param: "max_tokens", Action: ActionTranslated, To: "max_completion_tokens"},
	}
	if !reflect.DeepEqual(res.Applied, want) {
		t.Errorf("Applied = %+v, want %+v", res.Applied, want)
	}
}

// Categories are independent: Translate on with the cap off translates without
// clamping, which is how the broker runs a service without
// enforceMaxCompletionTokens.
func TestNormalize_CategoriesAreIndependent(t *testing.T) {
	row := equivRows["qwen vllm"]
	body := `{"reasoning_effort":"high","max_tokens":100000}`
	res, err := Normalize(Request{Body: []byte(body), Format: APIFormatOpenAI}, row, Options{Translate: On})
	if err != nil {
		t.Fatal(err)
	}
	if string(res.Body) != `{"chat_template_kwargs":{"enable_thinking":true},"max_tokens":100000}` {
		t.Errorf("body = %s, want reasoning translated and max_tokens left as sent", res.Body)
	}
	want := []Change{{Step: StepReasoning, Param: "reasoning_effort", Action: ActionTranslated, To: "chat_template_kwargs"}}
	if !reflect.DeepEqual(res.Applied, want) {
		t.Errorf("Applied = %+v, want %+v", res.Applied, want)
	}
}

// The input body is never modified in place, so a caller can normalize the
// same request again for another row (the gateway re-seals per candidate).
func TestNormalize_DoesNotMutateInput(t *testing.T) {
	for rowName, row := range equivRows {
		for _, b := range equivBodies {
			in := []byte(b.body)
			keep := append([]byte(nil), in...)
			if _, err := Normalize(Request{Body: in, Format: b.format}, row, allOn); err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(in, keep) {
				t.Errorf("%s / %s: input changed to %s", rowName, b.body, in)
			}
		}
	}
}

// A cap that does not fit the context window is reported but changes nothing.
func TestNormalize_NotInjectedIsReportedOnly(t *testing.T) {
	row := &Row{SupportedParameters: []string{"max_tokens"}, MaxCompletionTokens: 8192, ContextLength: 8192}
	body := []byte(`{"messages":[{"role":"user","content":"hello"}]}`)
	res, err := Normalize(Request{Body: body, Format: APIFormatOpenAI}, row, Options{CapOutput: On})
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(res.Body, body) {
		t.Errorf("body changed to %s", res.Body)
	}
	want := []Change{{Step: StepCapOutput, Param: "max_tokens", Action: ActionNotInjected, From: "16", To: "8192"}}
	if !reflect.DeepEqual(res.Applied, want) {
		t.Errorf("Applied = %+v, want %+v", res.Applied, want)
	}
}

// An encoding failure in an On category fails the request with a *StepError
// naming the step, keeping the step's own text; in a DryRun category it is
// ignored (DryRun changes nothing) and the step's changes are still reported.
// No real body reaches json.Marshal's error path, so the steps are faked.
func TestRunSteps_Errors(t *testing.T) {
	body := []byte(`{"a":1}`)
	boom := errors.New("failed to marshal x body: boom")
	failing := func(b []byte) ([]byte, []Change, error) {
		return b, []Change{{Param: "p", Action: ActionDropped, From: "v"}}, boom
	}
	rewrite := func(b []byte) ([]byte, []Change, error) {
		return []byte(`{"a":2}`), []Change{{Param: "a", Action: ActionCapped, To: "2"}}, nil
	}

	res, err := runSteps(body, []step{{StepCapOutput, On, rewrite}, {StepRequestProfile, On, failing}})
	var se *StepError
	if !errors.As(err, &se) || se.Step != StepRequestProfile || err.Error() != boom.Error() || !errors.Is(err, boom) {
		t.Fatalf("On: err = %#v, want *StepError{request_profile} with the step's text", err)
	}
	if !bytes.Equal(res.Body, body) || res.Applied != nil || res.Skipped != nil {
		t.Errorf("On: result = %+v, want the input body and nothing reported", res)
	}

	res, err = runSteps(body, []step{{StepCapOutput, On, rewrite}, {StepRequestProfile, DryRun, failing}})
	if err != nil {
		t.Fatalf("DryRun: err = %v, want nil", err)
	}
	if string(res.Body) != `{"a":2}` {
		t.Errorf("DryRun: body = %s, want the On step's rewrite kept", res.Body)
	}
	want := []Change{{Step: StepRequestProfile, Param: "p", Action: ActionDropped, From: "v"}}
	if !reflect.DeepEqual(res.Skipped, want) {
		t.Errorf("DryRun: Skipped = %+v, want %+v", res.Skipped, want)
	}
}
