package paramnorm

import (
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"
)

func values(t *testing.T, body string, ep Endpoint) Classification {
	t.Helper()
	c, err := Classify([]byte(body), ep)
	if err != nil {
		t.Fatalf("Classify: %v", err)
	}
	return c
}

func has(c Classification, param, class string) bool {
	for _, v := range c.Values {
		if v.Param == param && v.Class == class {
			return true
		}
	}
	return false
}

func TestClassifyChat(t *testing.T) {
	c := values(t, `{"model":"m","messages":[],"stream":true,"user":"u","metadata":{"a":1},
		"temperature":0.2,"logprobs":false,"top_logprobs":0,"n":1,"stop":[],"seed":null,
		"reasoning_effort":"high","enable_thinking":false,
		"tools":[{"type":"function","function":{}},{"type":"custom"}],
		"tool_choice":{"type":"function","function":{"name":"f"}},
		"response_format":{"type":"json_schema","json_schema":{"name":"x","strict":true}},
		"chat_template_kwargs":{"enable_thinking":true,"bogus":1},
		"max_completion_tokens":4096,"frobnicate":1,"thinking":"x"}`, EndpointChat)
	want := []Value{
		{Param: "chat_template_kwargs.enable_thinking", Class: "true"},
		{Param: "enable_thinking", Class: "false"},
		{Param: "max_completion_tokens", Class: ClassPresent, Number: 4096, HasNumber: true},
		{Param: "reasoning_effort", Class: "high"},
		{Param: "response_format", Class: "json_schema_strict"},
		{Param: "temperature", Class: ClassPresent},
		{Param: "thinking", Class: ClassUnknown}, // a container that is not an object
		{Param: "tool_choice", Class: "function"},
		{Param: "tools", Class: "custom"},
		{Param: "tools", Class: "function"},
	}
	if !reflect.DeepEqual(c.Values, want) {
		t.Errorf("values:\n got %+v\nwant %+v", c.Values, want)
	}
	// unknown fields and a member the catalog does not know
	if want := []string{"chat_template_kwargs.bogus", "frobnicate"}; !reflect.DeepEqual(c.Unknown, want) {
		t.Errorf("unknown: %v, want %v", c.Unknown, want)
	}
}

func TestClassifyMessages(t *testing.T) {
	c := values(t, `{"model":"m","messages":[],"max_tokens":1024,"system":"s","stop_sequences":["x"],
		"thinking":{"type":"enabled","budget_tokens":2048,"display":"omitted"},
		"tools":[{"name":"f","input_schema":{}},{"type":"custom","name":"g"},{"type":"web_search_20250305","name":"web_search"}],
		"tool_choice":{"type":"any","disable_parallel_tool_use":true},
		"output_config":{"effort":"max","format":{"type":"json_schema","schema":{}}},"cache_control":{"type":"ephemeral"}}`, EndpointMessages)
	for _, pc := range [][2]string{
		{"thinking.type", "enabled"}, {"thinking.budget_tokens", ClassPresent}, {"thinking.display", "omitted"},
		{"tools", "function"}, {"tools", "hosted"}, {"tool_choice", "required"}, {"parallel_tool_calls", ClassPresent},
		{"output_config.effort", "max"}, {"output_config.format", "json_schema_strict"}, {"stop_sequences", ClassPresent},
		{"max_tokens", ClassPresent},
	} {
		if !has(c, pc[0], pc[1]) {
			t.Errorf("missing %v in %+v", pc, c.Values)
		}
	}
	if has(c, "tools", "custom") || len(c.Unknown) != 0 {
		t.Errorf("an Anthropic custom tool is a function tool; unknown %v", c.Unknown)
	}
}

func TestClassifyResponses(t *testing.T) {
	c := values(t, `{"model":"m","input":"hi","reasoning":{"effort":"low","summary":"auto"},
		"text":{"format":{"type":"json_schema","strict":false,"schema":{}},"verbosity":"low"},
		"tools":[{"type":"local_shell"},{"type":"file_search"},{"type":"brand_new_tool"}],
		"tool_choice":{"type":"allowed_tools"},"include":["reasoning.encrypted_content"],"max_output_tokens":100}`, EndpointResponses)
	for _, pc := range [][2]string{
		{"reasoning.effort", "low"}, {"reasoning.summary", ClassPresent}, {"text.format", "json_schema"},
		{"text.verbosity", ClassPresent}, {"tools", "local_shell"}, {"tools", "hosted"}, {"tools", ClassUnknown},
		{"tool_choice", ClassUnknown}, {"include", ClassPresent}, {"max_output_tokens", ClassPresent},
	} {
		if !has(c, pc[0], pc[1]) {
			t.Errorf("missing %v in %+v", pc, c.Values)
		}
	}
	if _, err := Classify([]byte(`[1]`), EndpointChat); !errors.Is(err, ErrNotObject) {
		t.Errorf("non-object body: %v", err)
	}
	// text format "text" asks for nothing
	if c := values(t, `{"response_format":{"type":"text"}}`, EndpointChat); len(c.Values) != 0 {
		t.Errorf("text format: %+v", c.Values)
	}
}

func rowRules(t *testing.T, rows string) RowRules {
	t.Helper()
	return mustParse(t, bundle(1, "", rows)).ForRow(RowKey{Identity: "a", Model: "m"}, EndpointChat)
}

func TestDecideTable(t *testing.T) {
	rr := rowRules(t, `{"identity":"a","model":"m","params":{
		"temperature":"ignored","top_k":"rejected","seed":"supported",
		"logprobs":"ignored","n":"rejected","stop":"supported",
		"reasoning_effort":{"none":"translate:enable_thinking","medium":"translate:level:high","high":"supported"}}}`)
	cases := []struct {
		v      Value
		strict bool
		want   Disposition
	}{
		{Value{Param: "seed", Class: ClassPresent}, false, Forward},
		{Value{Param: "temperature", Class: ClassPresent}, false, Drop}, // ignored, drop class
		{Value{Param: "temperature", Class: ClassPresent}, true, Avoid}, // ... under require_parameters
		{Value{Param: "top_k", Class: ClassPresent}, false, Drop},       // refused, drop class
		{Value{Param: "top_p", Class: ClassPresent}, false, Forward},    // unknown drop class: forwarded
		{Value{Param: "top_p", Class: ClassPresent}, true, Avoid},       // ... unless require_parameters
		{Value{Param: "logprobs", Class: ClassPresent}, false, Avoid},   // ignored, reject class
		{Value{Param: "n", Class: ClassPresent}, false, Avoid},          // refused, reject class
		{Value{Param: "stop", Class: ClassPresent}, false, Forward},
		{Value{Param: "tools", Class: "function"}, false, Avoid}, // unknown, reject class
		{Value{Param: "reasoning_effort", Class: "none"}, false, Rewrite},
		{Value{Param: "reasoning_effort", Class: "medium"}, false, Rewrite}, // level approximation
		{Value{Param: "reasoning_effort", Class: "medium"}, true, Avoid},    // ... not under require_parameters
		{Value{Param: "reasoning_effort", Class: "none"}, true, Rewrite},    // a respelling still applies
		{Value{Param: "reasoning_effort", Class: "low"}, false, Avoid},      // unknown translate class
		{Value{Param: "user", Class: ClassPresent}, true, Forward},          // passthrough
		{Value{Param: "tools", Class: ClassUnknown}, false, Avoid},          // a class the catalog lacks
		{Value{Param: "frobnicate", Class: ClassPresent}, false, Avoid},     // a parameter it lacks
		{Value{Param: "max_tokens", Class: ClassPresent}, true, Forward},    // an output cap: limits decide
		{Value{Param: "thinking.budget_tokens", Class: ClassPresent}, false, Avoid},
	}
	for _, c := range cases {
		d := Decide(rr, c.v, PlanOptions{RequireParameters: c.strict})
		if d.Do != c.want {
			t.Errorf("%s/%s strict=%v: %s (%s), want %s", c.v.Param, c.v.Class, c.strict, d.Do, d.Reason, c.want)
		}
		if d.Do == Avoid && d.Reason == "" {
			t.Errorf("%s/%s: avoid without a reason", c.v.Param, c.v.Class)
		}
	}
	if d := Decide(rr, Value{Param: "reasoning_effort", Class: "none"}, PlanOptions{}); d.Translation != "enable_thinking" {
		t.Errorf("translation: %+v", d)
	}
}

func planBundle(t *testing.T) *Ruleset {
	t.Helper()
	return mustParse(t, bundle(1, "", `
		{"identity":"a","model":"m","params":{"tools":{"function":"supported"},"logprobs":"rejected","max_tokens":{"limits":{"max":4096}}}},
		{"identity":"b","model":"m","params":{"tools":{"function":"supported"},"logprobs":"supported","max_tokens":{"limits":{"max":8192}}}},
		{"identity":"c","model":"m","params":{"tools":{"function":"rejected"},"logprobs":"supported","max_tokens":{"limits":{"min":16,"max":65536}}}}`))
}

var abc = []RowKey{{Identity: "a", Model: "m"}, {Identity: "b", Model: "m"}, {Identity: "c", Model: "m"}}

func servable(p PlanResult) []string {
	var out []string
	for _, r := range p.Servable() {
		out = append(out, r.Row.Identity)
	}
	return out
}

func TestPlanRoutesAround(t *testing.T) {
	rs := planBundle(t)
	p, err := Plan(rs, []byte(`{"tools":[{"type":"function"}],"logprobs":true}`), EndpointChat, abc, PlanOptions{})
	if err != nil || !reflect.DeepEqual(servable(p), []string{"b"}) {
		t.Fatalf("servable %v err %v", servable(p), err)
	}
	// logprobs false asks for nothing
	if p, _ := Plan(rs, []byte(`{"tools":[{"type":"function"}],"logprobs":false}`), EndpointChat, abc, PlanOptions{}); !reflect.DeepEqual(servable(p), []string{"a", "b"}) {
		t.Errorf("logprobs false: %v", servable(p))
	}
	// nobody serves it: unsupported_parameter naming the value every row avoids
	_, err = Plan(rs, []byte(`{"n":3}`), EndpointChat, abc, PlanOptions{})
	var re *RejectError
	if !errors.As(err, &re) || re.Param != "n" {
		t.Fatalf("reject: %v", err)
	}
	// a row's entry for one endpoint does not answer another
	ep := mustParse(t, bundle(1, "", `{"identity":"a","model":"m","endpoint":"chat","params":{"tools":{"function":"supported"}}}`))
	if _, err := Plan(ep, []byte(`{"tools":[{"type":"function"}]}`), EndpointResponses, abc[:1], PlanOptions{}); !errors.As(err, &re) {
		t.Errorf("responses: %v", err)
	}
	if _, err := Plan(ep, []byte(`{"tools":[{"type":"function"}]}`), EndpointChat, abc[:1], PlanOptions{}); err != nil {
		t.Errorf("chat: %v", err)
	}
}

func TestPlanOutputCapD10(t *testing.T) {
	rs := planBundle(t)
	// a fits nothing above 4096: avoided while b or c fits
	p, err := Plan(rs, []byte(`{"max_tokens":6000}`), EndpointChat, abc, PlanOptions{})
	if err != nil || !reflect.DeepEqual(servable(p), []string{"b", "c"}) {
		t.Fatalf("6000: %v %v", servable(p), err)
	}
	// nobody fits: every row truncates to its own cap
	p, err = Plan(rs, []byte(`{"max_completion_tokens":100000}`), EndpointChat, abc, PlanOptions{})
	if err != nil || len(p.Servable()) != 3 || p.Rows[0].TruncateTo != 4096 || p.Rows[2].TruncateTo != 65536 {
		t.Fatalf("100000: %+v %v", p.Rows, err)
	}
	// ... refused under require_parameters
	if _, err := Plan(rs, []byte(`{"max_output_tokens":100000}`), EndpointChat, abc, PlanOptions{RequireParameters: true}); err == nil {
		t.Error("truncation under require_parameters")
	}
	// below a row's min avoids that row
	p, _ = Plan(rs, []byte(`{"max_tokens":8}`), EndpointChat, abc, PlanOptions{})
	if !reflect.DeepEqual(servable(p), []string{"a", "b"}) {
		t.Errorf("below min: %v", servable(p))
	}
}

func TestPlanUnknownFields(t *testing.T) {
	rs := planBundle(t)
	p, err := Plan(rs, []byte(`{"frobnicate":1}`), EndpointChat, abc, PlanOptions{})
	if err != nil || len(p.Servable()) != 3 || !reflect.DeepEqual(p.Classification.Unknown, []string{"frobnicate"}) {
		t.Fatalf("recorded only: %v %v", p.Classification.Unknown, err)
	}
	if _, err := Plan(rs, []byte(`{"frobnicate":1}`), EndpointChat, abc, PlanOptions{RefuseUnknownFields: true}); err == nil {
		t.Error("unknown field refused when asked")
	}
	if _, err := Plan(rs, []byte(`not json`), EndpointChat, abc, PlanOptions{}); !errors.Is(err, ErrNotObject) {
		t.Errorf("bad body: %v", err)
	}
}

func TestPlanRound1(t *testing.T) {
	// clear_thinking false and parallel_tool_calls false ask for something; their defaults do not
	c := values(t, `{"chat_template_kwargs":{"clear_thinking":false},"parallel_tool_calls":true}`, EndpointChat)
	if !has(c, "chat_template_kwargs.clear_thinking", ClassPresent) || has(c, "parallel_tool_calls", ClassPresent) {
		t.Errorf("defaults: %+v", c.Values)
	}
	if c := values(t, `{"chat_template_kwargs":{"clear_thinking":true},"parallel_tool_calls":false}`, EndpointChat); has(c, "chat_template_kwargs.clear_thinking", ClassPresent) || !has(c, "parallel_tool_calls", ClassPresent) {
		t.Errorf("non-defaults: %+v", c.Values)
	}
	// client-run vendor tools are no catalog class; provider-run ones are hosted
	c = values(t, `{"tools":[{"type":"bash_20250124","name":"bash"},{"type":"code_execution_20250522","name":"x"}]}`, EndpointMessages)
	if !has(c, "tools", ClassUnknown) || !has(c, "tools", "hosted") {
		t.Errorf("tools: %+v", c.Values)
	}
	// a whole number written as a float
	if c := values(t, `{"max_tokens":1024.0}`, EndpointChat); len(c.Values) != 1 || !c.Values[0].HasNumber || c.Values[0].Number != 1024 {
		t.Errorf("1024.0: %+v", c.Values)
	}
}

func TestPlanTruncationAndBudget(t *testing.T) {
	rs := mustParse(t, bundle(1, "", `
		{"identity":"a","model":"m","params":{"thinking.type":{"enabled":"supported"},"thinking.budget_tokens":{"present":"supported","limits":{"min":1024,"max":20000}},"max_tokens":{"limits":{"max":8000}}}},
		{"identity":"b","model":"m","params":{"thinking.type":{"enabled":"supported"},"thinking.budget_tokens":{"present":"supported"},"max_tokens":{"limits":{"max":16000}}}}`))
	rows := []RowKey{{Identity: "a", Model: "m"}, {Identity: "b", Model: "m"}}
	// nobody fits 32000: truncated, the largest cap first
	p, err := Plan(rs, []byte(`{"max_tokens":32000,"thinking":{"type":"enabled","budget_tokens":4000}}`), EndpointMessages, rows, PlanOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if s := p.Servable(); len(s) != 2 || s[0].Row.Identity != "b" || s[0].TruncateTo != 16000 || s[1].TruncateTo != 8000 {
		t.Fatalf("order: %+v", s)
	}
	// a budget the truncated cap cannot hold: that row is avoided
	p, _ = Plan(rs, []byte(`{"max_tokens":32000,"thinking":{"type":"enabled","budget_tokens":12000}}`), EndpointMessages, rows, PlanOptions{})
	if s := p.Servable(); len(s) != 1 || s[0].Row.Identity != "b" {
		t.Fatalf("budget above a's cap: %+v", s)
	}
	// a budget outside the row's budget limits avoids it
	p, _ = Plan(rs, []byte(`{"max_tokens":4000,"thinking":{"type":"enabled","budget_tokens":512}}`), EndpointMessages, rows, PlanOptions{})
	if s := p.Servable(); len(s) != 1 || s[0].Row.Identity != "b" {
		t.Fatalf("budget below a's min: %+v", s)
	}
}

func TestClassifyAdversarial(t *testing.T) {
	cases := []struct {
		body      string
		ep        Endpoint
		param     string
		wantClass string
	}{
		// case-variant duplicate keys: read the exact key, refuse the ambiguity
		{`{"response_format":{"type":"json_schema","Type":"text"}}`, EndpointChat, "response_format", ClassUnknown},
		{`{"response_format":{"type":"json_schema","json_schema":{"strict":true,"STRICT":false}}}`, EndpointChat, "response_format", ClassUnknown},
		{`{"tool_choice":{"type":"any","TYPE":"auto"}}`, EndpointMessages, "tool_choice", ClassUnknown},
		{`{"text":{"format":{"type":"json_schema","strict":true}}}`, EndpointResponses, "text.format", "json_schema_strict"},
		{`{"response_format":{"type":"json_object"}}`, EndpointChat, "response_format", "json_object"},
		// case variants of catalog fields
		{`{"N":3}`, EndpointChat, "n", ClassUnknown},
		{`{"Response_Format":{"type":"json_object"}}`, EndpointChat, "response_format", ClassUnknown},
		{`{"reasoning":{"Effort":"high"}}`, EndpointResponses, "reasoning.effort", ClassUnknown},
		// web_search_options {} turns search on
		{`{"web_search_options":{}}`, EndpointChat, "web_search_options", ClassPresent},
		// caps that are no count, or written oddly
		{`{"max_tokens":-5}`, EndpointChat, "max_tokens", ClassUnknown},
		{`{"max_tokens":10.5}`, EndpointChat, "max_tokens", ClassUnknown},
		{`{"max_tokens":"100"}`, EndpointChat, "max_tokens", ClassUnknown},
		{`{"max_tokens":1e6}`, EndpointChat, "max_tokens", ClassPresent},
		{`{"max_tokens":99999999999999999999}`, EndpointChat, "max_tokens", ClassPresent},
		// a function_call object
		{`{"function_call":{"name":"f"}}`, EndpointChat, "function_call", "function"},
		// disable_parallel_tool_use false asks for nothing
		{`{"tool_choice":{"type":"auto","disable_parallel_tool_use":false}}`, EndpointMessages, "tool_choice", "auto"},
	}
	for _, c := range cases {
		got := values(t, c.body, c.ep)
		if !has(got, c.param, c.wantClass) {
			t.Errorf("%s: want %s/%s, got %+v", c.body, c.param, c.wantClass, got.Values)
		}
	}
	if c := values(t, `{"tool_choice":{"type":"auto","disable_parallel_tool_use":false}}`, EndpointMessages); has(c, "parallel_tool_calls", ClassPresent) {
		t.Errorf("disable_parallel_tool_use false: %+v", c.Values)
	}
	if c := values(t, `{"logit_bias":{}}`, EndpointChat); len(c.Values) != 0 {
		t.Errorf("logit_bias {}: %+v", c.Values)
	}
	// tool types are gated per endpoint
	if c := values(t, `{"tools":[{"type":"local_shell"}]}`, EndpointChat); !has(c, "tools", ClassUnknown) {
		t.Errorf("local_shell on chat: %+v", c.Values)
	}
	if c := values(t, `{"tool_choice":{"type":"tool","name":"f"}}`, EndpointChat); !has(c, "tool_choice", ClassUnknown) {
		t.Errorf("Anthropic tool_choice on chat: %+v", c.Values)
	}
	// a literal dotted top-level key is unknown, and classification is deterministic
	for i := 0; i < 50; i++ {
		c := values(t, `{"thinking.budget_tokens":99999,"thinking":{"type":"enabled","budget_tokens":5}}`, EndpointMessages)
		var n int64
		for _, v := range c.Values {
			if v.Param == "thinking.budget_tokens" {
				n = v.Number
			}
		}
		if n != 5 || len(c.Unknown) != 1 {
			t.Fatalf("dotted key: %+v %v", c.Values, c.Unknown)
		}
	}
	// trailing data after the object
	if _, err := Classify([]byte(`{"n":1} {"n":5}`), EndpointChat); !errors.Is(err, ErrNotObject) {
		t.Errorf("trailing data: %v", err)
	}
	// a huge cap goes through D10 like any other
	rs := planBundle(t)
	p, err := Plan(rs, []byte(`{"max_tokens":1e9}`), EndpointChat, abc, PlanOptions{})
	if err != nil || p.Servable()[0].TruncateTo != 65536 {
		t.Errorf("1e9: %+v %v", p.Servable(), err)
	}
	if _, err := Plan(rs, []byte(`{"max_tokens":-1}`), EndpointChat, abc, PlanOptions{}); err == nil {
		t.Error("negative cap served")
	}
	if _, err := Plan(rs, []byte(`{"thinking":"disabled"}`), EndpointChat, abc, PlanOptions{}); err == nil {
		t.Error("non-object container served")
	}
}

func TestPlanCapEdges(t *testing.T) {
	rs := mustParse(t, bundle(1, "", `
		{"identity":"a","model":"m","params":{"max_tokens":{"limits":{"min":16,"max":100}}}},
		{"identity":"b","model":"m","params":{"max_tokens":{"limits":{"min":200,"max":1000}}}}`))
	rows := []RowKey{{Identity: "a", Model: "m"}, {Identity: "b", Model: "m"}}
	for _, c := range []struct {
		body string
		want []string
	}{
		{`{"max_tokens":16}`, []string{"a"}},  // at a's min: served; below b's
		{`{"max_tokens":100}`, []string{"a"}}, // at max: fits, no truncation
		{`{"max_tokens":15}`, nil},
	} {
		p, _ := Plan(rs, []byte(c.body), EndpointChat, rows, PlanOptions{})
		if got := servable(p); !reflect.DeepEqual(got, c.want) {
			t.Errorf("%s: %v want %v", c.body, got, c.want)
		}
		if len(p.Rows) > 0 && p.Rows[0].TruncateTo != 0 {
			t.Errorf("%s: truncated %d", c.body, p.Rows[0].TruncateTo)
		}
	}
	// the cap across spellings is the largest; an unservable row gets no TruncateTo
	rs = planBundle(t)
	p, err := Plan(rs, []byte(`{"max_tokens":10,"max_completion_tokens":100000,"logprobs":true}`), EndpointChat, abc, PlanOptions{})
	if err != nil || p.Rows[0].Servable || p.Rows[0].TruncateTo != 0 || p.Rows[1].TruncateTo != 8192 {
		t.Errorf("spellings: %+v %v", p.Rows, err)
	}
	// a drop-class value a row refuses under require_parameters makes it unservable
	rs = mustParse(t, bundle(1, "", `{"identity":"a","model":"m","params":{"top_k":"rejected"}}`))
	if _, err := Plan(rs, []byte(`{"top_k":5}`), EndpointChat, rows[:1], PlanOptions{RequireParameters: true}); err == nil {
		t.Error("drop-class avoid ignored")
	}
	// the reject names the value every row avoids, not the first row's first
	rs = mustParse(t, bundle(1, "", `
		{"identity":"a","model":"m","params":{"logprobs":"rejected","n":"rejected"}},
		{"identity":"b","model":"m","params":{"logprobs":"supported","n":"rejected"}}`))
	_, err = Plan(rs, []byte(`{"logprobs":true,"n":2}`), EndpointChat, rows, PlanOptions{})
	var re *RejectError
	if !errors.As(err, &re) || re.Param != "n" {
		t.Errorf("reject names %v", err)
	}
}

func TestClassifyRound2(t *testing.T) {
	if c := values(t, `{"tools":[{"type":"function","Type":"web_search"}]}`, EndpointChat); has(c, "tools", "function") || !has(c, "tools", ClassUnknown) {
		t.Errorf("tool type variant: %+v", c.Values)
	}
	if c := values(t, `{"tools":[{"type":"tool_search_tool_regex_20251119","name":"s"}]}`, EndpointMessages); !has(c, "tools", "hosted") {
		t.Errorf("tool search: %+v", c.Values)
	}
	if c := values(t, `{"n":1.0,"top_logprobs":0.0,"service_tier":"default"}`, EndpointChat); len(c.Values) != 0 {
		t.Errorf("no-ops: %+v", c.Values)
	}
	if c := values(t, `{"service_tier":"standard_only"}`, EndpointMessages); len(c.Values) != 0 {
		t.Errorf("standard_only: %+v", c.Values)
	}
	if c := values(t, `{"service_tier":"auto"}`, EndpointChat); !has(c, "service_tier", "auto") {
		t.Errorf("auto: %+v", c.Values)
	}
}

func TestPlanRound2(t *testing.T) {
	// odd spellings are not folded: refused rather than guessed
	for _, b := range []string{`{"reasoning_effort":" None "}`, `{"reasoning_effort":"HIGH"}`} {
		if c := values(t, b, EndpointChat); !has(c, "reasoning_effort", ClassUnknown) {
			t.Errorf("%s: %+v", b, c.Values)
		}
	}
	for _, b := range []string{
		`{"response_format":{"type":"json_schema","json_schema":{"strict":"true"}}}`,
		`{"response_format":{"type":"json_schema","json_schema":{"strict":1}}}`,
		`{"response_format":{"type":"json_schema","Json_schema":{"strict":true}}}`,
	} {
		if c := values(t, b, EndpointChat); !has(c, "response_format", ClassUnknown) {
			t.Errorf("%s: %+v", b, c.Values)
		}
	}
	if c := values(t, `{"tool_choice":{"type":"any"}}`, EndpointChat); !has(c, "tool_choice", ClassUnknown) {
		t.Errorf("any on chat: %+v", c.Values)
	}
	if c := values(t, `{"tool_choice":{"type":"auto","disable_parallel_tool_use":true,"Disable_parallel_tool_use":false}}`, EndpointMessages); !has(c, "parallel_tool_calls", ClassUnknown) {
		t.Errorf("ambiguous disable_parallel_tool_use: %+v", c.Values)
	}
	if c := values(t, `{"stop":""}`, EndpointChat); !has(c, "stop", ClassPresent) {
		t.Errorf(`stop "": %+v`, c.Values)
	}
	for _, b := range []string{`{"thinking_budget":"5"}`, `{"chat_template_kwargs":{"thinking_budget":-1}}`} {
		if c := values(t, b, EndpointChat); len(c.Values) != 1 || c.Values[0].Class != ClassUnknown {
			t.Errorf("%s: %+v", b, c.Values)
		}
	}
	if c := values(t, `{"max_tokens":99999999999999999999}`, EndpointChat); c.Values[0].Number <= 1<<40 {
		t.Errorf("overflow: %+v", c.Values)
	}
	// deterministic with two unknown variants of one field
	for i := 0; i < 50; i++ {
		c := values(t, `{"max_tokens":-5,"Max_tokens":200000}`, EndpointChat)
		if len(c.Values) != 1 || c.Values[0].HasNumber {
			t.Fatalf("variants: %+v", c.Values)
		}
	}

	rs := mustParse(t, bundle(1, "", `
		{"identity":"a","model":"m","params":{"thinking.type":{"enabled":"supported"},"thinking.budget_tokens":{"present":"supported","limits":{"min":1024,"max":32000}},"thinking_budget":"supported","max_tokens":{"limits":{"min":16,"max":64000}}}}`))
	row := []RowKey{{Identity: "a", Model: "m"}}
	for _, c := range []struct {
		body string
		ok   bool
	}{
		{`{"thinking":{"type":"enabled","budget_tokens":32000},"max_tokens":40000}`, true},  // at max
		{`{"thinking":{"type":"enabled","budget_tokens":32001},"max_tokens":40000}`, false}, // above max
		{`{"thinking":{"type":"enabled","budget_tokens":1024},"max_tokens":40000}`, true},   // at min
		{`{"thinking":{"type":"enabled","budget_tokens":1023},"max_tokens":40000}`, false},
		// truncation leaves room above the largest budget spelling
		{`{"thinking":{"type":"enabled","budget_tokens":2000},"thinking_budget":70000,"max_tokens":100000}`, false},
		{`{"thinking":{"type":"enabled","budget_tokens":2000},"thinking_budget":60000,"max_tokens":100000}`, true},
		// every budget spelling is checked against its limits
		{`{"thinking":{"type":"enabled","budget_tokens":40000},"thinking_budget":10,"max_tokens":50000}`, false},
		// every cap spelling is checked against min
		{`{"max_completion_tokens":2,"max_tokens":40000}`, false},
	} {
		p, _ := Plan(rs, []byte(c.body), EndpointMessages, row, PlanOptions{})
		if got := len(p.Servable()) == 1; got != c.ok {
			t.Errorf("%s: servable %v want %v (%+v)", c.body, got, c.ok, p.Rows)
		}
	}
	// truncating exactly to the budget leaves no room
	rs = mustParse(t, bundle(1, "", `{"identity":"a","model":"m","params":{"thinking.type":{"enabled":"supported"},"thinking.budget_tokens":"supported","max_tokens":{"limits":{"max":8000}}}}`))
	if p, _ := Plan(rs, []byte(`{"thinking":{"type":"enabled","budget_tokens":8000},"max_tokens":9000}`), EndpointMessages, row, PlanOptions{}); len(p.Servable()) != 0 {
		t.Errorf("cap equal to budget: %+v", p.Rows)
	}
	if p, _ := Plan(rs, []byte(`{"thinking":{"type":"enabled","budget_tokens":7999},"max_tokens":9000}`), EndpointMessages, row, PlanOptions{}); len(p.Servable()) != 1 {
		t.Errorf("cap one above budget: %+v", p.Rows)
	}
}

func TestPlanSitesLeaveNoCap(t *testing.T) {
	// two sites whose limits intersect to nothing: no cap fits the row
	rs := mustParse(t, bundle(1, "", `
		{"identity":"a","model":"m","site":"sg","params":{"max_tokens":{"limits":{"min":2000}}}},
		{"identity":"a","model":"m","site":"de","params":{"max_tokens":{"limits":{"max":1000}}}}`))
	p, err := Plan(rs, []byte(`{"max_tokens":5000}`), EndpointChat, []RowKey{{Identity: "a", Model: "m"}}, PlanOptions{})
	if err == nil || p.Rows[0].TruncateTo != 0 {
		t.Errorf("served: %+v", p.Rows)
	}
	if _, err := Plan(rs, []byte(`{}`), EndpointChat, []RowKey{{Identity: "a", Model: "m"}}, PlanOptions{}); err == nil {
		t.Error("no cap sent, yet served by a row no cap fits")
	}
}

func TestPlanRound3(t *testing.T) {
	row := []RowKey{{Identity: "a", Model: "m"}}
	// a budget translated to thinking.budget_tokens meets its limits
	rs := mustParse(t, bundle(1, "", `{"identity":"a","model":"m","params":{
		"thinking.budget_tokens":{"present":"supported","limits":{"max":32000}},
		"thinking_budget":"translate:thinking.budget_tokens","max_tokens":{"limits":{"max":64000}}}}`))
	if p, _ := Plan(rs, []byte(`{"thinking_budget":100000,"max_tokens":1000}`), EndpointChat, row, PlanOptions{}); len(p.Servable()) != 0 {
		t.Errorf("translated budget above limits: %+v", p.Rows)
	}
	if p, _ := Plan(rs, []byte(`{"thinking_budget":1000,"max_tokens":2000}`), EndpointChat, row, PlanOptions{}); len(p.Servable()) != 1 {
		t.Errorf("translated budget within limits: %+v", p.Rows)
	}
	// no cap sent: the injected cap must leave room above the budget
	rs = mustParse(t, bundle(1, "", `{"identity":"a","model":"m","params":{"thinking_budget":"supported","chat_template_kwargs.thinking_budget":"supported","max_tokens":{"limits":{"max":32000}}}}`))
	for _, c := range []struct {
		body string
		ok   bool
	}{
		{`{"thinking_budget":50000}`, false},
		{`{"chat_template_kwargs":{"thinking_budget":40000}}`, false},
		{`{"thinking_budget":1000}`, true},
	} {
		if p, _ := Plan(rs, []byte(c.body), EndpointChat, row, PlanOptions{}); (len(p.Servable()) == 1) != c.ok {
			t.Errorf("%s: %+v", c.body, p.Rows)
		}
	}
	// max_output_tokens is a cap like the others; the largest spelling must fit
	rs = mustParse(t, bundle(1, "", `
		{"identity":"a","model":"m","params":{"max_tokens":{"limits":{"max":1000}}}},
		{"identity":"b","model":"m","params":{"max_tokens":{"limits":{"max":100000}}}}`))
	ab := []RowKey{{Identity: "a", Model: "m"}, {Identity: "b", Model: "m"}}
	for _, b := range []string{`{"max_output_tokens":5000}`, `{"max_tokens":10,"max_completion_tokens":5000}`, `{"max_completion_tokens":5000,"max_tokens":10}`} {
		if p, _ := Plan(rs, []byte(b), EndpointChat, ab, PlanOptions{}); !reflect.DeepEqual(servable(p), []string{"b"}) {
			t.Errorf("%s: %v", b, servable(p))
		}
	}
	if c := values(t, `{"max_output_tokens":"5"}`, EndpointResponses); !has(c, "max_output_tokens", ClassUnknown) {
		t.Errorf("max_output_tokens string: %+v", c.Values)
	}
	// classification gaps
	for _, c := range []struct {
		body, param, class string
		ep                 Endpoint
	}{
		{`{"service_tier":"auto"}`, "service_tier", "auto", EndpointMessages},
		{`{"top_logprobs":1}`, "top_logprobs", ClassPresent, EndpointChat},
		{`{"response_format":{"type":"json_schema","strict":true,"json_schema":{"strict":false}}}`, "response_format", "json_schema_strict", EndpointChat},
		{`{"tools":{"type":"function"}}`, "tools", ClassUnknown, EndpointChat},
		{`{"tool_choice":{"type":"auto"}}`, "tool_choice", ClassUnknown, EndpointChat},
		{`{"tools":[{"type":1,"name":"x"}]}`, "tools", ClassUnknown, EndpointMessages},
		{`{"tools":[{"type":["web_search_20250305"],"name":"x"}]}`, "tools", ClassUnknown, EndpointMessages},
	} {
		if got := values(t, c.body, c.ep); !has(got, c.param, c.class) {
			t.Errorf("%s: want %s/%s, got %+v", c.body, c.param, c.class, got.Values)
		}
	}
}

func TestFoldIndexMatchesEqualFold(t *testing.T) {
	for _, c := range []struct{ key, want string }{
		{"N", "n"}, {"Response_Format", "response_format"}, {"\u017ftop", "stop"}, {"TOOLS", "tools"}, {"Thinking", "thinking"},
	} {
		key := c.key
		if got, ok := foldCatalog(key); !ok || got != c.want {
			t.Errorf("%q: %q %v", key, got, ok)
		}
	}
	if got, ok := foldMember("reasoning", "EFFORT"); !ok || got != "reasoning.effort" {
		t.Errorf("member: %q %v", got, ok)
	}
	if _, ok := foldCatalog("user_x"); ok {
		t.Error("non-variant matched")
	}
	// every judged catalog name is reachable from its upper-case form
	for name, spec := range catalog {
		if spec.category == CategoryPassthrough || !strings.EqualFold(strings.ToUpper(name), name) {
			continue
		}
		if got, ok := foldIndex()[foldKey(strings.ToUpper(name))]; !ok || got != name {
			t.Errorf("%s: %q", name, got)
		}
	}
}

func TestClassifyManyUnknownKeysIsCheap(t *testing.T) {
	var b strings.Builder
	b.WriteString("{")
	for i := 0; i < 50000; i++ {
		if i > 0 {
			b.WriteString(",")
		}
		fmt.Fprintf(&b, `"k%d":1`, i)
	}
	b.WriteString("}")
	start := time.Now()
	if _, err := Classify([]byte(b.String()), EndpointChat); err != nil {
		t.Fatal(err)
	}
	if d := time.Since(start); d > time.Second { // order-of-magnitude bound (was a catalog scan per key)
		t.Errorf("50k unknown keys: %v", d)
	}
}
