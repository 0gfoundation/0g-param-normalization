package paramnorm

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"math"
	"sort"
	"strings"
	"sync"
	"unicode"
)

// Value is one parameter value a request carries, in the catalog's terms: the
// parameter's name (dotted for a container member) and its value class.
// Number holds the value when it is a whole number (HasNumber); Plan reads it
// for the output cap and thinking budgets only.
type Value struct {
	Param     string
	Class     string
	Number    int64
	HasNumber bool
}

// ClassUnknown is the class of a value the catalog has no class for (a tool
// type, a tool_choice form or a level it does not know): rules never answer
// it, so callers treat it as refused.
const ClassUnknown = "?"

// Classification is what a request body carries.
type Classification struct {
	// Values are the parameter values sent, one per (parameter, class), in a
	// stable order. Values that change nothing are left out: null, an empty
	// list or object, logprobs false, top_logprobs 0, n 1, a disabled switch
	// that is off by default (enable_search false, reasoning_split false).
	Values []Value
	// Unknown are the body fields the catalog does not know. Phase 1 records
	// them; an enforcing caller refuses the request (design section 3). A
	// container sent as something other than an object, a case variant of a
	// catalog field, and a cap that is no count are Values of ClassUnknown
	// instead: refused on every row.
	Unknown []string
}

// ErrNotObject is returned for a body that is not a JSON object.
var ErrNotObject = errors.New("paramnorm: request body is not a JSON object")

// Classify returns the parameter values a request body carries on endpoint
// ep. Spellings follow the endpoint: on /v1/messages an Anthropic client tool
// (no type, or "custom") is a function tool and tool_choice "any" is
// "required"; on chat and /v1/responses "custom" is OpenAI's freeform tool.
func Classify(body []byte, ep Endpoint) (Classification, error) {
	var top map[string]json.RawMessage
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.UseNumber()
	if err := dec.Decode(&top); err != nil || top == nil {
		return Classification{}, ErrNotObject
	}
	if _, err := dec.Token(); err != io.EOF {
		return Classification{}, ErrNotObject // data after the object
	}
	c := &classifier{ep: ep, seen: map[[2]string]bool{}}
	for key, raw := range top {
		cat, known := CategoryOf(key)
		if strings.Contains(key, ".") {
			known = false // a dotted name is a container member, never a top-level field
		}
		switch {
		case !known:
			// A case variant of a catalog field (Response_Format, N): a
			// case-insensitive decoder upstream (new-api is Go) may take it for
			// the field, so the field is sent with a value nobody can judge.
			if name, ok := foldCatalog(key); ok {
				c.add(Value{Param: name, Class: ClassUnknown})
				continue
			}
			c.unknown = append(c.unknown, key)
		case cat == CategoryStructural || cat == CategoryPassthrough:
		case cat == CategoryContainer:
			c.container(key, raw)
		default:
			c.field(key, raw)
		}
	}
	sort.Slice(c.values, func(i, j int) bool {
		if c.values[i].Param != c.values[j].Param {
			return c.values[i].Param < c.values[j].Param
		}
		return c.values[i].Class < c.values[j].Class
	})
	sort.Strings(c.unknown)
	return Classification{Values: c.values, Unknown: c.unknown}, nil
}

type classifier struct {
	ep      Endpoint
	values  []Value
	unknown []string
	seen    map[[2]string]bool
}

func (c *classifier) add(v Value) {
	if v.Class == ClassUnknown {
		v.Number, v.HasNumber = 0, false // one unknown value per parameter, whichever was read first
	}
	k := [2]string{v.Param, v.Class}
	if c.seen[k] {
		return
	}
	c.seen[k] = true
	c.values = append(c.values, v)
}

// foldCatalog returns the top-level catalog field key is a case variant of.
// A passthrough field's variant changes nothing anyone judges.
func foldCatalog(key string) (string, bool) {
	name, ok := foldIndex()[foldKey(key)]
	if !ok || strings.Contains(name, ".") {
		return "", false // a member name is never a top-level field
	}
	return name, true
}

// foldKey maps s to a key equal for every string strings.EqualFold matches
// with it: each rune becomes the smallest rune of its simple-folding orbit
// (so "ſtop" and "STOP" meet "stop").
func foldKey(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		m := r
		for f := unicode.SimpleFold(r); f != r; f = unicode.SimpleFold(f) {
			if f < m {
				m = f
			}
		}
		b.WriteRune(m)
	}
	return b.String()
}

// foldIndex maps foldKey of each judged catalog name (top-level fields,
// containers, and "container.member" names) to the name, so a case variant is
// found in one lookup however many unknown keys a body has.
var foldIndex = sync.OnceValue(func() map[string]string {
	idx := map[string]string{}
	for name, spec := range catalog {
		if spec.category != CategoryPassthrough {
			idx[foldKey(name)] = name
		}
	}
	for name := range containers {
		idx[foldKey(name)] = name
	}
	return idx
})

// member returns obj[key] read by its exact name, as a Python upstream reads
// it. ambiguous is set when another member differs from key only in case: a
// Go decoder upstream may read that one instead.
func member(obj map[string]json.RawMessage, key string) (raw json.RawMessage, ok, ambiguous bool) {
	raw, ok = obj[key]
	for k := range obj {
		if k != key && strings.EqualFold(k, key) {
			ambiguous = true
		}
	}
	return raw, ok, ambiguous
}

func isNull(raw json.RawMessage) bool {
	return len(bytes.TrimSpace(raw)) == 0 || string(bytes.TrimSpace(raw)) == "null"
}

func (c *classifier) container(key string, raw json.RawMessage) {
	if isNull(raw) {
		return
	}
	var members map[string]json.RawMessage
	if err := json.Unmarshal(raw, &members); err != nil {
		// Not an object: nothing to judge member by member, so it is refused,
		// never forwarded unjudged.
		c.add(Value{Param: key, Class: ClassUnknown})
		return
	}
	for m, mraw := range members {
		name := key + "." + m
		if _, ok := catalog[name]; !ok {
			if folded, ok := foldMember(key, m); ok {
				c.add(Value{Param: folded, Class: ClassUnknown})
				continue
			}
			if cat, known := CategoryOf(name); known && cat == CategoryPassthrough {
				continue
			}
			c.unknown = append(c.unknown, name)
			continue
		}
		c.field(name, mraw)
	}
}

// field classifies one catalog parameter's value.
func (c *classifier) field(name string, raw json.RawMessage) {
	if isNull(raw) {
		return
	}
	spec := catalog[name]
	switch name {
	case "tools":
		c.tools(raw)
		return
	case "tool_choice":
		c.toolChoice(raw)
		return
	case "function_call":
		c.functionCall(raw)
		return
	case "response_format", "text.format", "output_config.format":
		c.outputFormat(name, raw)
		return
	}
	// A parameter with value classes other than "present": the class is the value.
	if !(len(spec.classes) == 1 && spec.classes[0] == ClassPresent) {
		class := classOf(spec, raw)
		if name == "service_tier" && c.ep == EndpointMessages {
			// Anthropic's spelling: auto, standard_only (as default).
			var s string
			_ = json.Unmarshal(raw, &s)
			if s == "standard_only" {
				class = "default"
			}
		}
		if name == "service_tier" && class == "default" {
			return // the default tier asks for nothing
		}
		c.add(Value{Param: name, Class: class})
		return
	}
	if noOp(name, raw) {
		return
	}
	v := Value{Param: name, Class: ClassPresent}
	if n, ok := intValue(raw); ok {
		v.Number, v.HasNumber = n, true
	}
	if numericParams[name] && (!v.HasNumber || v.Number < 0) {
		v.Class = ClassUnknown // a cap or budget that is no count: nobody can say it fits
	}
	c.add(v)
}

// classOf maps a scalar value to one of the parameter's classes: booleans to
// "true" / "false", strings as sent (an upstream may not fold " None " or
// "HIGH" to a level), anything else unknown.
func classOf(spec paramSpec, raw json.RawMessage) string {
	var b bool
	if json.Unmarshal(raw, &b) == nil {
		if b {
			return pick(spec, "true")
		}
		return pick(spec, "false")
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return pick(spec, s)
	}
	return ClassUnknown
}

func pick(spec paramSpec, class string) string {
	if spec.hasClass(class) {
		return class
	}
	return ClassUnknown
}

// numericParams are the parameters whose value is a token count checked
// against a row's limits.
var numericParams = map[string]bool{
	"max_tokens": true, "max_completion_tokens": true, "max_output_tokens": true,
	"thinking.budget_tokens": true, "thinking_budget": true, "chat_template_kwargs.thinking_budget": true,
}

// foldMember returns the catalog member name a container member is a case
// variant of.
func foldMember(container, m string) (string, bool) {
	name, ok := foldIndex()[foldKey(container+"."+m)]
	if !ok || !strings.HasPrefix(name, container+".") {
		return "", false
	}
	return name, true
}

// noOp reports a presence-judged value that changes nothing. An empty object
// asks for nothing only where it is a map of overrides (logit_bias): an
// options object such as web_search_options {} turns its feature on.
func noOp(name string, raw json.RawMessage) bool {
	t := string(bytes.TrimSpace(raw))
	switch t {
	case "[]":
		return true
	case "{}":
		return name == "logit_bias"
	}
	switch name {
	case "logprobs", "enable_search", "reasoning_split", "preserve_thinking", "chat_template_kwargs.preserve_thinking":
		return t == "false" // switches off by default
	case "chat_template_kwargs.clear_thinking", "thinking.clear_thinking", "parallel_tool_calls":
		return t == "true" // on by default: only false asks for something
	case "top_logprobs":
		n, ok := intValue(raw)
		return ok && n == 0
	case "n":
		n, ok := intValue(raw)
		return ok && n == 1
	}
	return false
}

func intValue(raw json.RawMessage) (int64, bool) {
	if t := bytes.TrimSpace(raw); len(t) == 0 || t[0] == '"' {
		return 0, false // json.Number would take a quoted number
	}
	var n json.Number
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	if dec.Decode(&n) != nil {
		return 0, false
	}
	if i, err := n.Int64(); err == nil {
		return i, true
	}
	// A whole number written as a float (1024.0, 1e6), or one too large for int64.
	f, err := n.Float64()
	switch {
	case err != nil && !math.IsInf(f, 0):
		return 0, false
	case f >= math.MaxInt64:
		return math.MaxInt64, true // above every limit
	case f <= math.MinInt64:
		return math.MinInt64, true
	case f != math.Trunc(f):
		return 0, false
	}
	return int64(f), true
}

// Hosted (provider-run) tool types: Responses built-ins and Anthropic server
// tools. Tools a vendor defines but the client runs (Anthropic bash,
// text_editor, computer, memory; Responses computer_use_preview, shell,
// apply_patch) are no class the catalog has: unknown.
var hostedToolPrefixes = []string{
	"web_search", "file_search", "code_interpreter", "image_generation", "mcp", "code_execution", "web_fetch",
	"tool_search_tool", // Anthropic tool search (regex, bm25)
}

func (c *classifier) tools(raw json.RawMessage) {
	var tools []map[string]json.RawMessage
	if err := json.Unmarshal(raw, &tools); err != nil {
		c.add(Value{Param: "tools", Class: ClassUnknown})
		return
	}
	for _, t := range tools {
		r, has, amb := member(t, "type")
		var typ string
		if has && !isNull(r) && json.Unmarshal(r, &typ) != nil {
			amb = true // a type that is not a string
		}
		if amb {
			c.add(Value{Param: "tools", Class: ClassUnknown})
			continue
		}
		c.add(Value{Param: "tools", Class: c.toolClass(typ)})
	}
}

func (c *classifier) toolClass(typ string) string {
	switch {
	case typ == "function":
		return "function"
	case c.ep == EndpointMessages && (typ == "" || typ == "custom"):
		return "function" // an Anthropic client tool
	case typ == "custom" && c.ep != EndpointMessages:
		return "custom" // OpenAI freeform tool
	case typ == "local_shell" && c.ep == EndpointResponses:
		return "local_shell"
	}
	for _, p := range hostedToolPrefixes {
		if strings.HasPrefix(typ, p) {
			return "hosted"
		}
	}
	return ClassUnknown
}

func (c *classifier) toolChoice(raw json.RawMessage) {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		switch s {
		case "none", "auto", "required":
			c.add(Value{Param: "tool_choice", Class: s})
		default:
			c.add(Value{Param: "tool_choice", Class: ClassUnknown})
		}
		return
	}
	var obj map[string]json.RawMessage
	if json.Unmarshal(raw, &obj) != nil || obj == nil {
		c.add(Value{Param: "tool_choice", Class: ClassUnknown})
		return
	}
	typRaw, _, amb := member(obj, "type")
	var typ string
	_ = json.Unmarshal(typRaw, &typ)
	class := ClassUnknown
	switch {
	case amb:
	case typ == "function":
		class = "function"
	case c.ep == EndpointMessages && typ == "tool":
		class = "function"
	case c.ep == EndpointMessages && typ == "any":
		class = "required"
	case c.ep == EndpointMessages && (typ == "auto" || typ == "none"):
		class = typ
	}
	c.add(Value{Param: "tool_choice", Class: class})
	if c.ep == EndpointMessages {
		dp, ok, amb := member(obj, "disable_parallel_tool_use")
		switch {
		case amb:
			c.add(Value{Param: "parallel_tool_calls", Class: ClassUnknown})
		case ok && string(bytes.TrimSpace(dp)) == "true":
			c.add(Value{Param: "parallel_tool_calls", Class: ClassPresent})
		}
	}
}

func (c *classifier) functionCall(raw json.RawMessage) {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		c.add(Value{Param: "function_call", Class: pick(catalog["function_call"], s)})
		return
	}
	c.add(Value{Param: "function_call", Class: "function"})
}

// outputFormat classifies response_format (chat), text.format (Responses) and
// output_config.format (Messages). A json_schema is strict when it says so;
// Anthropic structured output is always enforced.
func (c *classifier) outputFormat(name string, raw json.RawMessage) {
	var f map[string]json.RawMessage
	if json.Unmarshal(raw, &f) != nil || f == nil {
		c.add(Value{Param: name, Class: ClassUnknown})
		return
	}
	typRaw, _, amb := member(f, "type")
	var typ string
	_ = json.Unmarshal(typRaw, &typ)
	// strict at the top (Responses text.format) or inside json_schema (chat)
	strict, strictAmb := false, false
	isTrue := func(obj map[string]json.RawMessage) {
		r, ok, a := member(obj, "strict")
		t := string(bytes.TrimSpace(r))
		// anything but a JSON boolean ("true", 1) may be read either way upstream
		strictAmb = strictAmb || a || (ok && t != "true" && t != "false" && t != "null")
		strict = strict || (ok && t == "true")
	}
	isTrue(f)
	if js, ok, a := member(f, "json_schema"); a {
		strictAmb = true
	} else if ok {
		var inner map[string]json.RawMessage
		if json.Unmarshal(js, &inner) == nil {
			isTrue(inner)
		}
	}
	class := ClassUnknown
	switch {
	case amb:
	case typ == "text", typ == "json_object":
		class = typ
	case typ == "json_schema" && strictAmb:
	case typ == "json_schema" && (strict || name == "output_config.format"):
		class = "json_schema_strict"
	case typ == "json_schema":
		class = "json_schema"
	}
	if class == "text" {
		return // plain text is the default: nothing asked for
	}
	c.add(Value{Param: name, Class: class})
}
