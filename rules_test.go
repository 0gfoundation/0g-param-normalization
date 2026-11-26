package paramnorm

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math/rand/v2"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
)

func bundle(version int, families, rows string) []byte {
	return []byte(fmt.Sprintf(`{"schema":1,"version":%d,"env":"test","families":[%s],"rows":[%s]}`, version, families, rows))
}

func mustParse(t *testing.T, b []byte) *Ruleset {
	t.Helper()
	rs, err := ParseRules(b)
	if err != nil {
		t.Fatalf("ParseRules: %v", err)
	}
	return rs
}

var (
	supported = Status{Kind: StatusSupported}
	rejected  = Status{Kind: StatusRejected}
	ignored   = Status{Kind: StatusIgnored}
	unknown   = Status{}
)

func TestBuiltinRulesParse(t *testing.T) {
	s, err := NewStore(BuiltinRules(), StoreOptions{Env: "test"})
	if err != nil {
		t.Fatalf("builtin rules do not load: %v", err)
	}
	if got := s.Rules().Lookup(RowKey{Identity: "x", Model: "y"}, EndpointChat, "tool_choice", "auto"); got != unknown {
		t.Fatalf("empty builtin rules: got %+v, want unknown", got)
	}
}

func TestParseRulesRefusesWhatTheFormatCannotExpress(t *testing.T) {
	row := func(params string) []byte {
		return bundle(1, "", `{"identity":"a","model":"m","params":`+params+`}`)
	}
	cases := map[string][]byte{
		"unknown parameter":            row(`{"bogus":"supported"}`),
		"protected field":              row(`{"messages":"rejected"}`),
		"engine control field":         row(`{"chat_template_kwargs":"rejected"}`),
		"model field":                  row(`{"model":"rejected"}`),
		"unknown value class":          row(`{"tool_choice":{"sometimes":"supported"}}`),
		"bad status":                   row(`{"stop":"maybe"}`),
		"unknown translation":          row(`{"reasoning_effort":{"high":"translate:messages"}}`),
		"translation on wrong param":   row(`{"stop":"translate:enable_thinking"}`),
		"bare status on multi-class":   row(`{"tool_choice":"supported"}`),
		"limit on param without limit": row(`{"stop":{"present":"supported","limits":{"max":5}}}`),
		"negative limit":               row(`{"max_tokens":{"limits":{"max":-1}}}`),
		"zero limit":                   row(`{"max_tokens":{"limits":{"min":0}}}`),
		"non-integer limit":            row(`{"max_tokens":{"limits":{"max":1.5}}}`),
		"min above max":                row(`{"max_tokens":{"limits":{"min":100,"max":10}}}`),
		"bare max as limit":            row(`{"max_tokens":{"max":10}}`),
		"unknown bound":                row(`{"max_tokens":{"limits":{"ceiling":10}}}`),
		"empty limits":                 row(`{"max_tokens":{"limits":{}}}`),
		"none to a thinking level":     row(`{"reasoning_effort":{"none":"translate:level:max"}}`),
		"none to an effort":            row(`{"reasoning_effort":{"none":"translate:output_config.effort"}}`),
		"duplicate param key":          row(`{"stop":"supported","stop":"rejected"}`),
		"duplicate top-level key":      []byte(`{"schema":1,"version":9,"version":2,"env":"test","families":[],"rows":[]}`),
		"upper-case key":               []byte(`{"schema":1,"version":9,"VERSION":2,"env":"test","families":[],"rows":[]}`),
		"upper-case param":             row(`{"Stop":"supported"}`),
		"long s folds to s":            []byte(`{"schema":1,"version":2,"families":[],"rows":[],"verſion":99}`),
		"long s in params":             bundle(1, "", `{"identity":"x","model":"m","params":{"stop":"supported"},"paramſ":{"stop":"rejected"}}`),
		"non-ascii key":                row(`{"stöp":"supported"}`),
		"zero-width in identity":       bundle(1, "", "{\"identity\":\"aliyun\u200b\",\"model\":\"q\",\"params\":{\"stop\":\"rejected\"}}"),
		"look-alike in identity":       bundle(1, "", `{"identity":"аliyun","model":"q","params":{"stop":"rejected"}}`),
		"space in model":               bundle(1, "", `{"identity":"a","model":"q 3","params":{"stop":"rejected"}}`),
		"zero-width in model":          bundle(1, "", "{\"identity\":\"a\",\"model\":\"q\u200b\",\"params\":{\"stop\":\"rejected\"}}"),
		"short address":                bundle(1, "", `{"model":"q","address":"0xab","params":{"stop":"rejected"}}`),
		"non-hex address":              bundle(1, "", `{"model":"q","address":"0xzz00000000000000000000000000000000000000","params":{"stop":"rejected"}}`),
		"non-ascii site":               bundle(1, "", `{"identity":"a","model":"q","site":"dé","params":{"stop":"rejected"}}`),
		"non-ascii family identity":    bundle(1, `{"identity":"kіmi","models":"*","params":{"stop":"rejected"}}`, ""),
		"space in family pattern":      bundle(1, `{"identity":"a","models":"qwen *","params":{"stop":"rejected"}}`, ""),
		"pattern in row model":         bundle(1, "", `{"identity":"a","model":"gpt-5*","params":{"stop":"rejected"}}`),
		"bidi override in note":        bundle(1, "", "{\"identity\":\"a\",\"model\":\"q\",\"note\":\"ok \u202e x\",\"params\":{\"stop\":\"rejected\"}}"),
		"zero-width in source":         bundle(1, "", "{\"identity\":\"a\",\"model\":\"q\",\"source\":{\"stop\":\"probe\u2066\"},\"params\":{\"stop\":\"rejected\"}}"),
		"newline in family note":       bundle(1, `{"identity":"a","models":"*","note":"a\nb","params":{"stop":"rejected"}}`, ""),
		"control in bundle note":       []byte("{\"schema\":1,\"version\":1,\"note\":\"x\u0007\",\"env\":\"test\",\"families\":[],\"rows\":[]}"),
		"trailing unicode space":       append(bundle(1, "", ""), []byte("\u0085")...),
		"trailing bracket":             append(bundle(1, "", ""), []byte(`]]`)...),
		"site and siteless mix":        bundle(1, "", `{"identity":"a","model":"m","params":{"stop":"rejected"}},{"identity":"a","model":"m","site":"de","params":{"n":"rejected"}}`),
		"any identity on a row":        bundle(1, "", `{"identity":"*","model":"m","params":{"stop":"rejected"}}`),
		"empty params":                 row(`{}`),
		"empty param rule":             row(`{"tool_choice":{}}`),
		"source without rule":          bundle(1, "", `{"identity":"a","model":"m","params":{"stop":"rejected"},"source":{"tools":"probe"}}`),
		"unknown top-level field":      []byte(`{"schema":1,"version":1,"families":[],"rows":[],"extra":1}`),
		"unknown row field":            bundle(1, "", `{"identity":"a","model":"m","params":{"stop":"rejected"},"target":"x"}`),
		"no identity and no address":   bundle(1, "", `{"model":"m","params":{"stop":"rejected"}}`),
		"missing model":                bundle(1, "", `{"identity":"a","params":{"stop":"rejected"}}`),
		"duplicate row":                bundle(1, "", `{"identity":"a","model":"m","params":{"stop":"rejected"}},{"identity":"a","model":"m","params":{"n":"rejected"}}`),
		"wrong schema":                 []byte(`{"schema":2,"version":1,"env":"test","families":[],"rows":[]}`),
		"version zero":                 []byte(`{"schema":1,"version":0,"env":"test","families":[],"rows":[]}`),
		"trailing data":                append(bundle(1, "", ""), []byte(` {}`)...),
		"not json":                     []byte(`schema=1`),
	}
	for name, b := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := ParseRules(b); err == nil {
				t.Fatalf("accepted %s", b)
			}
		})
	}
}

func TestLookupLayers(t *testing.T) {
	rs := mustParse(t, bundle(3,
		`{"identity":"a1","models":"gpt-5*","params":{"stop":"rejected","top_k":"rejected","tool_choice":{"auto":"supported","required":"ignored"}}}`,
		`{"identity":"a1","model":"gpt-5.1","params":{"stop":"supported","reasoning_effort":{"none":"translate:enable_thinking","high":"supported"}}},
		 {"identity":"a1","model":"gpt-5.1","address":"0x00000000000000000000000000000000000000b1","params":{"tool_choice":{"required":"rejected"}}}`))

	row := RowKey{Identity: "a1", Model: "gpt-5.1"}
	check := func(row RowKey, param, class string, want Status) {
		t.Helper()
		if got := rs.Lookup(row, EndpointChat, param, class); got != want {
			t.Errorf("Lookup(%+v, %s, %s) = %+v, want %+v", row, param, class, got, want)
		}
	}
	check(row, "stop", ClassPresent, supported)    // row overrides family
	check(row, "top_k", ClassPresent, rejected)    // family default
	check(row, "tool_choice", "required", ignored) // family default
	check(row, "tool_choice", "none", unknown)     // nothing says
	check(row, "reasoning_effort", "none", Status{Kind: StatusTranslate, Translation: "enable_thinking"})
	check(RowKey{Identity: "a1", Model: "gpt-5.1", Address: "0x00000000000000000000000000000000000000b1"}, "tool_choice", "required", rejected) // address layer
	check(RowKey{Identity: "a1", Model: "gpt-5.1", Address: "0x00000000000000000000000000000000000000c1"}, "tool_choice", "required", ignored)  // other address: no override
	check(RowKey{Identity: "a1", Model: "gpt-4o"}, "stop", ClassPresent, unknown)                                                               // family does not match
	check(RowKey{Identity: "a2", Model: "gpt-5.1"}, "stop", ClassPresent, unknown)                                                              // other identity
	check(row, "bogus", ClassPresent, unknown)                                                                                                  // unknown parameter
}

func TestLookupMoreSpecificFamilyWins(t *testing.T) {
	rs := mustParse(t, bundle(1,
		`{"identity":"a","models":"*","params":{"stop":"rejected"}},{"identity":"a","models":"qwen*","params":{"stop":"supported"}}`, ""))
	if got := rs.Lookup(RowKey{Identity: "a", Model: "qwen3"}, EndpointChat, "stop", ClassPresent); got != supported {
		t.Fatalf("the more specific family should win: got %+v", got)
	}
	if got := rs.Lookup(RowKey{Identity: "a", Model: "glm"}, EndpointChat, "stop", ClassPresent); got != rejected {
		t.Fatalf("got %+v", got)
	}
}

func TestLookupSitesIntersect(t *testing.T) {
	rs := mustParse(t, bundle(1,
		`{"identity":"vod","models":"*","params":{"top_p":"supported"}}`,
		`{"identity":"vod","model":"m","site":"sg","params":{"stop":"supported","n":"supported","logprobs":"supported"}},
		 {"identity":"vod","model":"m","site":"de","params":{"stop":"supported","n":"rejected"}}`))
	row := RowKey{Identity: "vod", Model: "m"}
	cases := []struct {
		param string
		want  Status
	}{
		{"stop", supported},   // both sites agree
		{"n", unknown},        // sites disagree
		{"logprobs", unknown}, // one site has no rule for it
		{"top_p", supported},  // neither site names it: the family default holds for both
	}
	for _, c := range cases {
		if got := rs.Lookup(row, EndpointChat, c.param, ClassPresent); got != c.want {
			t.Errorf("%s: got %+v, want %+v", c.param, got, c.want)
		}
	}
}

func TestLimits(t *testing.T) {
	rs := mustParse(t, bundle(1,
		`{"identity":"a","models":"*","params":{"max_tokens":{"limits":{"min":16,"max":8192}}}},
		 {"identity":"a","models":"wide*","params":{"max_tokens":{"limits":{"max":32768}}}}`,
		`{"identity":"a","model":"big","params":{"max_tokens":{"limits":{"max":65536}}}},
		 {"identity":"a","model":"two","site":"x","params":{"max_tokens":{"limits":{"min":8,"max":32768}}}},
		 {"identity":"a","model":"two","site":"y","params":{"max_tokens":{"limits":{"min":32,"max":16384}}}},
		 {"identity":"a","model":"half","site":"x","params":{"max_tokens":{"limits":{"max":65536}}}},
		 {"identity":"a","model":"half","site":"y","params":{"stop":"supported"}}`))
	cases := []struct {
		model string
		want  Limits
	}{
		{"small", Limits{Min: 16, Max: 8192}},
		{"big", Limits{Min: 16, Max: 65536}},    // a row setting only max keeps the family min
		{"wide-1", Limits{Min: 16, Max: 32768}}, // a later family setting only max keeps the earlier min
		{"two", Limits{Min: 32, Max: 16384}},    // sites: highest min, lowest max
		{"half", Limits{Min: 16, Max: 8192}},    // a site without its own range keeps the family's

	}
	for _, c := range cases {
		got, ok := rs.Limits(RowKey{Identity: "a", Model: c.model}, EndpointChat, "max_tokens")
		if !ok || got != c.want {
			t.Errorf("%s: got %+v %v, want %+v", c.model, got, ok, c.want)
		}
	}
	if _, ok := rs.Limits(RowKey{Identity: "b", Model: "x"}, EndpointChat, "max_tokens"); ok {
		t.Error("limits for an unknown row")
	}
}

type fakeVerifier struct {
	signer string
	err    error
}

func (f fakeVerifier) Verify(_, _ []byte) (string, error) { return f.signer, f.err }

func TestStoreLoad(t *testing.T) {
	const signer = "0xAbC0000000000000000000000000000000000001"
	good := fakeVerifier{signer: strings.ToLower(signer)}
	v2 := bundle(2, "", `{"identity":"a","model":"m","params":{"stop":"supported"}}`)

	newStore := func() *Store {
		s, err := NewStore(BuiltinRules(), StoreOptions{Env: "test"})
		if err != nil {
			t.Fatal(err)
		}
		return s
	}
	unchanged := func(s *Store) {
		t.Helper()
		if s.Rules().Version() != 1 {
			t.Fatalf("rules changed after a failed load: version %d", s.Rules().Version())
		}
	}

	t.Run("signed and newer", func(t *testing.T) {
		s := newStore()
		if err := s.Load(v2, nil, good, signer); err != nil {
			t.Fatal(err)
		}
		if s.Rules().Version() != 2 || s.Rules().Lookup(RowKey{Identity: "a", Model: "m"}, EndpointChat, "stop", ClassPresent) != supported {
			t.Fatal("bundle not in force")
		}
		if s.Rules().Hash() == "" {
			t.Fatal("no hash")
		}
	})
	t.Run("bad signature", func(t *testing.T) {
		s := newStore()
		if err := s.Load(v2, nil, fakeVerifier{err: errors.New("bad")}, signer); err == nil {
			t.Fatal("accepted")
		}
		unchanged(s)
	})
	t.Run("wrong signer", func(t *testing.T) {
		s := newStore()
		if err := s.Load(v2, nil, fakeVerifier{signer: "0x0000000000000000000000000000000000000002"}, signer); err == nil {
			t.Fatal("accepted")
		}
		unchanged(s)
	})
	t.Run("no verifier", func(t *testing.T) {
		s := newStore()
		if err := s.Load(v2, nil, nil, signer); err == nil {
			t.Fatal("accepted")
		}
		if err := s.Load(v2, nil, good, ""); err == nil {
			t.Fatal("accepted without a signer")
		}
		unchanged(s)
	})
	t.Run("invalid bundle keeps the last valid rules", func(t *testing.T) {
		s := newStore()
		if err := s.Load(v2, nil, good, signer); err != nil {
			t.Fatal(err)
		}
		bad := bundle(3, "", `{"identity":"a","model":"m","params":{"messages":"rejected"}}`)
		if err := s.Load(bad, nil, good, signer); err == nil {
			t.Fatal("accepted")
		}
		if s.Rules().Version() != 2 {
			t.Fatalf("version %d, want 2", s.Rules().Version())
		}
	})
	t.Run("same or lower version refused", func(t *testing.T) {
		s := newStore()
		if err := s.Load(v2, nil, good, signer); err != nil {
			t.Fatal(err)
		}
		for _, v := range []int{2, 1} {
			err := s.Load(bundle(v, "", ""), nil, good, signer)
			if !errors.Is(err, ErrStaleRules) {
				t.Fatalf("version %d: err %v, want ErrStaleRules", v, err)
			}
		}
		if err := s.LoadUnsigned(bundle(2, "", "")); !errors.Is(err, ErrStaleRules) {
			t.Fatalf("unsigned rollback: err %v", err)
		}
	})
	t.Run("unsigned", func(t *testing.T) {
		s := newStore()
		if err := s.LoadUnsigned(v2); err != nil {
			t.Fatal(err)
		}
		if s.Rules().Version() != 2 {
			t.Fatal("not loaded")
		}
	})
}

func TestStoreConcurrentReadsAndLoads(t *testing.T) {
	s, err := NewStore(BuiltinRules(), StoreOptions{Env: "test"})
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for r := 0; r < 4; r++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 2000; i++ {
				_ = s.Rules().Lookup(RowKey{Identity: "a", Model: "m"}, EndpointChat, "stop", ClassPresent)
			}
		}()
	}
	for v := 2; v < 200; v++ {
		wg.Add(1)
		go func(v int) {
			defer wg.Done()
			_ = s.LoadUnsigned(bundle(v, "", `{"identity":"a","model":"m","params":{"stop":"supported"}}`))
		}(v)
	}
	wg.Wait()
	if s.Rules().Version() != 199 {
		t.Fatalf("version %d, want the highest loaded (199)", s.Rules().Version())
	}
}

func TestCatalogIsConsistent(t *testing.T) {
	for name, spec := range catalog {
		if (len(spec.classes) == 0) != (spec.category == CategoryPassthrough) {
			t.Errorf("%s: a %s parameter with %d value classes", name, spec.category, len(spec.classes))
		}
		for c := range spec.classCategory {
			if !spec.hasClass(c) {
				t.Errorf("%s: category for unknown class %q", name, c)
			}
		}
		if len(spec.translations) > 0 && spec.category != CategoryTranslate {
			t.Errorf("%s: translations on a %s parameter", name, spec.category)
		}
	}
	for _, f := range []string{"messages", "model", "tools_content", ParamChatTemplateKwargs, "chat_template", "logits_processors", "lora_path"} {
		if _, ok := catalog[f]; ok {
			t.Errorf("%s must not be nameable by rules", f)
		}
	}
}

func TestLookupMaxLevelAndStrictSchema(t *testing.T) {
	rs := mustParse(t, bundle(1, "", `{"identity":"a","model":"m","params":{
		"reasoning_effort":{"max":"supported","xhigh":"translate:level:high","none":"translate:enable_thinking"},
		"response_format":{"json_schema":"supported","json_schema_strict":"ignored"}}}`))
	row := RowKey{Identity: "a", Model: "m"}
	if got := rs.Lookup(row, EndpointChat, "reasoning_effort", "max"); got != supported {
		t.Errorf("max level: %+v", got)
	}
	if got := rs.Lookup(row, EndpointChat, "reasoning_effort", "xhigh"); got != (Status{Kind: StatusTranslate, Translation: "level:high"}) {
		t.Errorf("xhigh: %+v", got)
	}
	if got := rs.Lookup(row, EndpointChat, "response_format", "json_schema_strict"); got != ignored {
		t.Errorf("strict schema: %+v", got)
	}
}

func TestFamiliesAnyIdentityAndSlashes(t *testing.T) {
	rs := mustParse(t, bundle(1,
		`{"identity":"up1","models":"qwen/*","params":{"stop":"rejected"}},
		 {"identity":"*","models":"qwen/*","params":{"stop":"supported","n":"rejected"}}`, ""))
	if got := rs.Lookup(RowKey{Identity: "up2", Model: "qwen/qwen3"}, EndpointChat, "n", ClassPresent); got != rejected {
		t.Errorf("any-identity family, '*' across '/': %+v", got)
	}
	// The identity family applies after the any-identity one even though it comes first in the file.
	if got := rs.Lookup(RowKey{Identity: "up1", Model: "qwen/qwen3"}, EndpointChat, "stop", ClassPresent); got != rejected {
		t.Errorf("identity family should override: %+v", got)
	}
	for pattern, model := range map[string]string{"a*c": "abc", "*": "x/y", "gpt-5*-mini": "gpt-5.1-mini", "m": "m"} {
		if !matchModel(pattern, model) {
			t.Errorf("%q should match %q", pattern, model)
		}
	}
	for pattern, model := range map[string]string{"a*c": "abd", "gpt-5*-mini": "gpt-5.1", "m": "mm", "ab*b": "ab"} {
		if matchModel(pattern, model) {
			t.Errorf("%q should not match %q", pattern, model)
		}
	}
}

func TestNilRulesetAndZeroStore(t *testing.T) {
	var s Store
	if got := s.Rules().Lookup(RowKey{Identity: "a", Model: "m"}, EndpointChat, "stop", ClassPresent); got != unknown {
		t.Fatalf("zero store: %+v", got)
	}
	if _, ok := s.Rules().Limits(RowKey{Identity: "a", Model: "m"}, EndpointChat, "max_tokens"); ok {
		t.Fatal("zero store has limits")
	}
	if s.Rules().Version() != 0 || s.Rules().Hash() != "" {
		t.Fatal("zero store version/hash")
	}
}

func TestStoreFloorSurvivesRestart(t *testing.T) {
	good := fakeVerifier{signer: "0x1"}
	// A process that already served version 5 restarts with the floor raised to 5:
	// replaying the older signed version 2 must fail.
	s, err := NewStore(BuiltinRules(), StoreOptions{Env: "test", MinVersion: 5})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Load(bundle(2, "", ""), nil, good, "0x1"); !errors.Is(err, ErrStaleRules) {
		t.Fatalf("replay below the floor: err %v", err)
	}
	if err := s.LoadUnsigned(bundle(4, "", "")); !errors.Is(err, ErrStaleRules) {
		t.Fatalf("unsigned below the floor: err %v", err)
	}
	if err := s.Load(bundle(5, "", ""), nil, good, "0x1"); err != nil {
		t.Fatalf("version at the floor: %v", err)
	}
}

func TestParseRulesRefusesDeepNestingWithoutCrashing(t *testing.T) {
	const depth = 2_000_000
	b := []byte(`{"schema":1,"version":1,"env":"test","families":[],"rows":[],"x":` + strings.Repeat("[", depth) + strings.Repeat("]", depth) + `}`)
	if _, err := ParseRules(b); err == nil || !strings.Contains(err.Error(), "nested deeper") {
		t.Fatalf("err %v, want the depth limit", err)
	}
}

func TestAddressCaseInsensitive(t *testing.T) {
	rs := mustParse(t, bundle(1, "", `{"identity":"a","model":"m","params":{"tool_choice":{"required":"supported"}}},
		{"identity":"a","model":"m","address":"0xAbCd00000000000000000000000000000000Ef01","params":{"tool_choice":{"required":"ignored"},"max_tokens":{"limits":{"max":100}}}}`))
	for _, addr := range []string{"0xAbCd00000000000000000000000000000000Ef01", "0xabcd00000000000000000000000000000000ef01", "0XABCD00000000000000000000000000000000EF01"} {
		row := RowKey{Identity: "a", Model: "m", Address: addr}
		if got := rs.Lookup(row, EndpointChat, "tool_choice", "required"); got != ignored {
			t.Errorf("%s: %+v", addr, got)
		}
		if l, ok := rs.Limits(row, EndpointChat, "max_tokens"); !ok || l.Max != 100 {
			t.Errorf("%s limits: %+v %v", addr, l, ok)
		}
	}
	if _, err := ParseRules(bundle(1, "", `{"identity":"a","model":"m","address":"0xAB000000000000000000000000000000000000AB","params":{"stop":"rejected"}},{"identity":"a","model":"m","address":"0xab000000000000000000000000000000000000ab","params":{"n":"rejected"}}`)); err == nil {
		t.Error("the same address in two spellings should be a duplicate row")
	}
}

func TestBrokerRowsWithoutPublishedIdentity(t *testing.T) {
	// A self-hosted (decentralized) broker publishes no identity: the router stores
	// "" for it, so its rules are keyed by address (and "*" families).
	rs := mustParse(t, bundle(1,
		`{"identity":"*","models":"qwen3","params":{"stop":"supported","n":"rejected"}}`,
		`{"model":"qwen3","address":"0xAB000000000000000000000000000000000000AB","params":{"stop":"rejected"}},
		 {"identity":"Aliyun","model":"qwen3","params":{"n":"supported"}}`))
	if got := rs.Lookup(RowKey{Model: "qwen3", Address: "0xab000000000000000000000000000000000000ab"}, EndpointChat, "stop", ClassPresent); got != rejected {
		t.Errorf("address-only row: %+v", got)
	}
	if got := rs.Lookup(RowKey{Model: "qwen3", Address: "0xcd000000000000000000000000000000000000cd"}, EndpointChat, "stop", ClassPresent); got != supported {
		t.Errorf("other broker gets the family default: %+v", got)
	}
	// Identity compares case-insensitively.
	if got := rs.Lookup(RowKey{Identity: "aliyun", Model: "qwen3"}, EndpointChat, "n", ClassPresent); got != supported {
		t.Errorf("identity case: %+v", got)
	}
	// An empty identity never picks up an identity family or an identity row.
	if got := rs.Lookup(RowKey{Model: "qwen3"}, EndpointChat, "n", ClassPresent); got != rejected {
		t.Errorf("empty identity: %+v", got)
	}
}

func TestFreeTextAllowsOrdinaryWriting(t *testing.T) {
	mustParse(t, bundle(1, "", `{"identity":"a","model":"q","note":"探测 2026-11-19：json_schema 被忽略 (staging)","source":{"stop":"provider sheet, v2"},"params":{"stop":"rejected"}}`))
}

// TestCatalogCoversDesignSection3 pins the catalog to the parameter table of
// the design (section 3): every parameter it names, in its category.
func TestCatalogCoversDesignSection3(t *testing.T) {
	want := map[Category][]string{
		CategoryTranslate:   {"reasoning_effort", "reasoning.effort", "thinking.type", "thinking.budget_tokens", "thinking.display", "output_config.effort", "max_tokens", "max_completion_tokens", "max_output_tokens", "enable_thinking", "chat_template_kwargs.enable_thinking", "chat_template_kwargs.reasoning_effort", "chat_template_kwargs.thinking", "chat_template_kwargs.thinking_budget", "thinking_budget"},
		CategoryReject:      {"tools", "tool_choice", "response_format", "text.format", "stop", "stop_sequences", "logprobs", "top_logprobs", "n", "service_tier", "context_management", "output_config.format", "functions", "function_call", "web_search_options", "enable_search", "search_options", "mcp_servers", "container", "reasoning_split"},
		CategoryDrop:        {"temperature", "top_p", "top_k", "frequency_penalty", "presence_penalty", "seed", "parallel_tool_calls", "include", "reasoning.summary", "truncation", "text.verbosity", "repetition_penalty", "min_p", "logit_bias", "chat_template_kwargs.clear_thinking", "chat_template_kwargs.preserve_thinking", "verbosity", "thinking.clear_thinking", "reasoning.generate_summary", "preserve_thinking"},
		CategoryPassthrough: {"user", "store", "metadata", "prompt_cache_key", "safety_identifier", "cache_control"},
	}
	n := 0
	for cat, params := range want {
		for _, p := range params {
			n++
			if got, ok := CategoryOf(p); !ok || got != cat {
				t.Errorf("%s: category %q (known %v), want %q", p, got, ok, cat)
			}
		}
	}
	if n != len(catalog) {
		t.Errorf("catalog has %d parameters, the design table %d: update both together", len(catalog), n)
	}
	for _, c := range []string{"custom", "local_shell", "hosted"} {
		if !slices.Contains(ValueClasses("tools"), c) {
			t.Errorf("tools: no class %q", c)
		}
	}
}

func TestCategoryOfValue(t *testing.T) {
	cases := []struct {
		param, class string
		want         Category
		ok           bool
	}{
		{"reasoning_effort", "high", CategoryTranslate, true},
		{"reasoning_effort", "none", CategoryReject, true}, // thinking off is avoided (D9)
		{"reasoning.effort", "none", CategoryReject, true},
		{"thinking.type", "disabled", CategoryReject, true},
		{"thinking.type", "enabled", CategoryTranslate, true},
		{"tool_choice", "required", CategoryReject, true},
		{"tool_choice", "none", CategoryReject, true},
		{"tool_choice", "auto", CategoryDrop, true},
		{"stop", ClassPresent, CategoryReject, true},
		{"reasoning_effort", "High", "", false},
		{"user", ClassPresent, CategoryPassthrough, true},
		{"metadata", "anything", CategoryPassthrough, true},
		{"stream", ClassPresent, CategoryStructural, true},
		{"messages", ClassPresent, CategoryStructural, true},
		{"reasoning_effort", "minimal", CategoryTranslate, true}, // the lowest level, not thinking off
		{"thinking.display", "updates", CategoryDrop, true},
		{"thinking.display", "omitted", CategoryReject, true},
		{"thinking", "anything", CategoryContainer, true},
		{"reasoning", "anything", CategoryContainer, true},
		{"bogus", ClassPresent, "", false},
	}
	for _, c := range cases {
		got, ok := CategoryOfValue(c.param, c.class)
		if got != c.want || ok != c.ok {
			t.Errorf("%s/%s: %q %v, want %q %v", c.param, c.class, got, ok, c.want, c.ok)
		}
	}
	if cat, ok := CategoryOf("metadata"); !ok || cat != CategoryPassthrough {
		t.Errorf("metadata: %q %v", cat, ok)
	}
}

func TestRulesMayNotNamePassthroughParameters(t *testing.T) {
	for _, p := range []string{"user", "store", "metadata", "prompt_cache_key"} {
		b := bundle(1, "", `{"identity":"a","model":"m","params":{"`+p+`":"rejected"}}`)
		if _, err := ParseRules(b); err == nil || !strings.Contains(err.Error(), "passed through") {
			t.Errorf("%s: err %v", p, err)
		}
	}
}

func TestNewParametersParse(t *testing.T) {
	rs := mustParse(t, bundle(1, "", `{"identity":"a","model":"m","params":{
		"stop_sequences":"rejected",
		"text.format":{"json_schema_strict":"ignored"},
		"include":"ignored","reasoning.summary":"ignored","truncation":"rejected","text.verbosity":"ignored",
		"reasoning.effort":{"none":"translate:enable_thinking","minimal":"translate:level:low"},
		"thinking.type":{"disabled":"translate:enable_thinking","adaptive":"translate:reasoning_effort"},
		"thinking.display":{"summarized":"ignored"},
		"thinking.budget_tokens":{"present":"translate:reasoning_effort","limits":{"min":1024}},
		"max_tokens":{"limits":{"max":8192}},"max_completion_tokens":"translate:max_tokens","output_config.effort":{"max":"translate:level:high"}}}`))
	row := RowKey{Identity: "a", Model: "m"}
	if got := rs.Lookup(row, EndpointChat, "stop_sequences", ClassPresent); got != rejected {
		t.Errorf("stop_sequences: %+v", got)
	}
	if l, ok := rs.Limits(row, EndpointChat, "max_completion_tokens"); !ok || l.Max != 8192 {
		t.Errorf("max_completion_tokens limits: %+v %v", l, ok)
	}
	if l, ok := rs.Limits(row, EndpointChat, "thinking.budget_tokens"); !ok || l.Min != 1024 {
		t.Errorf("budget limits: %+v %v", l, ok)
	}
}

func TestFamilyOrderDoesNotMatter(t *testing.T) {
	specific := `{"identity":"up","models":"qwen3-coder*","params":{"tools":{"function":"rejected"},"max_tokens":{"limits":{"max":100}}}}`
	broad := `{"identity":"up","models":"*","params":{"tools":{"function":"supported"},"max_tokens":{"limits":{"max":500,"min":2}}}}`
	for _, fams := range []string{specific + "," + broad, broad + "," + specific} {
		rs := mustParse(t, bundle(1, fams, ""))
		row := RowKey{Identity: "up", Model: "qwen3-coder-x"}
		if got := rs.Lookup(row, EndpointChat, "tools", "function"); got != rejected {
			t.Errorf("%s: the more specific family must decide, got %+v", fams, got)
		}
		if l, _ := rs.Limits(row, EndpointChat, "max_tokens"); l.Max != 100 || l.Min != 2 {
			t.Errorf("limits bound by bound from the most specific setter: %+v", l)
		}
		if got := rs.Lookup(RowKey{Identity: "up", Model: "glm"}, EndpointChat, "tools", "function"); got != supported {
			t.Errorf("broad family elsewhere: %+v", got)
		}
	}
	// Equally specific families that disagree: unknown, whatever the order.
	a := `{"identity":"up","models":"qwen*","params":{"stop":"supported","max_tokens":{"limits":{"max":100}}}}`
	b := `{"identity":"up","models":"*en-x","params":{"stop":"rejected","max_tokens":{"limits":{"max":50}}}}`
	for _, fams := range []string{a + "," + b, b + "," + a} {
		rs := mustParse(t, bundle(1, fams, ""))
		row := RowKey{Identity: "up", Model: "qwen-x"}
		if got := rs.Lookup(row, EndpointChat, "stop", ClassPresent); got != unknown {
			t.Errorf("tie: %+v", got)
		}
		if l, _ := rs.Limits(row, EndpointChat, "max_tokens"); l.Max != 50 {
			t.Errorf("tie limits intersect: %+v", l)
		}
	}
}

func TestLookupKeysAreASCII(t *testing.T) {
	rs := mustParse(t, bundle(1, "", `{"identity":"kimi","model":"m","params":{"stop":"supported","max_tokens":{"limits":{"max":10}}}}`))
	kelvin := RowKey{Identity: "\u212aimi", Model: "m"} // KELVIN SIGN folds to k under Unicode rules
	if got := rs.Lookup(kelvin, EndpointChat, "stop", ClassPresent); got != unknown {
		t.Errorf("a look-alike identity matched: %+v", got)
	}
	if _, ok := rs.Limits(kelvin, EndpointChat, "max_tokens"); ok {
		t.Error("a look-alike identity got limits")
	}
	if got := rs.Lookup(RowKey{Identity: "KIMI", Model: "m"}, EndpointChat, "stop", ClassPresent); got != supported {
		t.Errorf("ASCII case folding: %+v", got)
	}
}

func TestParseRulesRefusesEscapes(t *testing.T) {
	for _, b := range [][]byte{
		[]byte(`{"schema":1,"version":1,"env":"test","families":[],"rows":[{"identity":"a","model":"qwen\\/q3","params":{"stop":"supported"}}]}`),
		[]byte(`{"schema":1,"version":1,"env":"test","families":[],"rows":[{"identity":"a","model":"m","params":{"\u0073top":"supported"}}]}`),
		[]byte(`{"schema":1,"version":1,"env":"test","families":[],"rows":[{"identity":"a","model":"m","params":{"stop":"\u0073upported"}}]}`),
	} {
		if _, err := ParseRules(b); err == nil {
			t.Errorf("accepted %s", b)
		}
	}
}

func TestBundleEnv(t *testing.T) {
	envBundle := func(v int, env string) []byte {
		return []byte(fmt.Sprintf(`{"schema":1,"version":%d,"env":%q,"families":[],"rows":[]}`, v, env))
	}
	if _, err := ParseRules([]byte(`{"schema":1,"version":1,"families":[],"rows":[]}`)); err == nil {
		t.Error("a bundle without env loaded")
	}
	s, err := NewStore(BuiltinRules(), StoreOptions{Env: "mainnet"})
	if err != nil {
		t.Fatal(err)
	}
	good := fakeVerifier{signer: "0x1"}
	if err := s.Load(envBundle(2, "staging"), nil, good, "0x1"); !errors.Is(err, ErrWrongEnv) {
		t.Errorf("staging bundle in mainnet: %v", err)
	}
	if err := s.Load(envBundle(2, EnvAny), nil, good, "0x1"); !errors.Is(err, ErrWrongEnv) {
		t.Errorf("a loaded bundle may not claim every environment: %v", err)
	}
	if err := s.Load(envBundle(2, "MainNet"), nil, good, "0x1"); err != nil {
		t.Errorf("env compares case-insensitively: %v", err)
	}
	for _, env := range []string{"staging", "mainnet"} {
		// An initial bundle skips the signature and version checks: only
		// compiled-in data (env any) may be one, even for the store's own env.
		if _, err := NewStore(envBundle(9, env), StoreOptions{Env: "mainnet"}); !errors.Is(err, ErrWrongEnv) {
			t.Errorf("initial bundle for %s: %v", env, err)
		}
	}
	for _, env := range []string{"", EnvAny} {
		if _, err := NewStore(BuiltinRules(), StoreOptions{Env: env}); err == nil {
			t.Errorf("store env %q accepted", env)
		}
	}
}

func TestInitialBundleNeverBlocksALoad(t *testing.T) {
	// A library release whose built-in rules carry version 10 must not lock out
	// the signed version 8 operators already deliver.
	initial := []byte(`{"schema":1,"version":10,"env":"any","families":[],"rows":[]}`)
	s, err := NewStore(initial, StoreOptions{Env: "test"})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Load(bundle(8, "", ""), nil, fakeVerifier{signer: "0x1"}, "0x1"); err != nil {
		t.Fatalf("signed bundle below the built-in version: %v", err)
	}
	if err := s.LoadUnsigned(bundle(8, "", "")); !errors.Is(err, ErrStaleRules) {
		t.Fatalf("after a load, versions must still rise: %v", err)
	}
}

func TestInitialBundleBelowFloorIsNotServed(t *testing.T) {
	// Signed v5 retired a rule and the floor was raised to 5; after a restart the
	// built-in v1 must not come back.
	s, err := NewStore(BuiltinRules(), StoreOptions{Env: "test", MinVersion: 5})
	if err != nil {
		t.Fatal(err)
	}
	if s.Rules() != nil {
		t.Fatalf("built-in v%d served below the floor", s.Rules().Version())
	}
	if got := s.Rules().Lookup(RowKey{Identity: "a", Model: "m"}, EndpointChat, "stop", ClassPresent); got != unknown {
		t.Fatalf("no rules must answer unknown: %+v", got)
	}
	if err := s.Load(bundle(5, "", ""), nil, fakeVerifier{signer: "0x1"}, "0x1"); err != nil || s.Rules().Version() != 5 {
		t.Fatalf("bundle at the floor: %v", err)
	}
	// At or above the floor, the initial bundle is served.
	s2, err := NewStore([]byte(`{"schema":1,"version":5,"env":"any","families":[],"rows":[]}`), StoreOptions{Env: "test", MinVersion: 5})
	if err != nil || s2.Rules().Version() != 5 {
		t.Fatalf("initial at the floor: %v", err)
	}
}

func TestNilAndZeroStore(t *testing.T) {
	var nilStore *Store
	if nilStore.Rules() != nil {
		t.Error("nil store serves rules")
	}
	if err := nilStore.LoadUnsigned(bundle(1, "", "")); err == nil {
		t.Error("nil store accepted a load")
	}
	var zero Store
	if zero.Rules() != nil {
		t.Error("zero store serves rules")
	}
	if err := zero.LoadUnsigned(bundle(1, "", "")); err == nil {
		t.Error("zero store accepted a load")
	}
}

type mutatingVerifier struct{}

// Verify approves, then scribbles over the buffer it was given.
func (mutatingVerifier) Verify(b, _ []byte) (string, error) {
	copy(b, bytes.Repeat([]byte{' '}, len(b)))
	return "0x1", nil
}

func TestLoadParsesTheBytesItVerified(t *testing.T) {
	s, err := NewStore(BuiltinRules(), StoreOptions{Env: "test"})
	if err != nil {
		t.Fatal(err)
	}
	b := bundle(2, "", `{"identity":"a","model":"m","params":{"stop":"supported"}}`)
	orig := bytes.Clone(b)
	if err := s.Load(b, nil, mutatingVerifier{}, "0x1"); err != nil {
		t.Fatalf("load: %v", err)
	}
	if !bytes.Equal(b, orig) {
		t.Fatal("Load let the verifier change the caller's buffer")
	}
	if s.Rules().Lookup(RowKey{Identity: "a", Model: "m"}, EndpointChat, "stop", ClassPresent) != supported {
		t.Fatal("rules in force are not the verified ones")
	}
}

func TestAddressLayerOverridesIntersectedSites(t *testing.T) {
	rs := mustParse(t, bundle(1, "", `{"identity":"a","model":"m","site":"s1","params":{"max_tokens":{"limits":{"max":100}}}},
		{"identity":"a","model":"m","site":"s2","params":{"max_tokens":{"limits":{"max":200}}}},
		{"identity":"a","model":"m","address":"0xab000000000000000000000000000000000000ab","params":{"max_tokens":{"limits":{"max":500}}}}`))
	if l, _ := rs.Limits(RowKey{Identity: "a", Model: "m"}, EndpointChat, "max_tokens"); l.Max != 100 {
		t.Errorf("sites intersect: %+v", l)
	}
	if l, _ := rs.Limits(RowKey{Identity: "a", Model: "m", Address: "0xab000000000000000000000000000000000000ab"}, EndpointChat, "max_tokens"); l.Max != 500 {
		t.Errorf("the address layer overrides: %+v", l)
	}
}

// TestThinkingOffNeverThinks: D9 in every spelling. Each class that turns
// thinking off is avoided when unsupported and can only become a native off
// switch, never a level or an effort that thinks.
func TestThinkingOffNeverThinks(t *testing.T) {
	off := []struct{ param, class string }{
		{"reasoning_effort", "none"},
		{"reasoning.effort", "none"},
		{"thinking.type", "disabled"},
		{"thinking.type", "between_tools"},
		{"enable_thinking", "false"},
		{"chat_template_kwargs.enable_thinking", "false"},
		{"chat_template_kwargs.reasoning_effort", "none"},
	}
	for _, o := range off {
		if c, _ := CategoryOfValue(o.param, o.class); c != CategoryReject {
			t.Errorf("%s %s: category %q, want reject", o.param, o.class, c)
		}
		for _, tr := range catalog[o.param].translationsFor(o.class) {
			if strings.HasPrefix(tr, "level:") || tr == "output_config.effort" || tr == "thinking.budget_tokens" {
				t.Errorf("%s %s may become %q", o.param, o.class, tr)
			}
		}
		for _, l := range ReasoningEffortLadder {
			b := bundle(1, "", fmt.Sprintf(`{"identity":"a","model":"m","params":{%q:{%q:"translate:level:%s"}}}`, o.param, o.class, l))
			if _, err := ParseRules(b); err == nil {
				t.Errorf("%s %s translated to level %s", o.param, o.class, l)
			}
		}
	}
	// Every other class of these parameters thinks ("minimal" included).
	for _, p := range []string{"reasoning_effort", "reasoning.effort", "thinking.type", "enable_thinking", "chat_template_kwargs.enable_thinking", "chat_template_kwargs.reasoning_effort"} {
		for _, c := range ValueClasses(p) {
			if c == "none" || c == "disabled" || c == "between_tools" || c == "false" {
				continue
			}
			if cat, _ := CategoryOfValue(p, c); cat != CategoryTranslate {
				t.Errorf("%s %s: %q", p, c, cat)
			}
		}
	}
}

func TestCatalogCannotBeChangedFromOutside(t *testing.T) {
	saved := ReasoningEffortLadder[0]
	ReasoningEffortLadder[0] = "messages"
	defer func() { ReasoningEffortLadder[0] = saved }()
	if slices.Contains(ValueClasses("output_config.effort"), "messages") {
		t.Fatal("the catalog shares the exported ladder")
	}
	vc := ValueClasses("tool_choice")
	vc[0] = "x"
	if ValueClasses("tool_choice")[0] == "x" {
		t.Fatal("ValueClasses returned the catalog's own slice")
	}
}

func TestDuplicatesInAnySpelling(t *testing.T) {
	cases := map[string][]byte{
		"site case":   bundle(1, "", `{"identity":"a","model":"m","site":"X","params":{"stop":"supported"}},{"identity":"a","model":"m","site":"x","params":{"stop":"rejected"}}`),
		"family":      bundle(1, `{"identity":"a","models":"q*","params":{"stop":"supported"}},{"identity":"a","models":"q*","params":{"stop":"rejected"}}`, ""),
		"family case": bundle(1, `{"identity":"A","models":"q*","params":{"stop":"supported"}},{"identity":"a","models":"q*","params":{"n":"rejected"}}`, ""),
	}
	for name, b := range cases {
		if _, err := ParseRules(b); err == nil || !strings.Contains(err.Error(), "duplicate") {
			t.Errorf("%s: err %v", name, err)
		}
	}
}

func TestAddressRuleNeedsThePublishedIdentity(t *testing.T) {
	// A row that publishes identity "a" is keyed under it: an address rule
	// written without the identity applies only to rows that publish none.
	rs := mustParse(t, bundle(1, "", `{"model":"m","address":"0xab000000000000000000000000000000000000ab","params":{"stop":"rejected"}}`))
	if got := rs.Lookup(RowKey{Identity: "a", Model: "m", Address: "0xab000000000000000000000000000000000000ab"}, EndpointChat, "stop", ClassPresent); got != unknown {
		t.Errorf("identity row picked up an identity-less address rule: %+v", got)
	}
	if got := rs.Lookup(RowKey{Model: "m", Address: "0xab000000000000000000000000000000000000ab"}, EndpointChat, "stop", ClassPresent); got != rejected {
		t.Errorf("identity-less row: %+v", got)
	}
}

func TestExactPatternBeatsWildcard(t *testing.T) {
	for _, fams := range []string{
		`{"identity":"up","models":"gpt-5.1*","params":{"stop":"rejected"}},{"identity":"up","models":"gpt-5.1","params":{"stop":"supported"}},{"identity":"up","models":"*gpt-5.1*","params":{"stop":"ignored"}}`,
		`{"identity":"up","models":"*gpt-5.1*","params":{"stop":"ignored"}},{"identity":"up","models":"gpt-5.1","params":{"stop":"supported"}},{"identity":"up","models":"gpt-5.1*","params":{"stop":"rejected"}}`,
	} {
		rs := mustParse(t, bundle(1, fams, ""))
		if got := rs.Lookup(RowKey{Identity: "up", Model: "gpt-5.1"}, EndpointChat, "stop", ClassPresent); got != supported {
			t.Errorf("exact id: %+v", got)
		}
		if got := rs.Lookup(RowKey{Identity: "up", Model: "gpt-5.1-mini"}, EndpointChat, "stop", ClassPresent); got != rejected {
			t.Errorf("prefix beats infix: %+v", got)
		}
		if got := rs.Lookup(RowKey{Identity: "up", Model: "x-gpt-5.1"}, EndpointChat, "stop", ClassPresent); got != ignored {
			t.Errorf("infix only: %+v", got)
		}
	}
	// A lone "*" family still applies (and alone in its group, decides).
	rs := mustParse(t, bundle(1, `{"identity":"*","models":"*","params":{"stop":"supported","max_tokens":{"limits":{"min":3,"max":9}}}}`, ""))
	if got := rs.Lookup(RowKey{Identity: "x", Model: "m"}, EndpointChat, "stop", ClassPresent); got != supported {
		t.Errorf("lone *: %+v", got)
	}
	if l, ok := rs.Limits(RowKey{Identity: "x", Model: "m"}, EndpointChat, "max_tokens"); !ok || l.Min != 3 || l.Max != 9 {
		t.Errorf("lone * limits: %+v %v", l, ok)
	}
}

func TestEmptyModelMatchesNothing(t *testing.T) {
	rs := mustParse(t, bundle(1, `{"identity":"*","models":"*","params":{"stop":"supported","max_tokens":{"limits":{"max":9}}}}`, ""))
	if got := rs.Lookup(RowKey{Identity: "x"}, EndpointChat, "stop", ClassPresent); got != unknown {
		t.Errorf("empty model: %+v", got)
	}
	if _, ok := rs.Limits(RowKey{Identity: "x"}, EndpointChat, "max_tokens"); ok {
		t.Error("empty model got limits")
	}
}

func TestContainers(t *testing.T) {
	for _, c := range []string{"reasoning", "text", "output_config", "thinking", "chat_template_kwargs"} {
		if !IsContainer(c) {
			t.Errorf("%s is not a container", c)
		}
	}
	// Every dotted catalog name lives under a container.
	for name := range catalog {
		if i := strings.IndexByte(name, '.'); i > 0 && !IsContainer(name[:i]) {
			t.Errorf("%s: %s is not a container", name, name[:i])
		}
	}
	if IsContainer("response_format") || IsContainer("metadata") {
		t.Error("judged-as-a-whole fields reported as containers")
	}
}

func TestNotesMayUseFullWidthSpace(t *testing.T) {
	mustParse(t, bundle(1, "", `{"identity":"a","model":"m","note":"中文　说明","params":{"stop":"supported"}}`))
}

// TestWalkRealBodies walks request bodies the way Plan will: containers are
// descended into, every other key judged as itself. Nothing a real client
// sends may come out unknown.
func TestWalkRealBodies(t *testing.T) {
	bodies := map[string]string{
		"anthropic": `{"model":"m","messages":[],"max_tokens":1024,"stream":true,"system":"s","stop_sequences":["x"],
			"thinking":{"type":"enabled","budget_tokens":2048,"display":"summarized"},"output_config":{"effort":"high"},
			"tools":[],"tool_choice":{"type":"auto"},"metadata":{"user_id":"u"},"service_tier":"auto","temperature":1}`,
		"responses": `{"model":"m","input":"hi","instructions":"i","stream":false,"store":false,"include":["reasoning.encrypted_content"],
			"reasoning":{"effort":"low","summary":"auto"},"text":{"format":{"type":"text"},"verbosity":"low"},
			"truncation":"auto","max_output_tokens":100,"prompt_cache_key":"k","parallel_tool_calls":true,"user":"u"}`,
		"chat": `{"model":"m","messages":[],"stream":true,"stream_options":{"include_usage":true},"max_completion_tokens":10,
			"reasoning_effort":"minimal","response_format":{"type":"json_object"},"logprobs":true,"top_logprobs":2,"n":1,
			"seed":1,"top_k":5,"enable_thinking":false,"chat_template_kwargs":{"enable_thinking":true,"reasoning_effort":"high"}}`,
	}
	for name, body := range bodies {
		var m map[string]json.RawMessage
		if err := json.Unmarshal([]byte(body), &m); err != nil {
			t.Fatal(name, err)
		}
		for k, v := range m {
			if cat, ok := CategoryOf(k); !ok {
				t.Errorf("%s: %s unknown", name, k)
			} else if cat == CategoryContainer {
				var inner map[string]json.RawMessage
				if err := json.Unmarshal(v, &inner); err != nil {
					t.Fatalf("%s: %s: %v", name, k, err)
				}
				for ik := range inner {
					if _, ok := CategoryOf(k + "." + ik); !ok {
						t.Errorf("%s: %s.%s unknown", name, k, ik)
					}
				}
			}
		}
	}
}

func TestOneOutputCapPerRow(t *testing.T) {
	rs := mustParse(t, bundle(1, "", `{"identity":"a","model":"m","params":{"max_tokens":{"limits":{"min":16,"max":8192}}}}`))
	row := RowKey{Identity: "a", Model: "m"}
	for _, p := range []string{"max_tokens", "max_completion_tokens", "max_output_tokens"} {
		if l, ok := rs.Limits(row, EndpointChat, p); !ok || l.Max != 8192 || l.Min != 16 {
			t.Errorf("%s: %+v %v", p, l, ok)
		}
	}
	for _, p := range []string{"max_completion_tokens", "max_output_tokens"} {
		b := bundle(1, "", fmt.Sprintf(`{"identity":"a","model":"m","params":{%q:{"present":"supported","limits":{"max":5}}}}`, p))
		if _, err := ParseRules(b); err == nil {
			t.Errorf("%s carries its own limits", p)
		}
	}
}

func TestRepeatedStarsRefused(t *testing.T) {
	for _, pat := range []string{"gpt**", "**", "a**b"} {
		if _, err := ParseRules(bundle(1, fmt.Sprintf(`{"identity":"*","models":%q,"params":{"stop":"supported"}}`, pat), "")); err == nil {
			t.Errorf("%s accepted", pat)
		}
	}
}

func TestNoSelfTranslation(t *testing.T) {
	for name, spec := range catalog {
		for _, c := range spec.classes {
			for _, tr := range spec.translationsFor(c) {
				if tr == name {
					t.Errorf("%s %s may translate to itself", name, c)
				}
			}
		}
	}
}

func TestEmptyRulesStartsASecondStore(t *testing.T) {
	s, err := NewStore(EmptyRules(), StoreOptions{Env: "test"})
	if err != nil {
		t.Fatal(err)
	}
	if got := s.Rules().Lookup(RowKey{Identity: "a", Model: "m"}, EndpointChat, "stop", ClassPresent); got != unknown {
		t.Errorf("%+v", got)
	}
	if err := s.LoadUnsigned(bundle(1, "", "")); err != nil {
		t.Errorf("first load after EmptyRules: %v", err)
	}
}

func TestGLMOffSwitchTranslation(t *testing.T) {
	// A GLM row with the template's enable_thinking switch but no
	// reasoning_effort translates none to that switch instead of being avoided.
	mustParse(t, bundle(1, "", `{"identity":"a","model":"glm","params":{"chat_template_kwargs.reasoning_effort":{"none":"translate:chat_template_kwargs"}}}`))
}

// TestBundleChecksEachOnTheirOwn: every bundle-level check, with everything
// else valid, so each case fails for the reason it names.
func TestBundleChecksEachOnTheirOwn(t *testing.T) {
	cases := map[string]string{
		"wrong schema":           `{"schema":2,"version":1,"env":"test","families":[],"rows":[]}`,
		"version zero":           `{"schema":1,"version":0,"env":"test","families":[],"rows":[]}`,
		"upper-case key beside":  `{"schema":1,"Version":9,"version":1,"env":"test","families":[],"rows":[]}`,
		"line separator in note": "{\"schema\":1,\"version\":1,\"env\":\"test\",\"note\":\"a\u2028b\",\"families\":[],\"rows\":[]}",
		"private use in note":    "{\"schema\":1,\"version\":1,\"env\":\"test\",\"note\":\"a\ue000b\",\"families\":[],\"rows\":[]}",
		"replacement char note":  "{\"schema\":1,\"version\":1,\"env\":\"test\",\"note\":\"a\ufffdb\",\"families\":[],\"rows\":[]}",
		"invalid utf-8 in note":  "{\"schema\":1,\"version\":1,\"env\":\"test\",\"note\":\"a\xffb\",\"families\":[],\"rows\":[]}",
		"address without 0x":     `{"schema":1,"version":1,"env":"test","families":[],"rows":[{"model":"m","address":"1xab000000000000000000000000000000000000ab","params":{"stop":"supported"}}]}`,
		"too many sites":         `{"schema":1,"version":1,"env":"test","families":[],"rows":[` + sites(maxSitesPerRow+1) + `]}`,
	}
	for name, b := range cases {
		if _, err := ParseRules([]byte(b)); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	if _, err := ParseRules([]byte(`{"schema":1,"version":1,"env":"test","families":[],"rows":[` + sites(maxSitesPerRow) + `]}`)); err != nil {
		t.Errorf("sites at the cap: %v", err)
	}
	big := append([]byte(`{"schema":1,"version":1,"env":"test","note":"`), bytes.Repeat([]byte("a"), maxBundleBytes)...)
	if _, err := ParseRules(append(big, []byte(`","families":[],"rows":[]}`)...)); err == nil || !strings.Contains(err.Error(), "bytes") {
		t.Errorf("oversized bundle: %v", err)
	}
}

func sites(n int) string {
	var parts []string
	for i := 0; i < n; i++ {
		parts = append(parts, fmt.Sprintf(`{"identity":"a","model":"m","site":"s%d","params":{"stop":"supported"}}`, i))
	}
	return strings.Join(parts, ",")
}

func TestTooManyFamiliesForOneIdentity(t *testing.T) {
	var fams []string
	for i := 0; i <= maxFamilies; i++ {
		fams = append(fams, fmt.Sprintf(`{"identity":"a","models":"m%d*","params":{"stop":"supported"}}`, i))
	}
	if _, err := ParseRules(bundle(1, strings.Join(fams, ","), "")); err == nil {
		t.Error("accepted more families than the cap")
	}
}

func TestFamilyMinTieIntersects(t *testing.T) {
	rs := mustParse(t, bundle(1, `{"identity":"a","models":"q*","params":{"max_tokens":{"limits":{"min":5}}}},{"identity":"a","models":"*x","params":{"max_tokens":{"limits":{"min":7}}}}`, ""))
	if l, _ := rs.Limits(RowKey{Identity: "a", Model: "qx"}, EndpointChat, "max_tokens"); l.Min != 7 {
		t.Errorf("tie on min must take the higher: %+v", l)
	}
}

type signerAndError struct{}

func (signerAndError) Verify(_, _ []byte) (string, error) { return "0x1", errors.New("bad signature") }

func TestLoadRefusesAVerifierError(t *testing.T) {
	s, err := NewStore(BuiltinRules(), StoreOptions{Env: "test"})
	if err != nil {
		t.Fatal(err)
	}
	// A verifier that names the right signer but reports an error is refused.
	if err := s.Load(bundle(2, "", ""), nil, signerAndError{}, "0x1"); err == nil {
		t.Fatal("accepted")
	}
}

func TestStoreEnvIsCaseInsensitive(t *testing.T) {
	s, err := NewStore(BuiltinRules(), StoreOptions{Env: "Test"})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.LoadUnsigned(bundle(2, "", "")); err != nil {
		t.Fatalf("store env Test, bundle env test: %v", err)
	}
}

func TestFamiliesOnlyApplyToTheirIdentity(t *testing.T) {
	rs := mustParse(t, bundle(1, `{"identity":"a","models":"*","params":{"stop":"supported"}},{"identity":"*","models":"*","params":{"n":"rejected"}}`, ""))
	if got := rs.Lookup(RowKey{Identity: "b", Model: "m"}, EndpointChat, "stop", ClassPresent); got != unknown {
		t.Errorf("identity a's family applied to b: %+v", got)
	}
	if got := rs.Lookup(RowKey{Identity: "b", Model: "m"}, EndpointChat, "n", ClassPresent); got != rejected {
		t.Errorf("* family: %+v", got)
	}
	if got := rs.Lookup(RowKey{Identity: "*", Model: "m"}, EndpointChat, "n", ClassPresent); got != rejected {
		t.Errorf("identity * as a key: %+v", got)
	}
}

func TestPatternBounds(t *testing.T) {
	for _, pat := range []string{strings.Repeat("a", maxPatternLen+1), "a*b*c*d*e*f"} {
		if _, err := ParseRules(bundle(1, fmt.Sprintf(`{"identity":"a","models":%q,"params":{"stop":"supported"}}`, pat), "")); err == nil {
			t.Errorf("%q accepted", pat)
		}
	}
	mustParse(t, bundle(1, fmt.Sprintf(`{"identity":"a","models":%q,"params":{"stop":"supported"}}`, "*a*b*c*"), ""))
}

// TestLookupCostAtTheCaps: the largest family groups the caps accept, every
// family matching the row, must keep one request's lookups (rows x
// parameters, ~600) within a few tens of milliseconds even under the race
// detector's slowdown.
func TestLookupCostAtTheCaps(t *testing.T) {
	var fams []string
	for _, id := range []string{"*", "a"} {
		for i := 0; i < maxFamilies; i++ {
			pat := fmt.Sprintf("*%s*x*%04d*", strings.Repeat("p", 40), i%10)
			if i >= 10 {
				pat = fmt.Sprintf("*%s*%03d*", strings.Repeat("p", 30+i%10), i)
			}
			fams = append(fams, fmt.Sprintf(`{"identity":%q,"models":%q,"params":{%s,"max_tokens":{"limits":{"max":%d}}}}`, id, pat, everyPresentParam, i+1))
		}
	}
	rs := mustParse(t, bundle(1, strings.Join(fams, ","), ""))
	var model strings.Builder
	model.WriteString(strings.Repeat("p", 40) + "x")
	for i := 0; i < maxFamilies; i++ {
		fmt.Fprintf(&model, "%04d%s%03d", i%10, strings.Repeat("p", 30+i%10), i)
	}
	row := RowKey{Identity: "a", Model: model.String()}
	if n := len(rs.familyGroups(row)[1]); n < maxFamilies/2 {
		t.Fatalf("only %d families match; the test must load the matcher", n)
	}
	// Plan takes one RowRules per candidate row (~10) and asks it for every
	// parameter (~60): 600 lookups.
	const rows, params = 10, 60
	start := time.Now()
	for i := 0; i < rows; i++ {
		rr := rs.ForRow(row, EndpointChat)
		for j := 0; j < params; j++ {
			rr.Lookup("stop", ClassPresent)
			rr.Limits("max_tokens")
		}
	}
	const perRequest = rows * params
	if d := time.Since(start); d > 100*time.Millisecond {
		t.Errorf("one request's lookups at the caps: %v", d)
	}
	t.Logf("%d lookups at the caps: %v", perRequest, time.Since(start))
}

func TestRound11Gaps(t *testing.T) {
	// A family's note is checked like a row's.
	if _, err := ParseRules(bundle(1, "{\"identity\":\"a\",\"models\":\"*\",\"note\":\"x\u202ey\",\"params\":{\"stop\":\"supported\"}}", "")); err == nil {
		t.Error("bidi override in a family note accepted")
	}
	// A family's identity is matched case-insensitively.
	rs := mustParse(t, bundle(1, `{"identity":"Kimi","models":"*","params":{"stop":"supported"}}`, ""))
	if got := rs.Lookup(RowKey{Identity: "kimi", Model: "k2"}, EndpointChat, "stop", ClassPresent); got != supported {
		t.Errorf("mixed-case family identity: %+v", got)
	}
	// A missing middle literal does not match.
	if matchModel("a*b*c", "axc") {
		t.Error("a*b*c matched axc")
	}
	// Both identity groups may hold the cap.
	var fams []string
	for _, id := range []string{"*", "a"} {
		for i := 0; i < maxFamilies; i++ {
			fams = append(fams, fmt.Sprintf(`{"identity":%q,"models":"m%d*","params":{"stop":"supported"}}`, id, i))
		}
	}
	mustParse(t, bundle(1, strings.Join(fams, ","), ""))
	// Rows at and over the cap.
	var rows []string
	for i := 0; i <= maxRows; i++ {
		rows = append(rows, fmt.Sprintf(`{"identity":"a","model":"m%d","params":{"stop":"supported"}}`, i))
	}
	if _, err := ParseRules(bundle(1, "", strings.Join(rows, ","))); err == nil {
		t.Error("rows over the cap accepted")
	}
	mustParse(t, bundle(1, "", strings.Join(rows[:maxRows], ",")))
	// The byte cap, exactly.
	head, tail := `{"schema":1,"version":1,"env":"test","note":"`, `","families":[],"rows":[]}`
	fill := maxBundleBytes - len(head) - len(tail)
	if _, err := ParseRules([]byte(head + strings.Repeat("a", fill) + tail)); err != nil {
		t.Errorf("bundle at the byte cap: %v", err)
	}
	if _, err := ParseRules([]byte(head + strings.Repeat("a", fill+1) + tail)); err == nil {
		t.Error("bundle one byte over the cap accepted")
	}
	// Load with no verifier but a signer is refused, not a panic.
	s, err := NewStore(BuiltinRules(), StoreOptions{Env: "test"})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Load(bundle(2, "", ""), nil, nil, "0x1"); err == nil {
		t.Error("load without a verifier accepted")
	}
	// A non-ASCII model never falls to "*" families.
	rs = mustParse(t, bundle(1, `{"identity":"*","models":"*","params":{"stop":"supported"}}`, ""))
	if got := rs.Lookup(RowKey{Identity: "a", Model: "m\u00e9"}, EndpointChat, "stop", ClassPresent); got != unknown {
		t.Errorf("non-ASCII model: %+v", got)
	}
}

type racingVerifier struct{ buf []byte }

// Verify approves, then changes the caller's buffer, as a concurrent writer could.
func (v racingVerifier) Verify(_, _ []byte) (string, error) {
	copy(v.buf, bytes.Repeat([]byte{' '}, len(v.buf)))
	return "0x1", nil
}

func TestLoadIsolatesTheCallersBuffer(t *testing.T) {
	s, err := NewStore(BuiltinRules(), StoreOptions{Env: "test"})
	if err != nil {
		t.Fatal(err)
	}
	b := bundle(2, "", `{"identity":"a","model":"m","params":{"stop":"supported"}}`)
	if err := s.Load(b, nil, racingVerifier{buf: b}, "0x1"); err != nil {
		t.Fatalf("load: %v", err)
	}
	if s.Rules().Lookup(RowKey{Identity: "a", Model: "m"}, EndpointChat, "stop", ClassPresent) != supported {
		t.Fatal("the bytes parsed were changed after verification")
	}
}

func TestRound12Gaps(t *testing.T) {
	// A pattern of exactly the maximum length loads.
	mustParse(t, bundle(1, fmt.Sprintf(`{"identity":"a","models":%q,"params":{"stop":"supported"}}`, strings.Repeat("a", maxPatternLen)), ""))
	// Sites are checked per endpoint: an every-endpoint entry without a site
	// beside chat entries with sites is fine.
	mustParse(t, bundle(1, "", `{"identity":"a","model":"m","params":{"stop":"supported"}},
		{"identity":"a","model":"m","endpoint":"chat","site":"x","params":{"stop":"supported"}},
		{"identity":"a","model":"m","endpoint":"chat","site":"y","params":{"stop":"supported"}}`))
	for _, f := range []string{"provider", "plugins", "attachments"} {
		if c, _ := CategoryOf(f); c != CategoryStructural {
			t.Errorf("%s: %q", f, c)
		}
	}
	if c, _ := CategoryOfValue("function_call", "auto"); c != CategoryDrop {
		t.Errorf("function_call auto: %q", c)
	}
	// A verifier that returns an empty signer and no error matches no signer.
	s, err := NewStore(BuiltinRules(), StoreOptions{Env: "test"})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Load(bundle(2, "", ""), nil, fakeVerifier{signer: ""}, ""); err == nil {
		t.Error("empty signer accepted")
	}
}

// everyPresentParam names every presence-judged catalog parameter, so cost
// tests merge full families.
var everyPresentParam = func() string {
	var parts []string
	for name, spec := range catalog {
		if len(spec.classes) == 1 && spec.classes[0] == ClassPresent && !spec.limits && spec.category != CategoryPassthrough {
			parts = append(parts, fmt.Sprintf("%q:\"rejected\"", name))
		}
	}
	slices.Sort(parts)
	return strings.Join(parts, ",")
}()

func TestPlainLookupCostAtTheCaps(t *testing.T) {
	var fams []string
	for _, id := range []string{"*", "a"} {
		for i := 0; i < maxFamilies; i++ {
			fams = append(fams, fmt.Sprintf(`{"identity":%q,"models":"*m%03d*","params":{%s}}`, id, i, everyPresentParam))
		}
	}
	rs := mustParse(t, bundle(1, strings.Join(fams, ","), ""))
	var model strings.Builder
	for i := 0; i < maxFamilies; i++ {
		fmt.Fprintf(&model, "m%03d", i)
	}
	row := RowKey{Identity: "a", Model: model.String()}
	start := time.Now()
	for i := 0; i < 600; i++ {
		rs.Lookup(row, EndpointChat, "stop", ClassPresent)
	}
	if d := time.Since(start); d > 300*time.Millisecond {
		t.Errorf("600 plain lookups at the caps: %v", d)
	}
	t.Logf("600 plain lookups at the caps: %v", time.Since(start))
	// Both paths answer alike.
	rr := rs.ForRow(row, EndpointChat)
	for name := range catalog {
		for _, c := range ValueClasses(name) {
			if a, b := rs.Lookup(row, EndpointChat, name, c), rr.Lookup(name, c); a != b {
				t.Errorf("%s %s: Lookup %+v, ForRow %+v", name, c, a, b)
			}
		}
	}
}

func TestRound13Gaps(t *testing.T) {
	// Mixed site-less and sited entries for one endpoint are refused.
	if _, err := ParseRules(bundle(1, "", `{"identity":"a","model":"m","endpoint":"chat","params":{"stop":"supported"}},{"identity":"a","model":"m","endpoint":"chat","site":"x","params":{"stop":"supported"}}`)); err == nil {
		t.Error("mixed sites on one endpoint accepted")
	}
	// Three equally specific families: unknown if any two disagree, in any order.
	three := [][]string{
		{`{"identity":"a","models":"ab*","params":{"stop":"rejected"}}`, `{"identity":"a","models":"*bc","params":{"stop":"rejected"}}`, `{"identity":"a","models":"a*c","params":{"stop":"ignored"}}`},
	}
	for _, f := range three {
		for _, order := range [][3]int{{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}} {
			rs := mustParse(t, bundle(1, f[order[0]]+","+f[order[1]]+","+f[order[2]], ""))
			row := RowKey{Identity: "a", Model: "abc"}
			if got := rs.Lookup(row, EndpointChat, "stop", ClassPresent); got != unknown {
				t.Errorf("order %v: %+v", order, got)
			}
			if got := rs.ForRow(row, EndpointChat).Lookup("stop", ClassPresent); got != unknown {
				t.Errorf("ForRow order %v: %+v", order, got)
			}
		}
	}
	// Min ties intersect in either order.
	for _, fams := range []string{
		`{"identity":"a","models":"q*","params":{"max_tokens":{"limits":{"min":5}}}},{"identity":"a","models":"*x","params":{"max_tokens":{"limits":{"min":7}}}}`,
		`{"identity":"a","models":"*x","params":{"max_tokens":{"limits":{"min":7}}}},{"identity":"a","models":"q*","params":{"max_tokens":{"limits":{"min":5}}}}`,
	} {
		rs := mustParse(t, bundle(1, fams, ""))
		if l, _ := rs.Limits(RowKey{Identity: "a", Model: "qx"}, EndpointChat, "max_tokens"); l.Min != 7 {
			t.Errorf("min tie: %+v", l)
		}
	}
	// An identity family may raise a "*" family's cap: it overrides.
	rs := mustParse(t, bundle(1, `{"identity":"*","models":"*","params":{"max_tokens":{"limits":{"max":5}}}},{"identity":"a","models":"*","params":{"max_tokens":{"limits":{"max":100}}}}`, ""))
	if l, _ := rs.Limits(RowKey{Identity: "a", Model: "m"}, EndpointChat, "max_tokens"); l.Max != 100 {
		t.Errorf("identity family over *: %+v", l)
	}
	// The caps, as numbers.
	if maxFamilies != 200 || maxRows != 50000 || maxSitesPerRow != 16 || maxBundleBytes != 8<<20 || maxPatternLen != 128 || maxPatternStar != 4 {
		t.Error("a cap changed: update the cost tests and the README")
	}
}

func TestEndpointOnRows(t *testing.T) {
	// Baidu VOD's GPT row: function tools refused on chat, accepted on
	// responses; the generator writes one entry per row and endpoint.
	addr := "0x00000000aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	rs := mustParse(t, bundle(1,
		`{"identity":"vod","models":"*","params":{"stop":"supported","max_tokens":{"limits":{"max":100}}}}`,
		fmt.Sprintf(`{"identity":"vod","model":"gpt","params":{"tools":{"function":"supported"}}},
		 {"identity":"vod","model":"gpt","endpoint":"chat","params":{"tools":{"function":"rejected"},"stop":"ignored","max_tokens":{"limits":{"max":50}}}},
		 {"identity":"vod","model":"gpt","address":%q,"params":{"tools":{"function":"ignored"}}},
		 {"identity":"vod","model":"gpt","address":%q,"endpoint":"responses","params":{"tools":{"function":"supported"}}}`, addr, addr)))
	row := RowKey{Identity: "vod", Model: "gpt"}
	cases := []struct {
		key  RowKey
		ep   Endpoint
		tool Status
		stop Status
		max  int
	}{
		{row, EndpointChat, rejected, ignored, 50},
		{row, EndpointResponses, supported, supported, 100},
		{row, EndpointMessages, supported, supported, 100},
		// The address layer comes after the row's endpoint entry; its own
		// endpoint entry comes last.
		{RowKey{Identity: "vod", Model: "gpt", Address: addr}, EndpointChat, ignored, ignored, 50},
		{RowKey{Identity: "vod", Model: "gpt", Address: addr}, EndpointResponses, supported, supported, 100},
	}
	for _, c := range cases {
		rr := rs.ForRow(c.key, c.ep)
		if got := rs.Lookup(c.key, c.ep, "tools", "function"); got != c.tool || rr.Lookup("tools", "function") != c.tool {
			t.Errorf("%v %s tools: %+v", c.key, c.ep, got)
		}
		if got := rs.Lookup(c.key, c.ep, "stop", ClassPresent); got != c.stop {
			t.Errorf("%v %s stop: %+v", c.key, c.ep, got)
		}
		if l, _ := rs.Limits(c.key, c.ep, "max_completion_tokens"); l.Max != c.max {
			t.Errorf("%v %s cap: %+v", c.key, c.ep, l)
		}
	}
	for _, ep := range []Endpoint{"", "Chat", "completions"} {
		if got := rs.Lookup(row, ep, "tools", "function"); got != unknown {
			t.Errorf("endpoint %q: %+v", ep, got)
		}
		if _, ok := rs.Limits(row, ep, "max_tokens"); ok {
			t.Errorf("endpoint %q got limits", ep)
		}
		if got := rs.ForRow(row, ep).Lookup("tools", "function"); got != unknown {
			t.Errorf("ForRow endpoint %q: %+v", ep, got)
		}
	}
}

func TestEndpointValidation(t *testing.T) {
	bad := map[string][]byte{
		"unknown endpoint":         bundle(1, "", `{"identity":"a","model":"m","endpoint":"completions","params":{"stop":"supported"}}`),
		"endpoint case":            bundle(1, "", `{"identity":"a","model":"m","endpoint":"Chat","params":{"stop":"supported"}}`),
		"duplicate":                bundle(1, "", `{"identity":"a","model":"m","endpoint":"chat","params":{"stop":"supported"}},{"identity":"a","model":"m","endpoint":"chat","params":{"n":"rejected"}}`),
		"endpoint on a family":     bundle(1, `{"identity":"a","models":"*","endpoint":"chat","params":{"stop":"rejected"}}`, ""),
		"address on a family":      bundle(1, `{"identity":"a","models":"*","address":"0x00000000aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","params":{"stop":"rejected"}}`, ""),
		"mixed sites per endpoint": bundle(1, "", `{"identity":"a","model":"m","endpoint":"chat","params":{"stop":"supported"}},{"identity":"a","model":"m","endpoint":"chat","site":"x","params":{"stop":"supported"}}`),
	}
	for name, b := range bad {
		if _, err := ParseRules(b); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	// One row for every endpoint and for one is not a duplicate; sites are
	// checked per endpoint.
	mustParse(t, bundle(1, "", `{"identity":"a","model":"m","params":{"stop":"supported"}},
		{"identity":"a","model":"m","endpoint":"chat","site":"x","params":{"stop":"rejected"}},
		{"identity":"a","model":"m","endpoint":"chat","site":"y","params":{"stop":"rejected"}}`))
}

// TestBothPathsAgree: Ruleset.Lookup / Limits and ForRow answer alike on
// random bundles, every catalog value and every endpoint (Plan uses ForRow,
// callers with one question use the Ruleset).
func TestBothPathsAgree(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	statuses := []string{`"supported"`, `"rejected"`, `"ignored"`}
	pats := []string{"*", "m*", "*1", "m1", "*m*", "m*1"}
	idents := []string{"*", "a", "b"}
	params := []string{"stop", "n", "top_k", "logprobs"}
	eps := []string{"", "chat", "messages", "responses"}
	addrs := []string{"", "0x00000000aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}
	for iter := 0; iter < 300; iter++ {
		seenF := map[string]bool{}
		var fams []string
		for i := 0; i < rng.IntN(6); i++ {
			id, pat := idents[rng.IntN(len(idents))], pats[rng.IntN(len(pats))]
			if seenF[id+pat] {
				continue
			}
			seenF[id+pat] = true
			fams = append(fams, fmt.Sprintf(`{"identity":%q,"models":%q,"params":{%q:%s,"max_tokens":{"limits":{"max":%d,"min":%d}}}}`,
				id, pat, params[rng.IntN(len(params))], statuses[rng.IntN(3)], 100+rng.IntN(100), 1+rng.IntN(20)))
		}
		seenR := map[string]bool{}
		var rows []string
		for i := 0; i < rng.IntN(6); i++ {
			ep, ad := eps[rng.IntN(len(eps))], addrs[rng.IntN(len(addrs))]
			if seenR[ep+ad] {
				continue
			}
			seenR[ep+ad] = true
			rows = append(rows, fmt.Sprintf(`{"identity":"a","model":"m1","endpoint":%q,"address":%q,"params":{%q:%s,"max_tokens":{"limits":{"max":%d}}}}`,
				ep, ad, params[rng.IntN(len(params))], statuses[rng.IntN(3)], 50+rng.IntN(200)))
		}
		data := bundle(1, strings.Join(fams, ","), strings.Join(rows, ","))
		data = bytes.ReplaceAll(data, []byte(`"endpoint":"",`), nil)
		data = bytes.ReplaceAll(data, []byte(`"address":"",`), nil)
		rs, err := ParseRules(data)
		if err != nil {
			t.Fatalf("%s: %v", data, err)
		}
		for _, key := range []RowKey{{Identity: "a", Model: "m1"}, {Identity: "a", Model: "m1", Address: addrs[1]}, {Identity: "b", Model: "m1"}, {Model: "m1", Address: addrs[1]}} {
			for _, ep := range []Endpoint{EndpointChat, EndpointMessages, EndpointResponses} {
				rr := rs.ForRow(key, ep)
				for _, p := range params {
					if a, b := rs.Lookup(key, ep, p, ClassPresent), rr.Lookup(p, ClassPresent); a != b {
						t.Fatalf("%v %s %s: Lookup %+v ForRow %+v\n%s", key, ep, p, a, b, data)
					}
				}
				for _, p := range []string{"max_tokens", "max_completion_tokens", "max_output_tokens"} {
					l1, ok1 := rs.Limits(key, ep, p)
					l2, ok2 := rr.Limits(p)
					if l1 != l2 || ok1 != ok2 {
						t.Fatalf("%v %s %s: Limits %+v/%v ForRow %+v/%v", key, ep, p, l1, ok1, l2, ok2)
					}
				}
			}
		}
	}
}

func TestFamilyCapAnswersEverySpelling(t *testing.T) {
	rs := mustParse(t, bundle(1, `{"identity":"a","models":"*","params":{"max_tokens":{"limits":{"max":77,"min":3}}}}`, ""))
	row := RowKey{Identity: "a", Model: "m"}
	for _, p := range []string{"max_tokens", "max_completion_tokens", "max_output_tokens"} {
		if l, ok := rs.Limits(row, EndpointChat, p); !ok || l.Max != 77 || l.Min != 3 {
			t.Errorf("Ruleset %s: %+v %v", p, l, ok)
		}
		if l, ok := rs.ForRow(row, EndpointChat).Limits(p); !ok || l.Max != 77 || l.Min != 3 {
			t.Errorf("ForRow %s: %+v %v", p, l, ok)
		}
	}
	// A family that sets only a max or only a min keeps it on the merged path.
	rs = mustParse(t, bundle(1, `{"identity":"a","models":"m*","params":{"max_tokens":{"limits":{"max":40}}}},{"identity":"*","models":"*","params":{"max_tokens":{"limits":{"min":2}}}}`, ""))
	if l, _ := rs.ForRow(row, EndpointChat).Limits("max_tokens"); l.Max != 40 || l.Min != 2 {
		t.Errorf("max-only and min-only families: %+v", l)
	}
}

func TestRound15Gaps(t *testing.T) {
	// Specificity weighs literal text twice a wildcard: a*b*c beats *bc on abc.
	if familySpecificity("a*b*c") <= familySpecificity("*bc") {
		t.Errorf("a*b*c %d vs *bc %d", familySpecificity("a*b*c"), familySpecificity("*bc"))
	}
	if familySpecificity("abcd*") <= familySpecificity("a*b*c*d*e") {
		t.Error("4 literals and 1 star must beat 5 literals and 4 stars")
	}
	rs := mustParse(t, bundle(1, `{"identity":"a","models":"*bc","params":{"stop":"rejected"}},{"identity":"a","models":"a*b*c","params":{"stop":"supported"}}`, ""))
	if got := rs.Lookup(RowKey{Identity: "a", Model: "abc"}, EndpointChat, "stop", ClassPresent); got != supported {
		t.Errorf("a*b*c over *bc: %+v", got)
	}
	// A row with no address never replays its row layers as address layers:
	// every-endpoint sites {x: supported, y: inherits} then chat sites
	// {x, y: rejected} give rejected, not unknown.
	rs = mustParse(t, bundle(1, "", `{"identity":"a","model":"m","site":"x","params":{"stop":"supported"}},
		{"identity":"a","model":"m","site":"y","params":{"n":"rejected"}},
		{"identity":"a","model":"m","endpoint":"chat","site":"x","params":{"stop":"rejected"}},
		{"identity":"a","model":"m","endpoint":"chat","site":"y","params":{"stop":"rejected"}}`))
	if got := rs.ForRow(RowKey{Identity: "a", Model: "m"}, EndpointChat).Lookup("stop", ClassPresent); got != rejected {
		t.Errorf("row layers replayed: %+v", got)
	}
	// ForRow with an invalid key answers unknown, even with "*" families.
	rs = mustParse(t, bundle(1, `{"identity":"*","models":"*","params":{"stop":"supported","max_tokens":{"limits":{"max":5}}}}`, ""))
	for _, k := range []RowKey{{Identity: "a"}, {Identity: "a", Model: "m\u00e9"}, {Identity: "\u212aimi", Model: "m"}} {
		rr := rs.ForRow(k, EndpointChat)
		if got := rr.Lookup("stop", ClassPresent); got != unknown {
			t.Errorf("ForRow %+v: %+v", k, got)
		}
		if _, ok := rr.Limits("max_tokens"); ok {
			t.Errorf("ForRow %+v got limits", k)
		}
	}
	// "*" is not an identity, site or env value.
	for name, b := range map[string][]byte{
		"row identity": bundle(1, "", `{"identity":"*","model":"m","params":{"stop":"supported"}}`),
		"site":         bundle(1, "", `{"identity":"a","model":"m","site":"*","params":{"stop":"supported"}}`),
		"env":          []byte(`{"schema":1,"version":1,"env":"*","families":[],"rows":[]}`),
	} {
		if _, err := ParseRules(b); err == nil {
			t.Errorf("%s * accepted", name)
		}
	}
}
