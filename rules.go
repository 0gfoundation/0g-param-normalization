package paramnorm

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"unicode"
)

// RulesSchema is the rule-data format version this library reads.
const RulesSchema = 1

// EnvAny, as a bundle's env, marks rules that hold in every environment. Only
// a Store's initial bundle may carry it (the built-in rules); a bundle loaded
// later must name the Store's environment exactly, so a bundle signed for one
// environment (staging) cannot be loaded in another (mainnet).
const EnvAny = "any"

// StatusKind is what a row does with one value of one parameter.
type StatusKind int

const (
	// StatusUnknown: no rule says; callers treat it as rejected.
	StatusUnknown StatusKind = iota
	// StatusSupported: the row serves the value as sent.
	StatusSupported
	// StatusRejected: the row refuses the value (an error from the upstream).
	StatusRejected
	// StatusIgnored: the row accepts the value but it has no effect.
	StatusIgnored
	// StatusTranslate: the row serves the value after the named translation.
	StatusTranslate
)

// Status is a row's handling of one parameter value. Translation names one of
// the code-defined translations for the value when Kind is StatusTranslate.
type Status struct {
	Kind        StatusKind
	Translation string
}

// Limits is a numeric range a row accepts for a parameter; 0 means no bound.
type Limits struct {
	Min int
	Max int
}

// RowKey names a row, in the terms the router stores it with (and the broker
// can derive for itself):
//   - Identity is the upstream identity the row publishes, as the router stores
//     it in provider_identity: for a forwarding broker (centralized or
//     standard), the provider_identity it lists in /models (the model entry's
//     providerIdentity, else the service-level one); for a supply line, the
//     part after '@' in its model key. "" when the row publishes none (a
//     self-hosted broker, a standard broker that sets none, a supply-line key
//     without '@'). Never a derived display identity (the router's "0g" for
//     Foundation brokers).
//   - Model is the row's model id as stored (the broker's model name; for a
//     supply line, the line's model id), not the router's canonical id.
//   - Address is the row's address as stored: the on-chain provider address
//     for a broker, the line's synthetic address for a supply line.
//
// Identity and Address compare case-insensitively, in rule data and lookups
// alike; Model is exact (model ids are case-sensitive). Rules written for an
// address apply only to it, on top of the rules for (Identity, Model), and sit
// under the row's published identity: an address row for a row that publishes
// one must name that identity too. A row that publishes no identity can be
// given rules only by address, or by "*" families. Rule data accepts visible
// ASCII only in these values, and addresses as 0x + 40 hex.
type RowKey struct {
	Identity string
	Model    string
	Address  string
}

// AnyIdentity, as a family's identity, makes it a model-family default that
// holds on every upstream; families naming an identity apply after it.
const AnyIdentity = "*"

// Wire format. Unknown fields are refused, so a typo cannot silently drop a rule.
type rulesFile struct {
	Schema   int          `json:"schema"`
	Version  uint64       `json:"version"`
	Env      string       `json:"env"`
	Note     string       `json:"note,omitempty"`
	Families []familyRule `json:"families"`
	Rows     []rowRule    `json:"rows"`
}

type familyRule struct {
	Identity string                     `json:"identity"` // an upstream identity, or AnyIdentity
	Models   string                     `json:"models"`   // model-id pattern; '*' matches any run of characters
	Params   map[string]json.RawMessage `json:"params"`
	Source   map[string]string          `json:"source,omitempty"`
	Note     string                     `json:"note,omitempty"`
}

type rowRule struct {
	Identity string                     `json:"identity"`
	Model    string                     `json:"model"`
	Address  string                     `json:"address,omitempty"`
	Site     string                     `json:"site,omitempty"`
	Endpoint string                     `json:"endpoint,omitempty"`
	Params   map[string]json.RawMessage `json:"params"`
	Source   map[string]string          `json:"source,omitempty"`
	Note     string                     `json:"note,omitempty"`
}

// limitBounds is the optional numeric range of one parameter; nil fields are
// unset, so a layer can set one bound and keep the other from the layer below.
type limitBounds struct {
	min, max *int
}

// paramRule is one parameter's rule after parsing.
type paramRule struct {
	statuses map[string]Status // value class -> status
	limits   limitBounds
}

type parsedFamily struct {
	identity, pattern string
	parts             []string // pattern split at '*', once, at parse time
	// specificity ranks families (familySpecificity). Among matching families
	// of one group, the most specific one decides.
	specificity int
	params      map[string]paramRule
}

// rowID names the entries of one layer of a row: its rules, or its rules for
// an address, each for every endpoint ("") or for one.
type rowID struct{ identity, model, address, endpoint string }

// Endpoint is the API a request came in on. A row can serve a parameter on one
// endpoint and not another (a supply line's conversion layer: function tools
// refused on chat but accepted on responses), so rule entries may name one.
type Endpoint string

const (
	EndpointChat      Endpoint = "chat"      // /v1/chat/completions
	EndpointMessages  Endpoint = "messages"  // /v1/messages
	EndpointResponses Endpoint = "responses" // /v1/responses
)

func validEndpoint(e Endpoint) bool {
	return e == EndpointChat || e == EndpointMessages || e == EndpointResponses
}

// familySpecificity ranks a family: by its pattern, two points per literal
// character less one per '*' (more literal text wins; at equal text, fewer
// wildcards: an exact model id beats "id*", which beats "*id*").
func familySpecificity(pattern string) int {
	stars := strings.Count(pattern, "*")
	return 2*(len(pattern)-stars) - stars
}

// checkEndpoint validates an entry's optional endpoint ("" = every endpoint).
func checkEndpoint(where, e string) error {
	if e != "" && !validEndpoint(Endpoint(e)) {
		return fmt.Errorf("rules: %s: endpoint %q: want chat, messages or responses (or none, for every endpoint)", where, e)
	}
	return nil
}

// Ruleset is one parsed, validated rule bundle. A nil *Ruleset has no rules:
// every value is unknown and there are no limits.
type Ruleset struct {
	version uint64
	env     string
	hash    string
	// families by identity group: AnyIdentity, or one lower-cased identity.
	families map[string][]parsedFamily
	// rows holds, per row (and optional address), one entry per site.
	rows map[rowID][]map[string]paramRule
}

// Version is the bundle's version number (0 for a nil Ruleset).
func (r *Ruleset) Version() uint64 {
	if r == nil {
		return 0
	}
	return r.version
}

// Env is the environment the bundle was written for (EnvAny for one that
// holds everywhere, such as the built-in rules; "" for a nil Ruleset).
func (r *Ruleset) Env() string {
	if r == nil {
		return ""
	}
	return r.env
}

// Hash is the hex SHA-256 of the bundle bytes the rules were parsed from.
func (r *Ruleset) Hash() string {
	if r == nil {
		return ""
	}
	return r.hash
}

// ParseRules parses and validates a rule bundle. It refuses anything the
// format cannot express or that could read differently to a person and to the
// parser: an unknown parameter, value class or translation; a translation not
// allowed for that value; limits on a parameter that takes none; an unknown
// field; a duplicate key anywhere; a key that is not lower-case; a duplicate
// row; a row given both with and without sites.
func ParseRules(data []byte) (*Ruleset, error) {
	// An escape lets a key or value read differently to a reviewer than to the
	// parser ("\u0073top" is stop, "qwen\/q3" is qwen/q3); write characters as
	// they are. No backslash appears anywhere in a bundle.
	if len(data) > maxBundleBytes {
		return nil, fmt.Errorf("rules: bundle is %d bytes, more than %d", len(data), maxBundleBytes)
	}
	if bytes.IndexByte(data, '\\') >= 0 {
		return nil, errors.New(`rules: backslash escapes are not allowed; write characters as they are`)
	}
	if err := checkKeys(data); err != nil {
		return nil, fmt.Errorf("rules: %w", err)
	}
	var f rulesFile
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&f); err != nil {
		return nil, fmt.Errorf("rules: %w", err)
	}
	if len(bytes.Trim(data[dec.InputOffset():], " \t\r\n")) > 0 {
		return nil, errors.New("rules: trailing data after the bundle")
	}
	if f.Schema != RulesSchema {
		return nil, fmt.Errorf("rules: schema %d, this library reads %d", f.Schema, RulesSchema)
	}
	if f.Version == 0 {
		return nil, errors.New("rules: version must be at least 1")
	}
	if !identityValue(f.Env) {
		return nil, fmt.Errorf("rules: env %q: name the environment the bundle is for (ASCII letters, digits, . _ -), or %q", f.Env, EnvAny)
	}
	if err := checkText("bundle", f.Note, nil); err != nil {
		return nil, err
	}
	sum := sha256.Sum256(data)
	rs := &Ruleset{version: f.Version, env: strings.ToLower(f.Env), hash: hex.EncodeToString(sum[:]), rows: map[rowID][]map[string]paramRule{}}

	// Families naming AnyIdentity apply first, then the rest (see Lookup).
	var anyFams, idFams []parsedFamily
	type famID struct{ identity, pattern string }
	seenFam := map[famID]bool{}
	for i, fam := range f.Families {
		where := fmt.Sprintf("families[%d]", i)
		if fam.Identity == "" || fam.Models == "" {
			return nil, fmt.Errorf("rules: %s: identity and models are required", where)
		}
		if fam.Identity != AnyIdentity && !identityValue(fam.Identity) {
			return nil, fmt.Errorf("rules: %s: identity %q: use ASCII letters, digits, . _ -", where, fam.Identity)
		}
		if !modelValue(fam.Models) {
			return nil, fmt.Errorf("rules: %s: models %q: use visible ASCII", where, fam.Models)
		}
		if len(fam.Models) > maxPatternLen || strings.Count(fam.Models, "*") > maxPatternStar {
			return nil, fmt.Errorf("rules: %s: models %q: at most %d characters and %d '*'", where, fam.Models, maxPatternLen, maxPatternStar)
		}
		if strings.Contains(fam.Models, "**") {
			return nil, fmt.Errorf("rules: %s: models %q: write one * (** matches the same models and would rank differently)", where, fam.Models)
		}
		if err := checkText(where, fam.Note, fam.Source); err != nil {
			return nil, err
		}
		params, err := parseParams(where, fam.Params, fam.Source)
		if err != nil {
			return nil, err
		}
		fid := famID{strings.ToLower(fam.Identity), fam.Models}
		if seenFam[fid] {
			return nil, fmt.Errorf("rules: %s: duplicate family %s / %s", where, fam.Identity, fam.Models)
		}
		seenFam[fid] = true
		pf := parsedFamily{identity: strings.ToLower(fam.Identity), pattern: fam.Models, parts: strings.Split(fam.Models, "*"), specificity: familySpecificity(fam.Models), params: params}
		if fam.Identity == AnyIdentity {
			anyFams = append(anyFams, pf)
		} else {
			idFams = append(idFams, pf)
		}
	}
	rs.families = map[string][]parsedFamily{}
	for _, fam := range append(anyFams, idFams...) {
		rs.families[fam.identity] = append(rs.families[fam.identity], fam)
		if len(rs.families[fam.identity]) > maxFamilies {
			return nil, fmt.Errorf("rules: more than %d families for identity %q", maxFamilies, fam.identity)
		}
	}
	if len(f.Rows) > maxRows {
		return nil, fmt.Errorf("rules: %d rows, more than %d", len(f.Rows), maxRows)
	}

	type siteID struct {
		row  rowID
		site string
	}
	seen := map[siteID]bool{}
	sited := map[rowID]bool{} // whether the row's entries name a site
	for i, row := range f.Rows {
		where := fmt.Sprintf("rows[%d]", i)
		if row.Model == "" {
			return nil, fmt.Errorf("rules: %s: model is required", where)
		}
		if row.Identity == "" && row.Address == "" {
			return nil, fmt.Errorf("rules: %s: identity or address is required (a row that publishes no identity is named by its address)", where)
		}
		if row.Identity == AnyIdentity {
			return nil, fmt.Errorf("rules: %s: identity %q is for families only", where, AnyIdentity)
		}
		if row.Identity != "" && !identityValue(row.Identity) {
			return nil, fmt.Errorf("rules: %s: identity %q: use ASCII letters, digits, . _ -", where, row.Identity)
		}
		if err := checkEndpoint(where, row.Endpoint); err != nil {
			return nil, err
		}
		if row.Site != "" && !identityValue(row.Site) {
			return nil, fmt.Errorf("rules: %s: site %q: use ASCII letters, digits, . _ -", where, row.Site)
		}
		if !modelValue(row.Model) {
			return nil, fmt.Errorf("rules: %s: model %q: use visible ASCII", where, row.Model)
		}
		if strings.Contains(row.Model, "*") {
			return nil, fmt.Errorf("rules: %s: model %q: a row names one model exactly; use families for a pattern", where, row.Model)
		}
		if err := checkText(where, row.Note, row.Source); err != nil {
			return nil, err
		}
		if row.Address != "" && !addressValue(row.Address) {
			return nil, fmt.Errorf("rules: %s: address %q: want 0x and 40 hex digits", where, row.Address)
		}
		row.Identity = strings.ToLower(row.Identity)
		row.Address = strings.ToLower(row.Address)
		row.Site = strings.ToLower(row.Site)
		id := rowID{row.Identity, row.Model, row.Address, row.Endpoint}
		key := siteID{id, row.Site}
		if seen[key] {
			return nil, fmt.Errorf("rules: %s: duplicate row %s / %s (address %q, endpoint %q, site %q)", where, row.Identity, row.Model, row.Address, row.Endpoint, row.Site)
		}
		seen[key] = true
		if prev, ok := sited[id]; ok && prev != (row.Site != "") {
			return nil, fmt.Errorf("rules: %s: %s / %s is given both with and without a site; give every entry a site", where, row.Identity, row.Model)
		}
		sited[id] = row.Site != ""
		params, err := parseParams(where, row.Params, row.Source)
		if err != nil {
			return nil, err
		}
		rs.rows[id] = append(rs.rows[id], params)
		if len(rs.rows[id]) > maxSitesPerRow {
			return nil, fmt.Errorf("rules: %s: more than %d sites for %s / %s", where, maxSitesPerRow, row.Identity, row.Model)
		}
	}
	return rs, nil
}

// Value checks: what a reviewer reads must be what matches, so identities,
// model ids and addresses are restricted to visible ASCII (no zero-width or
// look-alike characters).
func identityValue(v string) bool {
	return v != "" && allBytes(v, func(c byte) bool {
		return c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '.' || c == '_' || c == '-'
	})
}

func modelValue(v string) bool {
	return v != "" && allBytes(v, func(c byte) bool { return c >= 0x21 && c <= 0x7e })
}

func addressValue(v string) bool {
	if len(v) != 42 || (v[:2] != "0x" && v[:2] != "0X") {
		return false
	}
	return allBytes(v[2:], func(c byte) bool {
		return c >= '0' && c <= '9' || c >= 'a' && c <= 'f' || c >= 'A' && c <= 'F'
	})
}

// textValue checks free text (note, source): printable, with no control or
// format characters (bidi overrides could make a line read differently from
// what it contains).
func textValue(v string) bool {
	for _, r := range v {
		if r == unicode.ReplacementChar || unicode.IsControl(r) || unicode.Is(unicode.Cf, r) || !unicode.IsPrint(r) && !unicode.Is(unicode.Zs, r) {
			return false
		}
	}
	return true
}

func checkText(where string, note string, source map[string]string) error {
	if !textValue(note) {
		return fmt.Errorf("rules: %s: note has a control, format or invalid character", where)
	}
	for p, v := range source {
		if !textValue(v) {
			return fmt.Errorf("rules: %s: source.%s has a control, format or invalid character", where, p)
		}
	}
	return nil
}

func allBytes(v string, ok func(byte) bool) bool {
	for i := 0; i < len(v); i++ {
		if !ok(v[i]) {
			return false
		}
	}
	return true
}

// maxDepth bounds how deeply a bundle may nest; real bundles nest five levels.
const maxDepth = 32

// Size bounds. Lookups run per candidate row and parameter on every request,
// so a mistaken but validly signed bundle must not be able to make them slow:
// families are indexed by identity, and these cap what one bundle may hold.
const (
	maxBundleBytes = 8 << 20
	maxFamilies    = 200 // per identity group ("*", or one identity)
	maxPatternLen  = 128
	maxPatternStar = 4
	maxRows        = 50000
	maxSitesPerRow = 16
)

// keyChar reports whether c may appear in a key: lower-case ASCII letters,
// digits and _ . : - only.
func keyChar(c byte) bool {
	return c >= 'a' && c <= 'z' || c >= '0' && c <= '9' || c == '_' || c == '.' || c == ':' || c == '-'
}

// checkKeys walks the raw JSON and refuses duplicate keys in any object, keys
// outside lower-case ASCII, and nesting deeper than maxDepth. encoding/json
// takes the last duplicate and matches struct fields with Unicode case
// folding (so "verſion", with a long s, fills Version), so without this a
// bundle could read one way to its reviewer and load another way.
func checkKeys(data []byte) error {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	depth := 0
	var walk func() error
	walk = func() error {
		tok, err := dec.Token()
		if err != nil {
			return err
		}
		if tok == json.Delim('{') || tok == json.Delim('[') {
			depth++
			if depth > maxDepth {
				return fmt.Errorf("nested deeper than %d levels", maxDepth)
			}
			defer func() { depth-- }()
		}
		switch tok {
		case json.Delim('{'):
			seen := map[string]bool{}
			for dec.More() {
				kt, err := dec.Token()
				if err != nil {
					return err
				}
				k := kt.(string)
				if seen[k] {
					return fmt.Errorf("duplicate key %q", k)
				}
				seen[k] = true
				for i := 0; i < len(k); i++ {
					if !keyChar(k[i]) {
						return fmt.Errorf("key %q: keys use only a-z 0-9 _ . : -", k)
					}
				}
				if err := walk(); err != nil {
					return err
				}
			}
			_, err = dec.Token() // '}'
			return err
		case json.Delim('['):
			for dec.More() {
				if err := walk(); err != nil {
					return err
				}
			}
			_, err = dec.Token() // ']'
			return err
		}
		return nil
	}
	return walk()
}

func parseParams(where string, raw map[string]json.RawMessage, source map[string]string) (map[string]paramRule, error) {
	if len(raw) == 0 {
		return nil, fmt.Errorf("rules: %s: params is empty", where)
	}
	for p := range source {
		if _, ok := raw[p]; !ok {
			return nil, fmt.Errorf("rules: %s: source names %q, which has no rule here", where, p)
		}
	}
	out := make(map[string]paramRule, len(raw))
	for name, v := range raw {
		spec, ok := catalog[name]
		if !ok {
			return nil, fmt.Errorf("rules: %s: parameter %q is not one rules may name", where, name)
		}
		if spec.category == CategoryPassthrough {
			return nil, fmt.Errorf("rules: %s: parameter %q is passed through as sent and never judged; rules may not name it", where, name)
		}
		pr, err := parseParam(spec, v)
		if err != nil {
			return nil, fmt.Errorf("rules: %s.%s: %w", where, name, err)
		}
		out[name] = pr
	}
	return out, nil
}

// parseParam reads one parameter's rule: a bare status string (shorthand for a
// parameter judged by presence alone), or an object mapping value classes to
// statuses, plus "limits": {"min": n, "max": n} for a parameter that takes them.
func parseParam(spec paramSpec, v json.RawMessage) (paramRule, error) {
	var bare string
	if json.Unmarshal(v, &bare) == nil {
		if len(spec.classes) != 1 {
			return paramRule{}, fmt.Errorf("a bare status needs a parameter with one value class; give one per class of %v", spec.classes)
		}
		st, err := parseStatus(spec, spec.classes[0], bare)
		if err != nil {
			return paramRule{}, err
		}
		return paramRule{statuses: map[string]Status{spec.classes[0]: st}}, nil
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(v, &obj); err != nil {
		return paramRule{}, errors.New("want a status string or an object")
	}
	if len(obj) == 0 {
		return paramRule{}, errors.New("empty rule")
	}
	pr := paramRule{statuses: map[string]Status{}}
	for k, val := range obj {
		if k == "limits" {
			if !spec.limits {
				return paramRule{}, errors.New("this parameter takes no limits")
			}
			lb, err := parseLimits(val)
			if err != nil {
				return paramRule{}, err
			}
			pr.limits = lb
			continue
		}
		if !spec.hasClass(k) {
			return paramRule{}, fmt.Errorf("value class %q is not one of %v", k, spec.classes)
		}
		var s string
		if err := json.Unmarshal(val, &s); err != nil {
			return paramRule{}, fmt.Errorf("%q: want a status string", k)
		}
		st, err := parseStatus(spec, k, s)
		if err != nil {
			return paramRule{}, fmt.Errorf("%q: %w", k, err)
		}
		pr.statuses[k] = st
	}
	return pr, nil
}

func parseLimits(v json.RawMessage) (limitBounds, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(v, &raw); err != nil || len(raw) == 0 {
		return limitBounds{}, errors.New("limits: want {\"min\": n, \"max\": n}")
	}
	var lb limitBounds
	for k, val := range raw {
		var n int
		if err := json.Unmarshal(val, &n); err != nil || n <= 0 {
			return limitBounds{}, fmt.Errorf("limits.%s: want a positive integer", k)
		}
		switch k {
		case "min":
			lb.min = &n
		case "max":
			lb.max = &n
		default:
			return limitBounds{}, fmt.Errorf("limits: unknown bound %q", k)
		}
	}
	if lb.min != nil && lb.max != nil && *lb.min > *lb.max {
		return limitBounds{}, fmt.Errorf("limits: min %d is above max %d", *lb.min, *lb.max)
	}
	return lb, nil
}

func parseStatus(spec paramSpec, class, s string) (Status, error) {
	switch s {
	case "supported":
		return Status{Kind: StatusSupported}, nil
	case "rejected":
		return Status{Kind: StatusRejected}, nil
	case "ignored":
		return Status{Kind: StatusIgnored}, nil
	}
	if name, ok := strings.CutPrefix(s, "translate:"); ok {
		allowed := spec.translationsFor(class)
		if !containsString(allowed, name) {
			return Status{}, fmt.Errorf("translation %q is not allowed for %q (allowed: %v)", name, class, allowed)
		}
		return Status{Kind: StatusTranslate, Translation: name}, nil
	}
	return Status{}, fmt.Errorf("status %q: want supported, rejected, ignored or translate:<name>", s)
}

// matchModel matches a model id against a family pattern in which '*' stands
// for any run of characters, '/' included (model ids such as "qwen/qwen3").
func matchModel(pattern, model string) bool {
	return matchParts(strings.Split(pattern, "*"), model)
}

// matchParts is matchModel on a pattern already split at '*'.
func matchParts(parts []string, model string) bool {
	if len(parts) == 1 {
		return parts[0] == model
	}
	if !strings.HasPrefix(model, parts[0]) {
		return false
	}
	rest := model[len(parts[0]):]
	for _, p := range parts[1 : len(parts)-1] {
		i := strings.Index(rest, p)
		if i < 0 {
			return false
		}
		rest = rest[i+len(p):]
	}
	last := parts[len(parts)-1]
	return len(rest) >= len(last) && strings.HasSuffix(rest, last)
}

// norm returns the key in the case rule data is stored in, and false when the
// key cannot name any rule: rule data is visible ASCII only, and lower-casing
// anything else with Unicode rules would let a look-alike ("Kimi" with a
// Kelvin sign) match the rules of a different identity.
func (k RowKey) norm() (RowKey, bool) {
	if k.Model == "" {
		return RowKey{}, false // a row always has a model; never fall to "*" families
	}
	for _, v := range []string{k.Identity, k.Model, k.Address} {
		if !allBytes(v, func(c byte) bool { return c >= 0x21 && c <= 0x7e }) {
			return RowKey{}, false
		}
	}
	return RowKey{Identity: asciiLower(k.Identity), Model: k.Model, Address: asciiLower(k.Address)}, true
}

func asciiLower(v string) string {
	b := []byte(v)
	for i, c := range b {
		if c >= 'A' && c <= 'Z' {
			b[i] = c + ('a' - 'A')
		}
	}
	return string(b)
}

// familyGroups returns the families matching the row: [0] AnyIdentity, [1]
// the row's identity.
func (r *Ruleset) familyGroups(row RowKey) [2][]parsedFamily {
	var out [2][]parsedFamily
	ids := [2]string{AnyIdentity, row.Identity}
	for i, id := range ids {
		if id == "" || i == 1 && id == AnyIdentity {
			continue
		}
		for _, fam := range r.families[id] {
			if matchParts(fam.parts, row.Model) {
				out[i] = append(out[i], fam)
			}
		}
	}
	return out
}

// rowLayers are the row's own layers in the order they apply: its rules for
// every endpoint, its rules for this endpoint, then the same two for the
// address.
func rowLayers(row RowKey, ep Endpoint) []rowID {
	var out []rowID
	if row.Identity != "" {
		out = append(out, rowID{row.Identity, row.Model, "", ""}, rowID{row.Identity, row.Model, "", string(ep)})
	}
	if row.Address != "" {
		out = append(out, rowID{row.Identity, row.Model, row.Address, ""}, rowID{row.Identity, row.Model, row.Address, string(ep)})
	}
	return out
}

// familyLimits is one group's limits, bound by bound like familyStatus;
// equally specific families that disagree intersect (the higher min, the
// lower max).
func familyLimits(fams []parsedFamily, param string) limitBounds {
	var out limitBounds
	var bestMin, bestMax int
	for _, fam := range fams {
		l := fam.params[param].limits
		if l.min != nil && (out.min == nil || fam.specificity > bestMin || fam.specificity == bestMin && *l.min > *out.min) {
			out.min, bestMin = l.min, fam.specificity
		}
		if l.max != nil && (out.max == nil || fam.specificity > bestMax || fam.specificity == bestMax && *l.max < *out.max) {
			out.max, bestMax = l.max, fam.specificity
		}
	}
	return out
}

// Lookup returns the row's status for one value class of a parameter (class
// as ValueClasses names it; anything else is unknown). Check CategoryOfValue
// first: passthrough and structural parameters are never judged and look up
// as unknown here. Layers apply in order,
// each overriding the one before: matching AnyIdentity families, matching
// families of the row's identity, the row's rules, the row's rules for the
// address. Rows (and address rows) may name an endpoint: in each of those
// layers the entries for every endpoint apply first, then the entries naming
// the request's endpoint. Families hold for every endpoint. How a supply
// line's conversion layer combines with the vendor's capability is worked
// out where the rules are generated (one entry per row and endpoint), not by
// layering here. Within a family layer the most specific
// pattern (most literal characters) decides; equally specific families that disagree give
// StatusUnknown. When a row has several sites (one upstream served from
// several places), each site's status is its own rule if it has one, else the
// layer below; the row's status holds only if every site agrees, and is
// StatusUnknown otherwise.
//
// ep is the endpoint the request came in on; any other value answers unknown,
// so a caller cannot skip endpoint-specific rules by leaving it out.
// For many lookups on one row (Plan: every parameter of a request on each
// candidate), take ForRow once and ask it: Lookup scans the row's families on
// every call.
func (r *Ruleset) Lookup(row RowKey, ep Endpoint, param, class string) Status {
	if r == nil || !validEndpoint(ep) {
		return Status{}
	}
	row, ok := row.norm()
	if !ok {
		return Status{}
	}
	var st Status
	for _, group := range r.familyGroups(row) {
		if s, ok := familyStatus(group, param, class); ok {
			st = s
		}
	}
	for _, id := range rowLayers(row, ep) {
		st = r.siteStatus(id, param, class, st)
	}
	return st
}

// familyStatus is one group's status for one value class, as mergeGroup
// computes it for every class at once.
func familyStatus(fams []parsedFamily, param, class string) (Status, bool) {
	var st Status
	best, found := 0, false
	for _, fam := range fams {
		s, ok := fam.params[param].statuses[class]
		if !ok {
			continue
		}
		switch {
		case !found || fam.specificity > best:
			st, best, found = s, fam.specificity, true
		case fam.specificity == best && s != st:
			st = Status{}
		}
	}
	return st, found
}

// pc names one value class of one parameter.
type pc struct{ param, class string }

// groupView is one family group merged once per row: per value class the
// deciding status (most specific family; equally specific ones that disagree
// give unknown), per parameter the limits (familyLimits).
type groupView struct {
	status map[pc]Status
	limits map[string]limitBounds
}

func mergeGroup(fams []parsedFamily) groupView {
	type agg struct {
		st   Status
		spec int
	}
	seen := map[pc]agg{}
	params := map[string]bool{}
	for _, fam := range fams {
		for name, pr := range fam.params {
			if pr.limits.min != nil || pr.limits.max != nil {
				params[name] = true
			}
			for class, st := range pr.statuses {
				k := pc{name, class}
				a, ok := seen[k]
				switch {
				case !ok || fam.specificity > a.spec:
					seen[k] = agg{st, fam.specificity}
				case fam.specificity == a.spec && st != a.st:
					seen[k] = agg{Status{}, a.spec}
				}
			}
		}
	}
	v := groupView{status: make(map[pc]Status, len(seen)), limits: map[string]limitBounds{}}
	for k, a := range seen {
		v.status[k] = a.st
	}
	for name := range params {
		v.limits[name] = familyLimits(fams, name)
	}
	return v
}

// RowRules is one row's rules on one endpoint, with its matching families
// merged once: Plan takes one per candidate row and asks it for every
// parameter of the request. Its lookups answer as Ruleset.Lookup / Limits do.
type RowRules struct {
	r      *Ruleset
	ok     bool
	row    RowKey
	ep     Endpoint
	groups [2]groupView
}

// ForRow returns the row's rules on endpoint ep. An invalid endpoint or key
// gives a RowRules that answers unknown.
func (r *Ruleset) ForRow(row RowKey, ep Endpoint) RowRules {
	if r == nil || !validEndpoint(ep) {
		return RowRules{}
	}
	row, ok := row.norm()
	if !ok {
		return RowRules{}
	}
	rr := RowRules{r: r, ok: true, row: row, ep: ep}
	for i, g := range r.familyGroups(row) {
		rr.groups[i] = mergeGroup(g)
	}
	return rr
}

// Lookup answers as Ruleset.Lookup.
func (rr RowRules) Lookup(param, class string) Status {
	if !rr.ok {
		return Status{}
	}
	var st Status
	for _, g := range rr.groups {
		if s, ok := g.status[pc{param, class}]; ok {
			st = s
		}
	}
	for _, id := range rowLayers(rr.row, rr.ep) {
		st = rr.r.siteStatus(id, param, class, st)
	}
	return st
}

// siteStatus applies one layer: base is the status from the layers below.
func (r *Ruleset) siteStatus(id rowID, param, class string, base Status) Status {
	sites := r.rows[id]
	if len(sites) == 0 {
		return base
	}
	var out Status
	for i, site := range sites {
		s, ok := site[param].statuses[class]
		if !ok {
			s = base
		}
		if i == 0 {
			out = s
		} else if s != out {
			return Status{}
		}
	}
	return out
}

// Limits returns the row's numeric range for a parameter, layered like
// Lookup, bound by bound: a layer that sets only max keeps the min from below.
// How a request value is held against them is D10, across candidates: rows
// whose Max is below the request are avoided first; only when no row fits is
// the request truncated (the largest-cap candidate first, cut to its Max; a
// fallback cut to its own Max; refused under require_parameters); below a row's Min, that row is avoided.
// Sites intersect: each site starts from the layer below and applies its own
// bounds, then the row takes the highest min and the lowest max. The result
// can have Min above Max (layers that contradict each other): no value fits,
// and callers must treat the row as unable to serve the parameter.
// The output cap is one per row: max_completion_tokens and max_output_tokens
// answer from max_tokens.
func (r *Ruleset) Limits(row RowKey, ep Endpoint, param string) (Limits, bool) {
	if r == nil || !validEndpoint(ep) {
		return Limits{}, false
	}
	row, ok := row.norm()
	if !ok {
		return Limits{}, false
	}
	rr := RowRules{r: r, ok: true, row: row, ep: ep}
	for i, g := range r.familyGroups(row) {
		rr.groups[i] = groupView{limits: map[string]limitBounds{}}
		p := param
		if p == "max_completion_tokens" || p == "max_output_tokens" {
			p = "max_tokens"
		}
		rr.groups[i].limits[p] = familyLimits(g, p)
	}
	return rr.Limits(param)
}

// Limits answers as Ruleset.Limits.
func (rr RowRules) Limits(param string) (Limits, bool) {
	if !rr.ok {
		return Limits{}, false
	}
	if param == "max_completion_tokens" || param == "max_output_tokens" {
		param = "max_tokens"
	}
	var cur limitBounds
	for _, g := range rr.groups {
		cur = cur.override(g.limits[param])
	}
	for _, id := range rowLayers(rr.row, rr.ep) {
		cur = rr.r.siteLimits(id, param, cur)
	}
	if cur.min == nil && cur.max == nil {
		return Limits{}, false
	}
	var out Limits
	if cur.min != nil {
		out.Min = *cur.min
	}
	if cur.max != nil {
		out.Max = *cur.max
	}
	return out, true
}

func (b limitBounds) override(o limitBounds) limitBounds {
	if o.min != nil {
		b.min = o.min
	}
	if o.max != nil {
		b.max = o.max
	}
	return b
}

func (r *Ruleset) siteLimits(id rowID, param string, base limitBounds) limitBounds {
	sites := r.rows[id]
	if len(sites) == 0 {
		return base
	}
	var out limitBounds
	for i, site := range sites {
		s := base.override(site[param].limits)
		if i == 0 {
			out = s
			continue
		}
		if s.min != nil && (out.min == nil || *s.min > *out.min) {
			out.min = s.min
		}
		if s.max != nil && (out.max == nil || *s.max < *out.max) {
			out.max = s.max
		}
	}
	return out
}
