package paramnorm

import (
	"bytes"
	_ "embed"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
)

// builtinRules is the rule bundle compiled into this version of the library,
// which callers normally start a Store from (NewStore(BuiltinRules(), ...)).
// Its env is EnvAny: it holds in every environment.
//
//go:embed rules/builtin.json
var builtinRules []byte

// BuiltinRules returns a copy of the rule bundle compiled into this library.
func BuiltinRules() []byte { return append([]byte(nil), builtinRules...) }

// EmptyRules returns an initial bundle with no rules (env any), for a Store
// that must not start from the built-in broker rules: the router's own
// supply-line Store, whose rows would otherwise be served the built-in
// families until its first load.
func EmptyRules() []byte {
	return []byte(`{"schema":1,"version":1,"env":"any","families":[],"rows":[]}`)
}

// Verifier checks a bundle's signature and returns the signer's address. The
// library depends only on the standard library, so the caller supplies it (the
// broker: an EIP-191 recovery, as for its other signed operator inputs).
type Verifier interface {
	Verify(bundle, sig []byte) (signer string, err error)
}

// Store holds the rules in force. Reads never block; a load replaces the whole
// ruleset at once or leaves it as it was. Use NewStore; a zero or nil Store
// serves no rules (every value unknown) and refuses every load.
type Store struct {
	cur   atomic.Pointer[storeState]
	env   string
	floor uint64
}

// storeState is what a Store serves. initial marks the bundle the Store was
// created with: it is trusted compiled-in data and takes no part in the
// version check, so any later bundle at or above the floor replaces it.
type storeState struct {
	rs      *Ruleset
	initial bool
}

// StoreOptions configures a Store. Both come from configuration the process
// can trust (the broker: its attested compose file).
type StoreOptions struct {
	// Env names the environment the process serves ("staging", "mainnet").
	// Every bundle loaded later must name it exactly. Required.
	Env string
	// MinVersion is the lowest version a bundle may have. The version check
	// only remembers the bundle in force, which a restart forgets, and whoever
	// delivers bundles could then replay an older signed one: raise the floor
	// whenever a bundle must not be undone. 0 means no floor. The built-in
	// rules take no part in the version check, so after a library upgrade
	// brings newer built-in data, an older signed bundle can still replace
	// it: raise MinVersion with the upgrade to retire such bundles.
	MinVersion uint64
}

// NewStore returns a Store for an initial bundle (callers normally pass
// BuiltinRules()), which is trusted as compiled-in data: no signature, and its
// env must be EnvAny. The initial bundle is served only if its version is at
// least MinVersion; otherwise the Store serves no rules (every value unknown,
// so enforcing callers refuse) until a bundle at or above the floor loads: a
// restart must not bring back rules the floor was raised to retire. Either
// way the initial bundle never blocks a later one: a load is compared with
// the floor, and with the version of a bundle loaded before it, not with the
// initial bundle's version.
func NewStore(initial []byte, opts StoreOptions) (*Store, error) {
	if !identityValue(opts.Env) || strings.EqualFold(opts.Env, EnvAny) {
		return nil, fmt.Errorf("rules: store env %q: name the environment this process serves", opts.Env)
	}
	rs, err := ParseRules(initial)
	if err != nil {
		return nil, err
	}
	env := asciiLower(opts.Env)
	// Only compiled-in data (env "any") may start a Store: an initial bundle
	// skips the signature and the version check, so a bundle written for one
	// environment (the router's own, one a broker kept from an earlier load)
	// must come in through Load / LoadUnsigned like any other.
	if rs.env != EnvAny {
		return nil, fmt.Errorf("%w: an initial bundle must be for %q (built-in rules), got %q; load it after NewStore", ErrWrongEnv, EnvAny, rs.env)
	}
	s := &Store{env: env, floor: opts.MinVersion}
	if rs.version >= opts.MinVersion {
		s.cur.Store(&storeState{rs: rs, initial: true})
	}
	return s, nil
}

// Rules returns the ruleset in force (nil when the Store serves none; a nil
// *Ruleset answers every lookup with unknown).
func (s *Store) Rules() *Ruleset {
	if s == nil {
		return nil
	}
	if st := s.cur.Load(); st != nil {
		return st.rs
	}
	return nil
}

// ErrStaleRules is returned for a bundle whose version is not above the one
// loaded before it, or below the Store's minimum version.
var ErrStaleRules = errors.New("rules: stale version")

// ErrWrongEnv is returned for a bundle written for another environment.
var ErrWrongEnv = errors.New("rules: wrong environment")

// Load replaces the rules with a signed bundle, as a whole: nothing of the
// rules in force is kept, built-in model-family defaults included, so a
// bundle carries every family it needs. In order: the signature must
// verify and come from signer (case-insensitive, as hex addresses are), the
// bundle must parse and validate, name the Store's environment and pass the
// version checks. On any failure it returns the error and the rules in force
// stay. The bundle is copied on entry and the verifier gets its own copy, so
// neither the caller nor the verifier can change the bytes parsed after they
// were verified.
func (s *Store) Load(bundle, sig []byte, v Verifier, signer string) error {
	if v == nil || signer == "" {
		return errors.New("rules: a signed load needs a verifier and a signer")
	}
	bundle = bytes.Clone(bundle)
	got, err := v.Verify(bytes.Clone(bundle), sig)
	if err != nil {
		return fmt.Errorf("rules: signature: %w", err)
	}
	if !strings.EqualFold(got, signer) {
		return fmt.Errorf("rules: signed by %s, want %s", got, signer)
	}
	return s.replace(bundle)
}

// LoadUnsigned replaces the rules with a bundle the caller already trusts (the
// router reads its rules from its own repository). Validation, the environment
// and the version checks are the same as Load's. Never use it for bundles that
// arrive over a channel someone else controls: that is what Load is for.
func (s *Store) LoadUnsigned(bundle []byte) error { return s.replace(bytes.Clone(bundle)) }

func (s *Store) replace(bundle []byte) error {
	if s == nil || s.env == "" {
		return errors.New("rules: load into a Store not made by NewStore")
	}
	rs, err := ParseRules(bundle)
	if err != nil {
		return err
	}
	if rs.env != s.env {
		return fmt.Errorf("%w: bundle is for %q, this store serves %q", ErrWrongEnv, rs.env, s.env)
	}
	if rs.version < s.floor {
		return fmt.Errorf("%w: version %d is below the minimum %d", ErrStaleRules, rs.version, s.floor)
	}
	next := &storeState{rs: rs}
	for {
		old := s.cur.Load()
		if old != nil && !old.initial && rs.version <= old.rs.version {
			return fmt.Errorf("%w: version %d is not above %d, the version in force", ErrStaleRules, rs.version, old.rs.version)
		}
		if s.cur.CompareAndSwap(old, next) {
			return nil
		}
	}
}
