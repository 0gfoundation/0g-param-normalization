package paramnorm

// Mode says whether a category of handling runs, and whether its result is
// applied.
type Mode int

const (
	// Off skips the category.
	Off Mode = iota
	// DryRun computes the category's changes and reports them in
	// Result.Skipped, but leaves the body as it was.
	DryRun
	// On applies the category's changes and reports them in Result.Applied.
	On
)

// Options selects which categories Normalize runs. The zero value runs none,
// so each enforcement point turns categories on explicitly.
type Options struct {
	// Translate re-expresses a parameter under the spelling the row accepts
	// (max_tokens ↔ max_completion_tokens, reasoning_effort → the native
	// thinking control).
	Translate Mode
	// Drop removes values the row's engine does not accept where leaving the
	// upstream default stands as a valid reading (Row.RequestProfile).
	Drop Mode
	// CapOutput clamps the output-token cap to Row.MaxCompletionTokens and
	// injects it when the request has none (the broker's
	// enforceMaxCompletionTokens).
	CapOutput Mode
}

// Request is the request to normalize.
type Request struct {
	// Body is the JSON request body.
	Body []byte
	// Format is the API surface the request arrived on: APIFormatOpenAI or
	// APIFormatAnthropic.
	Format string
}

// Action is what was done to a parameter.
type Action string

const (
	// ActionTranslated: the parameter now appears as Change.To.
	ActionTranslated Action = "translated"
	// ActionDropped: the parameter was removed; Change.From holds the value
	// when it is worth reporting (truncated).
	ActionDropped Action = "dropped"
	// ActionCapped: the value was lowered to Change.To.
	ActionCapped Action = "capped"
	// ActionInjected: the parameter was absent and was set to Change.To.
	ActionInjected Action = "injected"
	// ActionNotInjected: a cap of Change.To would have been injected but did not
	// fit in the context window, so the request goes out without one. Reported
	// only; the body is not changed for it.
	ActionNotInjected Action = "not_injected"
)

// Change is one thing Normalize did (or, in DryRun, would do) to the request.
type Change struct {
	// Param is the parameter's dotted path in the body, e.g. "thinking.display".
	Param  string
	Action Action
	// From is the original value, when reported.
	From string
	// To is the new name (ActionTranslated) or value (ActionCapped,
	// ActionInjected, ActionNotInjected).
	To string
}

// bodyChanged reports whether any change altered the body.
func bodyChanged(changes []Change) bool {
	for _, c := range changes {
		if c.Action != ActionNotInjected {
			return true
		}
	}
	return false
}

// Result is the outcome of Normalize.
type Result struct {
	// Body is the request body to send; it is the input body when nothing was
	// applied.
	Body []byte
	// Applied lists the changes made to Body.
	Applied []Change
	// Skipped lists the changes a DryRun category would have made.
	Skipped []Change
}

// Normalize applies every enabled category of parameter handling for one
// upstream row, in a fixed order:
//
//  1. CapOutput: clamp or inject the output-token cap;
//  2. Translate: max_tokens ↔ max_completion_tokens;
//  3. Translate: reasoning_effort → the row's native thinking control;
//  4. Drop: values the row's request profile does not accept.
//
// The order is the broker's, and matters: the cap runs before the max_tokens
// rename so the rename normalizes whatever the client sent, and the profile
// runs last so it judges the body as it will be sent.
//
// row is the declaration of the row the request will be sent to; resolving it
// is the caller's job. A nil row leaves the request unchanged. The input body
// is never modified in place.
//
// An error means a body could not be re-encoded after a change; the request
// should fail rather than be sent half-normalized.
func Normalize(req Request, row *Row, opts Options) (Result, error) {
	res := Result{Body: req.Body}
	steps := []struct {
		mode Mode
		run  func([]byte) ([]byte, []Change, error)
	}{
		{opts.CapOutput, func(b []byte) ([]byte, []Change, error) { return capMaxOutputTokens(b, row) }},
		{opts.Translate, func(b []byte) ([]byte, []Change, error) { return translateMaxTokens(b, row) }},
		{opts.Translate, func(b []byte) ([]byte, []Change, error) { return translateReasoning(b, row) }},
		{opts.Drop, func(b []byte) ([]byte, []Change, error) {
			if row == nil {
				return b, nil, nil
			}
			return applyRequestProfile(b, row.RequestProfile, req.Format)
		}},
	}
	for _, s := range steps {
		if s.mode == Off {
			continue
		}
		out, changes, err := s.run(res.Body)
		if err != nil {
			return Result{Body: req.Body}, err
		}
		if s.mode == DryRun {
			res.Skipped = append(res.Skipped, changes...)
			continue
		}
		res.Body = out
		res.Applied = append(res.Applied, changes...)
	}
	return res, nil
}
