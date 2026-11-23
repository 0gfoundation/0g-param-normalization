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

// Step names the part of Normalize that produced a Change.
type Step string

const (
	// StepCapOutput: the output-token cap (Options.CapOutput).
	StepCapOutput Step = "cap_output"
	// StepMaxTokens: max_tokens ↔ max_completion_tokens (Options.Translate).
	StepMaxTokens Step = "max_tokens"
	// StepReasoning: reasoning_effort → the native thinking control
	// (Options.Translate).
	StepReasoning Step = "reasoning"
	// StepRequestProfile: values the row's request profile does not accept
	// (Options.Drop).
	StepRequestProfile Step = "request_profile"
)

// Change is one thing Normalize did (or, in DryRun, would do) to the request.
type Change struct {
	// Step is the part of Normalize that produced the change.
	Step Step
	// Param is the parameter's dotted path in the body, e.g. "thinking.display".
	Param  string
	Action Action
	// From is the original value, when reported. For ActionNotInjected it is
	// the estimated prompt size in tokens that left no room for the cap.
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
// An error (a *StepError) means a body could not be re-encoded after a change
// in an On category; the request should fail rather than be sent
// half-normalized. A DryRun category never fails the request: it changes
// nothing, so an encoding error there is ignored and its changes are still
// reported.
func Normalize(req Request, row *Row, opts Options) (Result, error) {
	return runSteps(req.Body, []step{
		{StepCapOutput, opts.CapOutput, func(b []byte) ([]byte, []Change, error) { return capMaxOutputTokens(b, row) }},
		{StepMaxTokens, opts.Translate, func(b []byte) ([]byte, []Change, error) { return translateMaxTokens(b, row) }},
		{StepReasoning, opts.Translate, func(b []byte) ([]byte, []Change, error) { return translateReasoning(b, row) }},
		{StepRequestProfile, opts.Drop, func(b []byte) ([]byte, []Change, error) {
			if row == nil {
				return b, nil, nil
			}
			return applyRequestProfile(b, row.RequestProfile, req.Format)
		}},
	})
}

// StepError is the error Normalize returns when a step could not re-encode
// the body. Its text is the step's own; Step lets a caller label it.
type StepError struct {
	Step Step
	Err  error
}

func (e *StepError) Error() string { return e.Err.Error() }
func (e *StepError) Unwrap() error { return e.Err }

type step struct {
	step Step
	mode Mode
	run  func([]byte) ([]byte, []Change, error)
}

func runSteps(body []byte, steps []step) (Result, error) {
	res := Result{Body: body}
	for _, s := range steps {
		if s.mode == Off {
			continue
		}
		out, changes, err := s.run(res.Body)
		for i := range changes {
			changes[i].Step = s.step
		}
		if s.mode == DryRun {
			res.Skipped = append(res.Skipped, changes...)
			continue
		}
		if err != nil {
			return Result{Body: body}, &StepError{Step: s.step, Err: err}
		}
		res.Body = out
		res.Applied = append(res.Applied, changes...)
	}
	return res, nil
}
