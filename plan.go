package paramnorm

import (
	"fmt"
	"sort"
	"strings"
)

// Disposition is what to do with one value on one row.
type Disposition string

const (
	// Forward: send the value as it is.
	Forward Disposition = "forward"
	// Rewrite: send it after the named translation (Decision.Translation), and
	// notify the caller (D15).
	Rewrite Disposition = "rewrite"
	// Drop: remove it and notify (D7).
	Drop Disposition = "drop"
	// Avoid: the row cannot serve the request.
	Avoid Disposition = "avoid"
)

// PlanOptions are the request's routing preferences that change decisions.
type PlanOptions struct {
	// RequireParameters: the caller asked for strict guarantees
	// (provider.require_parameters): a drop-class value the row does not serve
	// as sent, an unknown drop-class value, a level approximation and an
	// output cap that would need truncating all avoid the row instead (D7, D9,
	// D10, D15).
	RequireParameters bool
	// RefuseUnknownFields: a body field the catalog does not know avoids every
	// row. Off in phase 1, which only records them (design section 3).
	RefuseUnknownFields bool
}

// Decision is what a row does with one value, and why.
type Decision struct {
	Value
	Status      Status
	Category    Category
	Do          Disposition
	Translation string
	Reason      string
}

// RowPlan is one candidate row's plan for a request.
type RowPlan struct {
	Row RowKey
	// Servable: no value avoids the row.
	Servable bool
	// Decisions: one per value the request carries (Classification.Values).
	Decisions []Decision
	// TruncateTo, when set, is the output cap the request must be cut to on
	// this row (D10: no candidate fits the request, so it is truncated to the
	// largest cap first and to each fallback's own cap). It is a rewrite to
	// notify (D15) like a Rewrite decision: the cap's own Decision stays Forward.
	TruncateTo int64
}

// PlanResult is the plan for a request over its candidate rows, in the order given.
type PlanResult struct {
	Rows           []RowPlan
	Classification Classification
}

// Servable returns the rows that can serve the request, in the order given,
// except that when the request is truncated (D10) the largest cap comes first.
func (p PlanResult) Servable() []RowPlan {
	var out []RowPlan
	for _, r := range p.Rows {
		if r.Servable {
			out = append(out, r)
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].TruncateTo > out[j].TruncateTo })
	return out
}

// RejectError is returned by Plan when no candidate can serve the request: the
// unsupported_parameter error (D11). Param and Class name the value that
// blocked (the one every candidate avoids, when there is one).
type RejectError struct {
	Param  string
	Class  string
	Reason string
}

func (e *RejectError) Error() string {
	return fmt.Sprintf("unsupported_parameter: %s (%s): %s", e.Param, e.Class, e.Reason)
}

// Decide returns what row rr does with value v (the table in rules_catalog.go).
func Decide(rr RowRules, v Value, opts PlanOptions) Decision {
	d := Decision{Value: v}
	cat, known := CategoryOfValue(v.Param, v.Class)
	if !known || v.Class == ClassUnknown {
		d.Category, d.Do, d.Reason = CategoryReject, Avoid, "not a value the catalog knows"
		return d
	}
	d.Category = cat
	if cat == CategoryPassthrough || cat == CategoryStructural || cat == CategoryContainer {
		d.Do = Forward
		return d
	}
	d.Status = rr.Lookup(v.Param, v.Class)
	drop := cat == CategoryDrop
	switch d.Status.Kind {
	case StatusSupported:
		d.Do = Forward
	case StatusTranslate:
		d.Translation = d.Status.Translation
		if opts.RequireParameters && strings.HasPrefix(d.Translation, TranslationLevelPrefix) {
			d.Do, d.Reason = Avoid, "a level approximation under require_parameters"
		} else {
			d.Do = Rewrite
		}
	case StatusIgnored, StatusRejected:
		what := map[StatusKind]string{StatusIgnored: "ignored", StatusRejected: "refused"}[d.Status.Kind]
		switch {
		case drop && !opts.RequireParameters:
			d.Do, d.Reason = Drop, "the row "+what+" it"
		case drop:
			d.Do, d.Reason = Avoid, "the row "+what+" it (require_parameters)"
		default:
			d.Do, d.Reason = Avoid, "the row "+what+" it"
		}
	default: // unknown
		switch {
		case isOutputCap(v.Param):
			// Every row takes an output cap: whether it fits is the row's
			// limits' question (D10), not this status's.
			d.Do = Forward
		case drop && !opts.RequireParameters:
			d.Do = Forward // as before rules existed
		case drop:
			d.Do, d.Reason = Avoid, "no rule says the row serves it (require_parameters)"
		default:
			d.Do, d.Reason = Avoid, "no rule says the row serves it"
		}
	}
	return d
}

// outputCapParams are the spellings of the output cap; Limits answers each
// from max_tokens.
var outputCapParams = []string{"max_tokens", "max_completion_tokens", "max_output_tokens"}

func isOutputCap(param string) bool {
	for _, p := range outputCapParams {
		if p == param {
			return true
		}
	}
	return false
}

// Plan decides, for each candidate row, whether it can serve a request body
// sent on endpoint ep, and what is translated or dropped there. It does not
// change the body. When no candidate can serve the request it returns the
// plan and a *RejectError; an error other than that is a body that is not a
// JSON object.
//
// The output cap follows D10 across the candidates: rows whose cap is below
// the request are avoided while another servable row fits it; when none fits,
// every servable row gets TruncateTo (its own cap), so a caller tries the
// largest cap first (refused under require_parameters); a row whose cap would
// leave no room above the request's thinking budget is avoided instead (the
// upstream refuses budget >= max_tokens). A request below a row's min avoids
// that row; so does a thinking budget outside the row's budget limits.
//
// Pass every candidate, before health and the caller's own routing rules
// filter them: a RejectError then means no row serves the request (400), and
// the caller tells 503 (served only by unhealthy rows) and 403 (only by rows
// its rules exclude) from the plan (D11). A request that sends no output cap
// gets the row's (ForRow(...).Limits) injected by the enforcing caller (D6).
func Plan(rs *Ruleset, body []byte, ep Endpoint, rows []RowKey, opts PlanOptions) (PlanResult, error) {
	cls, err := Classify(body, ep)
	if err != nil {
		return PlanResult{}, err
	}
	res := PlanResult{Classification: cls}
	// Every budget spelling is checked against its limits; the largest is the
	// one truncation must leave room for.
	var budgets []Value
	var budget *Value
	for i, v := range cls.Values {
		if (v.Param == "thinking.budget_tokens" || v.Param == "thinking_budget" || v.Param == "chat_template_kwargs.thinking_budget") && v.HasNumber {
			budgets = append(budgets, v)
			if budget == nil || v.Number > budget.Number {
				budget = &cls.Values[i]
			}
		}
	}
	// Every cap spelling is checked against the row's min (an upstream may
	// read any of them); the largest is the one that must fit under its max.
	var caps []Value
	var capValue *Value
	for i, v := range cls.Values {
		if isOutputCap(v.Param) && v.HasNumber {
			caps = append(caps, v)
			if capValue == nil || v.Number > capValue.Number {
				capValue = &cls.Values[i]
			}
		}
	}
	type capFit struct {
		max      int64
		hasMax   bool
		tooSmall bool
	}
	fits := make([]capFit, len(rows))
	for i, row := range rows {
		rr := rs.ForRow(row, ep)
		rp := RowPlan{Row: row, Servable: true}
		for _, v := range cls.Values {
			d := Decide(rr, v, opts)
			rp.Decisions = append(rp.Decisions, d)
			if d.Do == Avoid {
				rp.Servable = false
			}
		}
		if opts.RefuseUnknownFields && len(cls.Unknown) > 0 {
			rp.Servable = false
			rp.Decisions = append(rp.Decisions, Decision{
				Value: Value{Param: cls.Unknown[0], Class: ClassUnknown}, Category: CategoryReject, Do: Avoid,
				Reason: "a field the catalog does not know",
			})
		}
		for _, b := range budgets {
			target := b.Param
			if st := rr.Lookup(b.Param, ClassPresent); st.Kind == StatusTranslate && st.Translation == "thinking.budget_tokens" {
				target = st.Translation // a budget sent as another spelling meets the target's limits
			}
			if l, ok := rr.Limits(target); ok {
				switch {
				case l.Min > 0 && b.Number < int64(l.Min):
					rp.Servable = false
					rp.Decisions = append(rp.Decisions, capDecision(b, fmt.Sprintf("below the row's minimum thinking budget %d", l.Min)))
				case l.Max > 0 && b.Number > int64(l.Max):
					rp.Servable = false
					rp.Decisions = append(rp.Decisions, capDecision(b, fmt.Sprintf("above the row's maximum thinking budget %d", l.Max)))
				}
			}
		}
		if capValue == nil {
			// No cap sent: the cap an enforcing caller injects (the row's max,
			// D6) must still fit, and leave room above the thinking budget.
			if l, ok := rr.Limits("max_tokens"); ok && l.Max > 0 {
				reason := ""
				switch {
				case l.Min > l.Max:
					reason = fmt.Sprintf("the row's limits leave no cap (min %d above max %d)", l.Min, l.Max)
				case budget != nil && int64(l.Max) <= budget.Number:
					reason = fmt.Sprintf("the row's cap %d leaves no room above the thinking budget %d", l.Max, budget.Number)
				}
				if reason != "" {
					rp.Servable = false
					rp.Decisions = append(rp.Decisions, Decision{Value: Value{Param: "max_tokens", Class: ClassPresent}, Category: CategoryTranslate, Do: Avoid, Reason: reason})
				}
			}
		}
		if capValue != nil {
			if l, ok := rr.Limits(capValue.Param); ok {
				switch {
				case l.Max > 0 && l.Min > l.Max:
					// Sites' limits can intersect to nothing: no cap fits the row.
					rp.Servable = false
					rp.Decisions = append(rp.Decisions, capDecision(*capValue, fmt.Sprintf("the row's limits leave no cap (min %d above max %d)", l.Min, l.Max)))
				default:
					for _, cv := range caps {
						if l.Min > 0 && cv.Number < int64(l.Min) {
							rp.Servable = false
							rp.Decisions = append(rp.Decisions, capDecision(cv, fmt.Sprintf("below the row's minimum %d", l.Min)))
						}
					}
					if l.Max > 0 && capValue.Number > int64(l.Max) {
						fits[i] = capFit{max: int64(l.Max), tooSmall: true}
					}
				}
			}
		}
		res.Rows = append(res.Rows, rp)
	}
	if capValue != nil {
		anyFits := false
		for i, rp := range res.Rows {
			if rp.Servable && !fits[i].tooSmall {
				anyFits = true
			}
		}
		for i := range res.Rows {
			if !fits[i].tooSmall || !res.Rows[i].Servable {
				continue
			}
			switch {
			case anyFits:
				res.Rows[i].Servable = false
				res.Rows[i].Decisions = append(res.Rows[i].Decisions, capDecision(*capValue,
					fmt.Sprintf("above the row's cap %d while another row fits it", fits[i].max)))
			case opts.RequireParameters:
				res.Rows[i].Servable = false
				res.Rows[i].Decisions = append(res.Rows[i].Decisions, capDecision(*capValue,
					fmt.Sprintf("above the row's cap %d (require_parameters)", fits[i].max)))
			case budget != nil && fits[i].max <= budget.Number:
				res.Rows[i].Servable = false
				res.Rows[i].Decisions = append(res.Rows[i].Decisions, capDecision(*capValue,
					fmt.Sprintf("truncating to the row's cap %d leaves no room above the thinking budget %d", fits[i].max, budget.Number)))
			default:
				res.Rows[i].TruncateTo = fits[i].max
			}
		}
	}
	if len(res.Servable()) == 0 && len(rows) > 0 {
		return res, rejectFor(res)
	}
	return res, nil
}

func capDecision(v Value, reason string) Decision {
	return Decision{Value: v, Category: CategoryTranslate, Do: Avoid, Reason: reason}
}

// rejectFor names the value that blocked: one every row avoids, else the
// first row's first.
func rejectFor(res PlanResult) *RejectError {
	count := map[[2]string]int{}
	first := map[[2]string]Decision{}
	var order [][2]string
	for _, rp := range res.Rows {
		seen := map[[2]string]bool{}
		for _, d := range rp.Decisions {
			if d.Do != Avoid {
				continue
			}
			k := [2]string{d.Param, d.Class}
			if seen[k] {
				continue
			}
			seen[k] = true
			if _, ok := first[k]; !ok {
				first[k] = d
				order = append(order, k)
			}
			count[k]++
		}
	}
	for _, k := range order {
		if count[k] == len(res.Rows) {
			d := first[k]
			return &RejectError{Param: d.Param, Class: d.Class, Reason: d.Reason}
		}
	}
	if len(order) > 0 {
		d := first[order[0]]
		return &RejectError{Param: d.Param, Class: d.Class, Reason: d.Reason}
	}
	return &RejectError{Reason: "no candidate rows"}
}
