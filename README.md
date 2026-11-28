# 0g-param-normalization

Shared request-parameter normalization for 0G inference: one definition of how
each upstream row (model × upstream identity) handles each request parameter and
value, used by every enforcement point.

| Consumer | Handles |
|---|---|
| 0G router | plaintext requests: route around unsupported upstreams, rewrite, reject; encrypted requests: route on their cleartext parameters and the gateway's type signals, never rewrite |
| serving broker | encrypted requests: rewrite after decrypting, inside the TEE |

The E2EE gateway does not use this library: it seals the request and sends the
router type-only signals for the sealed fields it routes on.

## Handling categories

| Category | When | Action |
|---|---|---|
| Translate | An equivalent spelling exists | Rewrite for the target upstream and report it |
| Reject / avoid | Dropping it changes cost or output structure | Skip upstreams that can't serve it; fail before dispatch if none can |
| Drop and notify | Sampling-only tweaks | Remove it, serve the request, report it |

Unknown parameters and values are treated as reject / avoid.

## Usage

One entry point does all parameter handling for one upstream row:

```go
res, err := paramnorm.Normalize(
	paramnorm.Request{Body: body, Format: paramnorm.APIFormatOpenAI},
	row, // *paramnorm.Row: the declaration of the row the request goes to
	paramnorm.Options{Translate: paramnorm.On, CapOutput: paramnorm.On},
)
// res.Body: the body to send
// res.Applied: what was changed (for response headers / trace)
// res.Skipped: what a DryRun category would have changed
```

Each category (`Translate`, `Drop`, `CapOutput`) is `Off`, `DryRun` or `On`,
so every enforcement point turns categories on one at a time, and can measure
first. The zero `Options` changes nothing. The order of the steps is fixed
inside the library. Resolving which row a request targets is the caller's job.

Each `Change` names the `Step` that produced it. An error is a `*StepError`
naming the step that failed; a `DryRun` category never fails the request.

`Sanitize` (engine-control fields; independent of the row) and
`AdvertisedSupportedParameters` (what a row lists in `/v1/models`) are the
other entry points.

## Rules

Rules record, per row and per parameter value, one of: supported, rejected,
ignored (accepted, no effect), translate (to a translation defined in this
library), or nothing (unknown, treated as rejected). They come from collected
provider capabilities, admission probes and human review.

The data is JSON (`rules/builtin.json` is compiled in):

```json
{"schema": 1, "version": 2, "env": "mainnet",
 "families": [{"identity": "*", "models": "gpt-5*", "params": {"max_tokens": {"limits": {"min": 16}}}},
              {"identity": "up1", "models": "gpt-5*", "params": {"top_k": "rejected"}}],
 "rows": [{"identity": "up1", "model": "gpt-5.1", "params": {
   "tool_choice": {"auto": "supported", "required": "ignored"},
   "reasoning_effort": {"none": "translate:enable_thinking"},
   "max_tokens": {"limits": {"max": 65536}}}}]}
```

- A row is (upstream identity, model); a row that publishes no identity is
  named by `address` alone.
- A row entry may name an `endpoint` (`chat`, `messages`, `responses`): a
  supply line's conversion layer can make one row serve a parameter on one
  endpoint and not another. `Lookup` / `Limits` take the request's endpoint
  (required; anything else answers unknown); in each row layer the entries
  for every endpoint apply first, then those naming the request's endpoint.
  Families hold for every endpoint. How a conversion layer combines with the
  vendor's capability is computed by the generator that writes the rules
  (one entry per row and endpoint), not by layering in the bundle. A
  lower layer's answer is inherited and cannot be cleared, so when rows share
  (identity, model), each row's final answer goes on its own `address`
  entries.
- `ForRow(row, endpoint)` merges a row's families once; Plan asks it for every
  parameter of a request.
- Families give defaults by model-id pattern (`*` matches any characters);
  `"identity": "*"` families hold on every upstream and apply first. Among
  matching families of one layer the most specific pattern decides (two
  points per literal character, less one per `*`: an exact id beats `id*`,
  which beats `*id*`), so the order they are written in never matters; equally specific ones that disagree give
  unknown. A row's own rules override families; an optional `address` layer
  overrides those (for a row that publishes an identity, an address rule must
  name it too). Limits layer bound by bound.
  Several `site`s of one row are intersected: a status holds only if every
  site has it (a site without its own rule inherits the layer below).
- The parameters, value classes and translations rules may name are a fixed
  table in code (`rules_catalog.go`), down to which translations each value
  may use (`none` can only become a native thinking-off switch). It covers the
  Chat Completions, Anthropic Messages and Responses spellings (`stop_sequences`,
  `text.format`, `reasoning.effort`, ...); `CategoryOf` / `CategoryOfValue`
  give each parameter's handling (request structure such as `messages` or
  `stream` reports `structural`; `IsContainer` names the objects such as
  `reasoning` whose members are judged as `reasoning.effort`), and the passthrough ones (`user`, `store`,
  `metadata`, `prompt_cache_key`) cannot be named by rules at all. Only
  `none` (and `thinking: disabled`) turn thinking off; `minimal` is the
  lowest level. Anything
  else fails to load, as do duplicate keys, rows, sites and families, keys
  that are not lower-case and any backslash escape, so rule data can never target `messages`, `model`, media,
  engine controls or any other field, and reads the same to a reviewer as to
  the parser. (A translation such as `enable_thinking` does write into the
  engine's thinking switch: that is code in this library, picked by name,
  never a field the data chooses.)
- `Store` serves the rules in force: `NewStore(BuiltinRules(),
  StoreOptions{Env, MinVersion})`, then `Load` a signed bundle (the caller
  supplies the `Verifier`, e.g. EIP-191) or `LoadUnsigned` one it already
  trusts. A loaded bundle replaces the rules in force as a whole (built-in
  family defaults included: a bundle repeats the families it needs). It must
  validate, name the store's `env` exactly (a
  staging bundle never loads on mainnet), carry a higher `version` than the
  bundle loaded before it and be at least `MinVersion`; on any failure the
  rules in force stay. The built-in rules (`"env": "any"`) take no part in the
  version check, so a library upgrade never locks out signed bundles, and are
  not served when their version is below `MinVersion`: the store then serves
  nothing (every value unknown) until a bundle at the floor arrives, so a
  restart cannot bring back rules the floor retired. The caller keeps `Env`
  and `MinVersion` in trusted configuration, and raises `MinVersion` with a
  library upgrade whose built-in rules must not be replaced by older signed
  bundles. `Ruleset.Hash` is the bundle's
  SHA-256, for logs.
- A second Store (the router's supply-line rules) starts from `EmptyRules()`,
  not the built-in broker rules. The output cap is one per row: only
  `max_tokens` carries limits (even on a row that rejects the `max_tokens`
  spelling), and `Limits` answers `max_completion_tokens` /
  `max_output_tokens` from it.
- A container field (`thinking`, `reasoning`, ...) sent as anything but a
  JSON object (or null, which counts as not sent) is unknown and refused.
- `rules/builtin.json` is public: only models already listed in the public
  mainnet `/v1/providers` belong in it.

## Plan

`Plan(rules, body, endpoint, rows, PlanOptions{RequireParameters, RefuseUnknownFields})`
decides, for each candidate row, whether it can serve a request and what is
translated or dropped there, without changing the body:

- `Classify` lists the values the body carries in the catalog's terms (one
  per parameter and value class), per endpoint spelling, leaving out values
  that ask for nothing (null, empty lists, `logprobs: false`, `n: 1`, ...),
  and the fields the catalog does not know.
- `Decide` applies the status x category table in `rules_catalog.go` to each
  value on each row: forward, rewrite (a named translation, to be notified),
  drop (and notify), or avoid the row.
- The output cap follows D10 across the candidates: rows whose cap is below
  the request are avoided while another row fits; when none fits, each row
  gets `TruncateTo`, its own cap.
- When no row can serve the request, `Plan` returns `*RejectError`
  (`unsupported_parameter`, D11) naming the value that blocked.

A router in dry-run mode records the plan and routes as before.

## Constraints

- Standard library only (CI enforces it), so importing it never pulls a
  dependency into the router or the broker (a TEE image).
- Consumers pin a released version.

## Development

```sh
go test ./...
```

## License

Apache-2.0
