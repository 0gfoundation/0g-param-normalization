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

Rules are keyed by row and record a status per parameter value: supported,
rejected, ignored, translate-to, or unknown (treated as rejected). They come from
admission probes plus human review.

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
