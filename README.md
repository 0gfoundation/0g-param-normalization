# 0g-param-normalization

Shared request-parameter normalization for 0G inference: one definition of how
each upstream row (model × upstream identity) handles each request parameter and
value, used by every enforcement point.

| Consumer | Handles |
|---|---|
| 0G router | plaintext requests: route around unsupported upstreams, rewrite, reject |
| E2EE gateway | encrypted requests: same, inside the TEE, before sealing |
| serving broker | fallback rewrite for encrypted requests from older clients |

Design: 0g-router#1536.

## Handling categories

| Category | When | Action |
|---|---|---|
| Translate | An equivalent spelling exists | Rewrite for the target upstream and report it |
| Reject / avoid | Dropping it changes cost or output structure | Skip upstreams that can't serve it; fail before dispatch if none can |
| Drop and notify | Sampling-only tweaks | Remove it, serve the request, report it |

Unknown parameters and values are treated as reject / avoid.

## Rules

Rules are keyed by row and record a status per parameter value: supported,
rejected, ignored, translate-to, or unknown (treated as rejected). They come from
admission probes plus human review.

## Constraints

- Standard library only (CI enforces it): the E2EE gateway compiles this into a
  measured image.
- Consumers pin a released version; the version and rules-data hash travel with
  each processed request so downstream points can skip duplicate work.

## Development

```sh
go test ./...
```

## License

Apache-2.0
