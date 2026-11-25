// Package paramnorm decides, for each request parameter and value, how a given
// upstream row (model × upstream identity) handles it, and rewrites requests
// accordingly.
//
// Every parameter value falls into one of three categories:
//   - translate: an equivalent spelling exists; rewrite it for the target upstream;
//   - reject / avoid: dropping it would change cost or output structure; skip
//     upstreams that cannot serve it, and fail before dispatch if none can;
//   - drop and notify: sampling-only tweaks; remove them and tell the caller.
//
// Separately, Sanitize removes engine-control fields (a request chat template,
// template kwargs beyond the thinking switches, a pickled logit processor, a
// LoRA path): a security step that does not depend on the upstream row.
//
// The router (plaintext requests) and the serving broker (encrypted requests,
// after decrypting) import the same version, so the rules are defined once.
// This module depends only on the standard library.
package paramnorm
