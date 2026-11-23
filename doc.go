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
// The router (plaintext requests), the E2EE gateway (encrypted requests) and the
// serving broker (fallback for encrypted requests) import the same version, so
// the rules are defined once. This module depends only on the standard library.
package paramnorm
