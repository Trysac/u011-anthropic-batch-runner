# Standard-library transport with a per-item state file and custom_id attempt suffixes

Status: accepted (generated, pending human review)

## Context

Batch scripts lose track of which items succeeded, expired, errored or returned invalid output, and rerunning everything wastes money. The official SDK has no per-item state or requeue. The library must stay dependency-free, never leak the API key, and survive crashes between steps.

## Decision

Use urllib with a manually assembled opener that has no redirect handler, so x-api-key is never forwarded and any 302 is an http_error; base_url must be https and results_url must share scheme, host and port. Each request gets custom_id '<key>--a<attempt>' (key <= 59 chars, attempt 1-9, total <= 64) so late or stale result lines are matched to the current attempt and ignored otherwise. Item state lives in one JSON state file written through an fsynced temp file and os.replace, capped at 64 MiB, and holds the system prompt (checked against system_sha256) and user content so requeue needs no extra input, but never the API key. Collect reads all results before changing state and persists atomically, so a failed stream or write changes nothing. Terminal states (succeeded, failed, exhausted) are never rewritten. Only submitted items accept a result, which also makes the first line per custom_id win. Prompt caching via cache_control ephemeral on the system block is best effort in batches and needs the model's minimum cacheable length.

## Consequences

There is no third-party code to audit and secrets cannot reach the state file. The state file can be large for big jobs and contains prompts, so it should be protected like the inputs (it is created with owner-only permissions by mkstemp). A crash between a successful batch creation and the state write loses the batch id, which is inherent to a local file and documented by the order of operations. Polling, cancel, streaming, tool use and cost tracking are out of scope. Depth limiting of JSON relies on the interpreter recursion limit and maps RecursionError to an ignored line.
