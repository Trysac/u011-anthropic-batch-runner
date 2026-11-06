# Project record

## Objective

Run jobs on the Anthropic Message Batches API with cached system prompts, per-item state in an atomic local file, output validation and requeue of retryable failures

## Scope

- In scope: the public API documented in [README.md](../README.md), typed and tested.
- Out of scope: anything that needs runtime dependencies or I/O without an accepted ADR.

## Timeline

| Date | Event |
|---|---|
| 2026 | Repository created from the factory utility template. |

## Effort metrics

Measured values only; a value that has not been measured stays "pending measurement".

| Metric | Value |
|---|---|
| Tokens (input / output) | 614541 / 117323 |
| Hours (human) | pending measurement |
| Cost (USD) | 0.9563 |

## Factory run u011 (measured from API usage)

| Phase | Model | Batch | Input | Output | Cache write | Cache read | Cost (USD) |
|---|---|---|---|---|---|---|---|
| generation | claude-sonnet-5-5 | yes | 2594 | 100739 | 0 | 4246 | 0.5067 |
| fix-1 | claude-sonnet-5-5 | no | 6 | 1827 | 8928 | 41726 | 0.0448 |
| fix-2 | claude-sonnet-5-5 | no | 6 | 1278 | 7770 | 41138 | 0.0363 |
| fix-3 | claude-sonnet-5-5 | no | 10 | 955 | 8711 | 77914 | 0.0391 |
| fix-4 | claude-sonnet-5-5 | no | 8 | 1216 | 7868 | 56340 | 0.0375 |
| fix-1 | claude-sonnet-5-5 | no | 8 | 1061 | 7058 | 56784 | 0.0339 |
| audit | claude-sonnet-5-5 | no | 12 | 3293 | 32989 | 135695 | 0.1290 |
| fix-2 | claude-sonnet-5-5 | no | 6 | 1244 | 7812 | 40957 | 0.0361 |
| docs | claude-haiku-5-5 | no | 157091 | 5710 | 0 | 0 | 0.0928 |

Total: $0.9563. Prices: Anthropic list prices.
