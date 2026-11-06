# anthropic-batch-runner

[![PyPI version](https://img.shields.io/pypi/v/trysac-anthropic-batch-runner)](https://pypi.org/project/trysac-anthropic-batch-runner/)
[![License: MIT](https://img.shields.io/pypi/l/trysac-anthropic-batch-runner)](LICENSE)
[![CI](https://github.com/Trysac/u011-anthropic-batch-runner/actions/workflows/ci.yml/badge.svg)](https://github.com/Trysac/u011-anthropic-batch-runner/actions/workflows/ci.yml)

Run jobs on the Anthropic Message Batches API with cached system prompts, per-item state in an atomic local file, output validation and requeue of retryable failures

## Problem

Batch jobs cut LLM cost in half, but scripts that submit a batch and parse the results lose track of
which items succeeded, expired, errored or returned invalid output, and rerunning everything wastes
money. This library keeps a resumable state file per job, builds requests with a cached shared system
prompt, matches results by `custom_id`, validates outputs and requeues only items worth retrying. The
official SDK has no per-item state or requeue.

## Install

```sh
pip install trysac-anthropic-batch-runner
# or
uv add trysac-anthropic-batch-runner
```

Requires Python 3.10 or newer. No runtime dependencies.

## Usage

```python
import os

from anthropic_batch_runner import BatchJob, HttpBatchTransport, JobItem

items = [JobItem("a1", "Summarize: ..."), JobItem("a2", "Summarize: ...")]
job = BatchJob.create(
    "job.json", items, model="claude-sonnet-4-5", system="You summarize.", max_tokens=500
)
transport = HttpBatchTransport(api_key=os.environ["ANTHROPIC_API_KEY"])

job.submit(transport)
# Later, once the batch has ended:
print(job.collect(transport, validator=lambda text: len(text) > 0))
job.requeue_failed(transport)  # resubmits only retryable items, or returns None
```

CLI (the key comes only from `ANTHROPIC_API_KEY`):

```sh
python -m anthropic_batch_runner submit --state job.json --items items.jsonl \
    --model claude-sonnet-4-5 --system system.txt --max-tokens 500
python -m anthropic_batch_runner collect --state job.json
python -m anthropic_batch_runner requeue --state job.json
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) and [decisions](docs/decisions/)
- [Testing](docs/TESTING.md) and [security](docs/SECURITY.md)
- [Project record](docs/PROJECT.md) and [stack](docs/STACK.md)
- [Changelog](CHANGELOG.md) and [contributing guide](CONTRIBUTING.md)

## License

[MIT](LICENSE) (c) 2026 Miguel Gomez
