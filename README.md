# anthropic-batch-runner

[![PyPI version](https://img.shields.io/pypi/v/trysac-anthropic-batch-runner)](https://pypi.org/project/trysac-anthropic-batch-runner/)
[![License: MIT](https://img.shields.io/pypi/l/trysac-anthropic-batch-runner)](LICENSE)
[![CI](https://github.com/Trysac/u011-anthropic-batch-runner/actions/workflows/ci.yml/badge.svg)](https://github.com/Trysac/u011-anthropic-batch-runner/actions/workflows/ci.yml)

Run jobs on the Anthropic Message Batches API with cached system prompts, per-item state in an atomic local file, output validation and requeue of retryable failures

## Problem

Describe in two or three sentences the concrete problem this utility solves, who hits it, and what
they do today without it. Replace this paragraph when the first real feature lands.

## Install

```sh
pip install trysac-anthropic-batch-runner
# or
uv add trysac-anthropic-batch-runner
```

Requires Python 3.10 or newer. No runtime dependencies.

## Usage

```python
import anthropic_batch_runner

# normalize() trims surrounding whitespace and lower-cases the text.
print(anthropic_batch_runner.normalize("  Hello World "))  # -> "hello world"

# Blank input is rejected explicitly instead of returning an empty string.
try:
    anthropic_batch_runner.normalize("   ")
except ValueError as exc:
    print(f"rejected: {exc}")
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) and [decisions](docs/decisions/)
- [Testing](docs/TESTING.md) and [security](docs/SECURITY.md)
- [Project record](docs/PROJECT.md) and [stack](docs/STACK.md)
- [Changelog](CHANGELOG.md) and [contributing guide](CONTRIBUTING.md)

## License

[MIT](LICENSE) (c) 2026 Miguel Gomez
