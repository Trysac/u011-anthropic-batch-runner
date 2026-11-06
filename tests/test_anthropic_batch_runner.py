"""Behavior tests for the public API of anthropic_batch_runner."""

from __future__ import annotations

import pytest

import anthropic_batch_runner


def test_normalize_trims_and_lowercases() -> None:
    assert anthropic_batch_runner.normalize("  Hello World ") == "hello world"


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_normalize_rejects_blank_input(value: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        anthropic_batch_runner.normalize(value)


def test_version_is_semver() -> None:
    parts = anthropic_batch_runner.__version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
