# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added a Message Batches job runner with cached system prompts, an atomic per-item state file, output validation, requeue of retryable failures, a urllib transport that never follows redirects, and a `submit`/`collect`/`requeue` CLI.
- Initial project structure from the factory utility template.
