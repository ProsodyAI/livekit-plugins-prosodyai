# Contributing

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

[`examples/agent.py`](examples/agent.py) is the public usage surface.

## Checks

```bash
python -m pytest tests
ruff check .
python -m build
python -m twine check dist/*
```

The package must remain importable as `from livekit.plugins import prosodyai`, and importing it
must register exactly one `Plugin` instance with the LiveKit Agents runtime. The primary API is
`livekit.plugins.prosodyai`; the `livekit_plugins_prosodyai` namespace is a compatibility re-export.

## Maintainer automation

Pull requests and pushes to `dev` or `main` run the same CI matrix. After a
green `dev` push, the promotion workflow opens or updates the `dev` to `main`
pull request only when `PROSODYAI_RELEASE_TOKEN` is configured. It verifies
that the current `dev` head exactly matches the SHA that passed CI. It cannot
merge the pull request.

Use a fine-grained release token restricted to `ProsodyAI/livekit-plugins-prosodyai` with
`Contents: read` and `Pull requests: write`. If the token is absent, the
workflow emits a notice and stays green.

After every green `main` push, CI sends a `prosodyai_livekit_main_updated`
repository dispatch to `ProsodyAI/prosodyai` with these payload fields:

- `livekit_sha`: the exact merged `main` commit
- `source_repository`: `ProsodyAI/livekit-plugins-prosodyai`
- `source_ref`: `main`

The public repository needs a `PROSODYAI_ROOT_DISPATCH_TOKEN` Actions secret.
Use a fine-grained token restricted to `ProsodyAI/prosodyai` with only the
target repository's `Contents: write` permission, which GitHub requires to
create a repository dispatch. The root repository must listen for the event
and record or propose the new `livekit_sha`. If the secret is absent, CI emits
a notice and stays green so the root repository can poll as a fallback.

Publishing is separate from merging. To release:

1. Update `project.version` in `pyproject.toml` and `__version__` in
   `livekit/plugins/prosodyai/version.py` on `dev`.
2. Merge the green promotion pull request into `main`.
3. Create and push an explicit `vX.Y.Z` tag at that merged `main` commit.
4. Manually run the `Publish` workflow with the existing tag and confirmation enabled.
5. Approve the protected `pypi` environment after reviewing the resolved tag, SHA, and version.

The `pypi` environment must use PyPI Trusted Publishing for
`.github/workflows/release.yml`. Repository variable `PYPI_PUBLISH_ENABLED` must remain `false`
until that trust is configured and a release is intentionally approved. Pushing a tag alone never
publishes the package.
