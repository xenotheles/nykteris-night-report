# Releasing

Releases are built and published by `.github/workflows/release.yml` when a `v*` tag is pushed.

## One-time setup on PyPI

1. Sign in at https://pypi.org (create an account if needed, with two-factor authentication).
2. Go to *Your projects* > *Publishing* > *Add a new pending publisher* and enter:
   - PyPI project name: `nykteris`
   - Owner: `xenotheles`
   - Repository name: `nykteris-night-report`
   - Workflow name: `release.yml`
   - Environment name: `pypi`
3. In the GitHub repository settings, under *Environments*, create an environment called `pypi`
   (optionally with yourself as a required reviewer, so every upload waits for a click).

No API token is stored anywhere; PyPI trusts this workflow directly.

## Each release

1. Set `version` in `pyproject.toml` and `__version__` in `src/nykteris/__init__.py`.
2. Move the CHANGELOG entry from "unreleased" to the date.
3. Merge to `main`, then tag and push:

   ```sh
   git tag v0.1.0
   git push origin v0.1.0
   ```

The workflow refuses to publish when the tag and the package version differ, and runs the tests first.
