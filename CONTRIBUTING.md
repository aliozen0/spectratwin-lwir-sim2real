# Contributing to SpectraTwin

## Development philosophy

Contributions should preserve the project's primary value: a reproducible synthetic-data research system with clear scientific boundaries. Large feature count is not a goal.

## Before opening a change

- Identify the subsystem and public contract affected by the change.
- Decide whether the change is a bug fix, implementation detail or new requirement.
- Document new behavior and long-lived architecture decisions with the change.

## Branch and PR style

Keep one concern per PR. Prefer a vertical slice that includes implementation, tests and docs over a large refactor followed by separate validation work.

A PR should state:

- problem,
- affected public contract,
- solution,
- alternatives considered when non-obvious,
- tests run,
- data/schema impact,
- experiment impact,
- known limitations.

## Code standards

- Typed Python at public boundaries.
- Pydantic contracts for persisted structured data.
- Explicit seeded randomness.
- Configurable values belong in config/schema, not scattered literals.
- `pathlib.Path` for filesystem paths.
- Structured logging for dataset jobs.
- Use meaningful domain names (`BandRadiance`, `FrameRecord`) rather than generic manager/helper classes.

## Quality checks

The authoritative commands will be implemented in repository tooling, but the intended gate is:

```bash
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest -q
```

Any GPU/render-heavy test should be separated from fast CI and documented.

## Research contribution rules

Every experiment must have:

- explicit hypothesis/question,
- frozen dataset versions,
- exact config,
- seed,
- model/checkpoint provenance,
- evaluation target,
- result artifacts,
- conclusion that distinguishes observation from interpretation.

Do not delete unfavorable runs from the research narrative merely because they do not support the initial hypothesis.


## Environment convention

Keep environment-specific paths and machine details out of committed defaults
and documentation. Remote runs must be tied to committed Git state;
notebook-only implementation changes are not acceptable contribution evidence.
