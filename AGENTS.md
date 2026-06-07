# Agent Instructions

These instructions are for LLM/coding agents working in this repository.

## Primary Rule

Use `veydra_model_standard` as the canonical source for runtime and parser logic.

- Preferred imports:
  - `from veydra_model_standard import VeydraModelStandard, SimulationContext`
  - `from veydra_model_standard.parser import parse_veydra_model_directory, build_flow_and_feedback_from_source_map`
- Do not introduce new dependencies on legacy parser paths under `shared/curator/tools/model_analyzer/*`.

## Parser Parity Requirements

When implementing analysis features:

1. Use package parser APIs first (`veydra_model_standard.parser`).
2. Keep output shapes stable for cloud and local consumers.
3. Avoid maintaining duplicate parser logic in multiple places.
4. If frontend parser artifacts are updated, ensure they are generated from package-owned source.

## MCP / Cloud Integration Expectations

If you touch MCP structure analysis behavior:

- `get_model_structure` should prefer live cloud parsing and use package parser logic.
- Fallback behavior is acceptable, but must clearly indicate fallback in the response payload.
- Maintain deterministic JSON error payloads for tool consumers.

## Packaging and Versioning

- Treat this package as standalone and reusable.
- Update `pyproject.toml` metadata when public API scope changes.
- Keep license and README consistent with exported capabilities.

## Documentation Expectations

- Keep README focused on:
  - objectives and differentiators,
  - quick start,
  - parser API usage,
  - licensing.
- Include runnable examples for both runtime and parser usage.

## Editing Safety

- Preserve backward-compatible public APIs unless explicitly changing a major version.
- Avoid broad refactors unrelated to the requested task.
- Prefer small, testable changes and validate imports after edits.
