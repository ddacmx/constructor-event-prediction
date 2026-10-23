# Constructor Fabric usage

The project artifacts were authored and registered through the installed Constructor Studio SDLC workflows in this order: `doc-prd`, `doc-adr`, `doc-design`, `decompose`, `doc-feature`, and `implement`. The authoring workflows (`doc-prd`, `doc-adr`, `doc-design`, `decompose`, and `doc-feature`) route to `cf-write-docs`; `implement` routes to `cf-coding`. The installed `cf-analyze` skill was available but was not invoked in the recorded artifact-authoring or implementation path.

The Constructor Studio engine is immutably pinned to `c8bf672016c47b6daa662882eb390865081f89a5` in `.cf-studio/version.toml`. The `constructorfabric/studio-kit-sdlc` kit is immutably pinned to `431812695894580d6933b088484fe14b663ad163` in `.cf-studio/config/core.toml`.

Implementation uses real `@cpt-*` scope markers at functions/classes and instruction markers around the smallest associated implementation fragments. The five FEATURE artifacts are registered with FULL traceability after their production code and tests are present. Higher-level PRD, ADR, DESIGN, and DECOMPOSITION artifacts remain DOCS-ONLY.

Validation and traceability use the installed `constructor-studio` executable rather than the conflicting PowerShell `cfs` alias, with a process-scoped UTF-8 setting:

```powershell
$env:PYTHONUTF8='1'
$studio = (Get-Command constructor-studio -ErrorAction Stop).Source
& $studio validate --local-only
& $studio spec-coverage --system event-prediction --verbose
& $studio validate-toc architecture/features/benchmarking-evaluation.md
```

The pinned CLI's `validate` command performs structure, cross-reference, and code-traceability checks; `spec-coverage` reports CDSL marker coverage; `validate-toc` checks the benchmark FEATURE's table of contents. Dependency mapping was also executed with `constructor-studio map --format json --out <temporary-path> --local-only -v`. Verification is an explicit multi-command procedure rather than a wrapper: run project CLI verification, pytest with branch coverage and the 90% threshold, and `pip check` as shown in the README; then run the Studio commands above. Each command retains its own output and exit status, and no command is claimed as executed unless its actual output was observed.

The final recorded Fabric gates passed with 0 errors and 0 warnings, all 27 required CPT IDs resolved to implementation/test evidence, and specification coverage of 11/23 checks (38.0%; granularity 0.2930). The dependency map contained 34 nodes and 214 edges with no dangling references. The separate final benchmark completed with 37 hash-bound artifacts; Fabric governed its requirements and traceability but did not execute model training.

No Studio or kit update was performed after pinning. The local `artifacts/smoke` and `artifacts/initial` directories are preliminary legacy evidence; their joblib models are not accepted by the safe loader and are excluded from the proposed public release.
