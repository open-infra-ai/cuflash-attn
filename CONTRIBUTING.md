# Contributing to cuflash

cuflash is maintained as a **from-scratch CUDA FlashAttention implementation** — built with optimization-iteration discipline (scalar → WMMA tensor-core forward, FlashDecoding/Split-KV, Roofline analysis). Favor fixes, refactors, documentation accuracy, and workflow simplification over feature growth.

## Prerequisites

- NVIDIA GPU with Compute Capability 7.0+ for full local validation
- CUDA Toolkit 12.x
- CMake 3.20+ (presets require 3.20; plain `cmake -B build` works with 3.18+)
- A C++17-capable compiler
- Node.js 18+ for documentation work

## Build, Test, and Format

```bash
cmake --preset release
cmake --build --preset release
ctest --preset release --output-on-failure

find . \( -name "*.cu" -o -name "*.cuh" -o -name "*.cpp" -o -name "*.h" \) \
  ! -path "*/build/*" | xargs clang-format -i

cd docs
npm ci
npm run docs:build
```

## Scope Policy

1. Preserve the current CUDA API surface and supported `head_dim` values: `32`, `64`, `128`.
2. Prefer deleting stale workflow layers, duplicated docs, and unused scaffolding over extending them.
3. Keep documentation aligned with the real repository structure and supported behavior.
4. Use CMake presets; do not introduce parallel build systems or AI control frameworks.
5. Work directly on the current branch unless a maintainer asks for a different flow.

## Documentation and Docs Site

- Keep GitHub Pages focused on product documentation, not process artifacts.
- Record project history only in the root `CHANGELOG.md`.
- If a page duplicates repository content without adding user value, remove or consolidate it.

## Code Style

- `clang-format` with the repository `.clang-format`
- namespaces `lower_case`, classes `CamelCase`, functions `lower_case`
- return `FlashAttentionError` from public APIs instead of throwing exceptions

## Review Checklist

Before opening a PR, confirm:

1. The build/test commands still make sense for the current environment.
2. Documentation and workflow files describe what the repository actually does today.
3. No deleted or deprecated process files are still linked from README, docs, or workflows.
