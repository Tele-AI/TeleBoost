# TeleBoost architecture

## Scope

This public source tree implements Wan training only. External families may be
provided by separately distributed plugins through the exact-name lazy program
entry-point contract; they are not built-in products or release artifacts.

## Layers

```text
cli
 └─> programs                 # sole composition root
      ├─> training            # model/algorithm/engine-neutral lifecycle
      ├─> engines             # FSDP and TeleTron distributed mechanisms
      ├─> models/wan          # Wan model, sampling, attention, parallel math
      ├─> algorithms          # pure math and structurally compatible hooks
      ├─> reward              # registry, routing, execution, providers
      ├─> datasets            # contracts and Wan preprocessing
      └─> config

recipes                       # declarative only; never imported by teleboost
third_party                   # source mirrors; never bundled into root artifacts
```

A program is a typed binding of `(family, algorithm, engine, policy)`. Most
programs are rows in `programs/builtins.py` assembled by common wiring; only a
topology that cannot be expressed by that assembler owns special composition
code.

## Dependency rules

- `programs` is the only layer allowed to assemble concrete families,
  algorithms, engines, rewards, and training hooks.
- `training/core` imports no concrete model family or family adapter.
- `training/families/wan` may adapt Wan training semantics but does not import
  the composition root.
- `models/wan` owns model computation; distributed framework mechanics live in
  `engines`.
- `engines/teletron` receives concrete model factories by injection and never
  imports models or training.
- `reward` never imports `programs`.
- `teleboost` never imports `recipes`.
- vendored `third_party` code never imports its TeleBoost host.

These rules are executable in `tests/architecture/`.

## Public source boundary

The Git branch is the release source of truth. Non-Wan family paths are not
retained as hidden implementations, compatibility facades, inactive registry
entries, recipes, dependency extras, or build-time rewrite inputs. The release
builder fails if this contract regresses, so source, sdist, and wheel remain
isomorphic at the product-family boundary.
