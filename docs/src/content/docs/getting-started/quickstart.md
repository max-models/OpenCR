---
title: Quickstart
description: Run the command line entry point of opencr.
---

First, ensure that `opencr` is [installed](/opencr/getting-started/installation/).

## Basic usage

After installation, you can run the application with:

```bash
opencr
```

which calls `opencr.main.main` and prints a greeting.

## Use the package from Python

```python
from opencr.main import main

main()
```

See the [API reference](/opencr/api/opencr/) for every module and function.

## Profiling

opencr's coarse operations are [scope-profiler](https://github.com/max-models/scope-profiler)
regions (`opencr.main` here), tagged with the opt-in tag `opencr`: they are
off by default and cost a fraction of a microsecond per call. A code that profiles itself with
scope-profiler turns them on with the tag:

```python
import scope_profiler as sp

sp.setup(enable_tags=["opencr"])
```

or without editing the code, with `SCOPE_PROFILER_ENABLE_TAGS=opencr`,
`scope-profiler run --enable-tag opencr` or `enable_tags = ["opencr"]` in the
TOML configuration. New regions use `@sp.profile("opencr.<name>", tags=PROFILE_TAGS)`
with `PROFILE_TAGS` from the package's `_profiling` module.
