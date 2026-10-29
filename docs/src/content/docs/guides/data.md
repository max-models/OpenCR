---
title: Data and provenance
description: How data enter OpenCR, what is recorded about them, and what OpenCR may ship.
---

## Provenance

Every reaction carries a `Provenance`: the reference (citation, DOI or URL, the equation or
table used), version and retrieval date, units and the conversions applied, ranges of validity,
interpolation and extrapolation, uncertainty and **licence**. Inverse reactions by detailed
balance need none: they are derived from their forward reaction.

```python
from opencr.data import check_network_data

check_network_data(model.network)  # [] when every reaction has its data and provenance
model.manifest()["sources"]  # every reference with its provenance
```

## Checks

Importers check what they read (`opencr.data.checks`): table shapes, missing or infinite
values, increasing energies and temperatures, non-negative values, thresholds, values that look
like cm² instead of m², level energies increasing with $n$, and the required provenance fields.
Problems are collected and raised together as a `DataError`.

## Formats

- **Native**: `save_network(network, path)` and `load_network(path)` write and read a versioned
  JSON document (`"format": "opencr-network", "version": 1`) with the species, states,
  reactions, rate data and provenance. Python-function rates are written as tables. Files of a
  newer major version are refused.
- **LXCat**: `opencr.data.importers.lxcat_reactions(path, target, products, database)` turns
  the excitation, ionization and attachment blocks of an LXCat file into reactions.
- **Rate tables**: `read_rate_table(path, reference, unit_factor=1e-6)` for two columns of
  temperature (eV) and rate coefficient.
- **ADAS adf11**: `read_adf11(path)` reads effective coefficients, for comparisons.

## Licences

OpenCR ships only data it may redistribute: published analytic fits
(`licence="published-fit"`) and public-domain constants. LXCat and OPEN-ADAS data may not be
redistributed; OpenCR reads the user's own downloads and records their terms in the provenance
(`Provenance.redistributable` is then false).
