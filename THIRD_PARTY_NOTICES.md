# Bundled code and dependencies

## synthgen

`vendor/synthgen/` contains the compiler, rules, deterministic generator and verifier used by Cardinal Studio. `engine_snapshot.json` records the bundled package version and source fingerprint. The files are required for local validation and optional dataset generation; they are not an external checkout dependency.

The supplied generator snapshot does not include a license or redistribution notice. Its presence here does not establish permission to publish it. Before public distribution, obtain the relevant owner's authorization and applicable license/attribution terms, then include those terms accurately. This document does not grant a license or determine ownership.

Reference contracts and rule descriptions should also be reviewed for publication permission. No open-source license for Cardinal Studio is asserted by this checkout.

## Installed Python dependencies

Runtime packages are declared in `requirements.txt`; optional test packages are in `requirements-dev.txt`. They are installed from the configured package index, not copied into this repository. Their respective licenses and notices remain applicable. Do not commit `.venv/` or redistribute installed packages as part of a source cleanup.
