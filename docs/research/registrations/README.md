# Prospective registration records

One JSON file per registration of the ETF forecasters, added by a registration pull request (`docs/research/prospective_registration.md`, section 1). None exists yet: nothing is registered.

- **Written by** `scripts/forecast_prospective.py registration-record --bundle <bundle> --registration-doc-commit <commit>`, as `<bundle_id>.json`. A file is written once.
- **What it holds:**
  - the frozen bundle's manifest SHA-256 and its six registry entries (model ids, versions and entry hashes);
  - the config hash, the training-data fingerprint, the band method and the zero-volatility rule;
  - the code commit and the Git object ids of the registered code paths;
  - the commit of the registration document.
- **What it never holds:** vendor data, a forecast value or the ledger head hash. It is safe to commit publicly.
- **The registration time** is GitHub's merge time of the pull request that adds the file. After the owner merges it, `scripts/forecast_prospective.py register --record <file> --registration-commit <merge commit> --registered-at <merge time>` appends the ledger's first entry. The job then refuses any forecast that does not match this record.
