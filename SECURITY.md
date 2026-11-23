# Security

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's "Report a
vulnerability" button on the repository's Security tab, not in a public issue.

In scope, in particular: any way for a caller to spend past a budget, to have
spend booked to another tenant, or to have usage events dropped or duplicated.

## Verifying a release

Releases are built and published by the `release.yml` workflow, with no
long-lived upload credentials. Each file carries a signed provenance
attestation tying it to this repository and the tagged commit. To check a
downloaded file:

```bash
gh attestation verify spendgate-0.1.0-py3-none-any.whl --repo OWNER/spendgate
```

PyPI also shows the attestation on each file's page.

## What this library does not protect against

- It trusts the `tenant` you pass. Deriving the tenant from an authenticated
  caller is the application's job.
- It meters calls made through it. Code that calls a provider directly, with
  the same API key, is not limited.
