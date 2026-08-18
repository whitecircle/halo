# Security Policy

## Reporting a vulnerability

Please report security issues **privately** — do not open a public issue for a
vulnerability.

- Use GitHub's [private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
  ("Report a vulnerability" under the repository's **Security** tab), or
- email **hello@whitecircle.ai**.

We aim to acknowledge a report within a few business days and will coordinate a fix and
disclosure timeline with you.

## Supported versions

Halo is released as Docker images plus this source repository. Security fixes target the
`main` branch and the most recent image tags.

## Secrets hygiene

This repository must never contain credentials. The following are gitignored and must
**never** be committed:

- `keys/` — signing keys and any private keys
- `.env` — `WANDB_API_KEY`, `HF_TOKEN`, API keys
- `*.pem` — certificates and private keys

If you believe a secret was committed, rotate it immediately and contact the maintainers —
rotation, not just removal from history, is required.
