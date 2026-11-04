# Infrastructure

The images, the rollout servers, the EP transport and the clusters Halo runs on.

- **[Docker](docker.md)** — The training and rollout-server images: launch recipe, contents, build, baked environment and registry pins.
- **[Rollout servers](rollout-servers.md)** — The vLLM 0.32.0 and SGLang 0.5.22 containers for RL: weight sync, server patches, required flags and the families each engine serves.
- **[DeepEP](deepep.md)** — The all-to-all kernels Expert Parallelism requires: build, rank-uniform knobs, buffer sizing, transport backends and cross-node EP over EFA.
- **[Ray cluster](ray.md)** — The rollout actor pool for async GRPO: lifecycle, sizing, multi-node setup and monitoring.
- **[Continuous Integration](ci.md)** — The hosted gates, the self-hosted CPU and GPU tiers, the Hub seed and the contribution gate.
- **[AWS authentication](aws-auth.md)** — Optional credentials for S3 datasets and a private ECR registry.
- **[SkyPilot](skypilot.md)** — Provisioned multi-node EP on AWS (EFA) and Nebius (InfiniBand).
- **[Nomad](nomad.md)** — Batch job specs for an existing Nomad cluster: GPU devices, scratch volume and multi-node rendezvous.
- **[RunPod](runpod.md)** — A manual multi-node EP runbook for RunPod InfiniBand pods.
