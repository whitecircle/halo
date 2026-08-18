# Infrastructure

Runtime image, cloud deployments, and hardware-specific dependency installation.

---

- **[Docker](docker.md)** — Build, push, and run the images — Flash Attention, DeepEP V2, and all deps prebuilt for Hopper and Blackwell.
- **[Continuous Integration](ci.md)** — Hosted lint/docs gates and the self-hosted GPU tier — triggers, security controls, and how to enable it.
- **[AWS authentication](aws-auth.md)** — SSO, ECR, and S3 credentials for pulling images and datasets.
- **[DeepEP installation](deepep.md)** — High-throughput all-to-all dispatch/combine kernels for MoE Expert Parallelism, required for EP.
- **[RunPod multi-node](runpod.md)** — Multi-node EP on RunPod InfiniBand — network setup, torchrun, and validation.
- **[SkyPilot deployment](skypilot.md)** — Cloud deployment on AWS and Nebius — provisioning, storage, and cross-node EP.
- **[Nomad deployment](nomad.md)** — Batch job specs for an existing Nomad cluster — GPU devices, scratch volume, and multi-node rendezvous.
- **[Rollout servers](rollout-servers.md)** — The vLLM and SGLang serving containers for RL: weight sync, required flags, throughput, and what each engine refuses.
- **[Ray cluster](ray.md)** — The rollout actor pool for environmental GRPO: lifecycle, sizing, multi-node setup, CPU budgeting, and monitoring.
