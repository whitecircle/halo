# Infrastructure

The images, the rollout servers, the EP transport and the clusters Halo runs on.

- [Docker](docker.md): the training and rollout-server images, with the launch recipe, contents, build, baked environment and registry pins.
- [Rollout servers](rollout-servers.md): weight sync, adapter sync, server patches, required flags and the families each engine serves, for the vLLM 0.32.0 and SGLang 0.5.22 containers and the vLLM build that serves Inkling 2.
- [DeepEP](deepep.md): the all-to-all kernels Expert Parallelism requires, with their build, rank-uniform knobs, buffer sizing, transport backends and cross-node EP over EFA.
- [Ray cluster](ray.md): the rollout actor pool for async GRPO, with its lifecycle, sizing, multi-node setup and monitoring.
- [Continuous Integration](ci.md): the hosted gates, the self-hosted CPU and GPU tiers, the Hub seed and the contribution gate.
- [AWS authentication](aws-auth.md): optional credentials for S3 datasets and a private ECR registry.
- [SkyPilot](skypilot.md): provisioned multi-node EP on AWS (EFA) and Nebius (InfiniBand).
- [Nomad](nomad.md): batch job specs for an existing Nomad cluster, covering GPU devices, the scratch volume and multi-node rendezvous.
- [RunPod](runpod.md): a manual multi-node EP runbook for RunPod InfiniBand pods.
