# Infrastructure

Runtime image, cloud deployments, and hardware-specific dependency installation.

---

<!-- markdownlint-disable MD030 -- mkdocs-material grid cards require the 4-space content indent -->

<div class="grid cards" markdown>

-   :material-docker:{ .lg .middle } **Docker**

    ---

    Build, push, and run the images — Flash Attention, DeepEP V2, and all deps prebuilt for Hopper and Blackwell.

    [:octicons-arrow-right-24: Docker](docker.md)

-   :material-check-decagram:{ .lg .middle } **Continuous Integration**

    ---

    Hosted lint/docs gates and the self-hosted GPU tier — triggers, security controls, and how to enable it.

    [:octicons-arrow-right-24: Continuous Integration](ci.md)

-   :material-key-chain:{ .lg .middle } **AWS authentication**

    ---

    SSO, ECR, and S3 credentials for pulling images and datasets.

    [:octicons-arrow-right-24: AWS authentication](aws-auth.md)

-   :material-lan-connect:{ .lg .middle } **DeepEP installation**

    ---

    High-throughput all-to-all dispatch/combine kernels for MoE Expert Parallelism, required for EP.

    [:octicons-arrow-right-24: DeepEP installation](deepep.md)

-   :material-server-network:{ .lg .middle } **RunPod multi-node**

    ---

    Multi-node EP on RunPod InfiniBand — network setup, torchrun, and validation.

    [:octicons-arrow-right-24: RunPod multi-node](runpod.md)

-   :material-cloud-cog:{ .lg .middle } **SkyPilot deployment**

    ---

    Cloud deployment on AWS and Nebius — provisioning, storage, and cross-node EP.

    [:octicons-arrow-right-24: SkyPilot deployment](skypilot.md)

-   :material-cube-outline:{ .lg .middle } **Nomad deployment**

    ---

    Batch job specs for an existing Nomad cluster — GPU devices, scratch volume, and multi-node rendezvous.

    [:octicons-arrow-right-24: Nomad deployment](nomad.md)

-   :material-server-outline:{ .lg .middle } **Rollout servers**

    ---

    The vLLM and SGLang serving containers for RL: weight sync, required flags, throughput, and what each engine refuses.

    [:octicons-arrow-right-24: Rollout servers](rollout-servers.md)

-   :material-sitemap:{ .lg .middle } **Ray cluster**

    ---

    The rollout actor pool for environmental GRPO: lifecycle, sizing, multi-node setup, CPU budgeting, and monitoring.

    [:octicons-arrow-right-24: Ray cluster](ray.md)

</div>
