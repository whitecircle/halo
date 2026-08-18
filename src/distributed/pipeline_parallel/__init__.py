"""Pipeline parallelism — the outermost dimension, designed to cross NVLink domains: the world splits
into ``pp_size`` contiguous rank blocks, each owning a contiguous slice of the decoder layers and running
EP / pure-ETP + FSDP2 unchanged inside itself; only the P2P boundary activations leave a domain.
``split`` (per-family split contract + partition), ``stage`` (the stage module), ``lazy_loader``
(stage-aware loading), ``groups`` (per-chain P2P groups), ``runtime`` (the ``torch.distributed.pipelining`` seam)."""
