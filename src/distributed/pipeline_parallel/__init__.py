"""Pipeline parallelism: the world splits into ``pp_size`` contiguous rank blocks, each holding a
contiguous slice of the decoder layers and running EP / pure-ETP + FSDP2 unchanged inside itself.
Only the P2P boundary activations leave a block, which is why the split is placed across NVLink
domains. ``split`` (per-family split contract + partition), ``stage`` (the stage module),
``lazy_loader`` (stage-aware loading), ``groups`` (per-chain P2P groups), ``runtime``
(``torch.distributed.pipelining``)."""
