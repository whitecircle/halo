# NeMo 26.08.01 (ships Megatron Bridge 0.6.1 / Megatron Core 0.19.1 as editable installs in /opt/venv, TE 2.16,
# torch 2.13 nv26.6, DeepEP 1.2.1) upgraded to the newest releases Megatron Bridge 0.6.2 + Megatron Core 0.19.2
# (wheels from PyPI, no dependency changes).
FROM nvcr.io/nvidia/nemo:26.08.01@sha256:fdd6e9c7929b76c8624ddfea939dda345beaf67695d9c3994994c18f9e79b9bf
RUN uv pip uninstall --python /opt/venv/bin/python megatron-bridge megatron-core && \
    uv pip install --python /opt/venv/bin/python --no-deps megatron-bridge==0.6.2 megatron-core==0.19.2 && \
    printf 'all:\n\t@true\n' > /opt/venv/lib/python3.12/site-packages/megatron/core/datasets/Makefile
RUN python -c "import importlib.util as u; print(u.find_spec('megatron.bridge').origin, u.find_spec('megatron.core').origin)" && \
    grep -E '^(MAJOR|MINOR|PATCH) ' /opt/venv/lib/python3.12/site-packages/megatron/bridge/package_info.py
