"""Manifest-driven GPU suite entrypoint.

This module holds no test logic — it is the single parametrized node that pytest expands,
via ``tests/gpu/conftest.py``, into one launch per ``(script, args)`` in
``tests/gpu/manifest.py``. Each ``GPUCase`` shells out ``torchrun`` for its script and
classifies the result (PASS / FAIL / ERROR / SKIP). Run it with the usual selectors::

    pytest -m "gpu and core"      # PR tier
    pytest -m gpu                 # full nightly tier
    pytest -m "gpu and ep"        # just Expert-Parallelism tests
"""


def test_gpu_script(gpu_case, tmp_path):
    gpu_case.run(tmp_path)
