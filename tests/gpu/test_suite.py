"""The one parametrized node pytest expands, via ``tests/gpu/conftest.py``, into a launch per row of
``tests/gpu/manifest.py``."""


def test_gpu_script(gpu_case, tmp_path):
    gpu_case.run(tmp_path)
