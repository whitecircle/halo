#!/usr/bin/env python
"""Every training script's parser carries ``trust_remote_code`` to the model arguments its loaders read. TRL moved
the field onto its trainer configs, so the toolkit's ``ModelArguments`` declares it; a bare ``ModelConfig`` fails
every load."""

from pathlib import Path

import pytest

from halo.cli import training_methods
from halo.configs.script_arguments import ModelArguments
from tests.common.repo import REPO_ROOT
from tests.cpu.config.test_examples_parse import TRAINING_ROOT, parser_for

_TRAINING_SCRIPTS = sorted(set(training_methods(REPO_ROOT).values()))


@pytest.mark.parametrize("script", _TRAINING_SCRIPTS, ids=lambda path: path.name)
def test_a_yaml_trust_remote_code_reaches_the_model_arguments(script: Path, tmp_path):
    parser = parser_for(str(script.relative_to(TRAINING_ROOT)))
    assert ModelArguments in parser.dataclass_types, (
        f"{script.name} parses its model arguments without trust_remote_code"
    )
    # bf16/use_cpu keep TrainingArguments constructible on a GPU-less runner; the distillation config
    # requires its teacher.
    yaml = {"model_name_or_path": "org/remote", "output_dir": str(tmp_path), "bf16": False, "use_cpu": True}
    yaml |= {"teacher_model": "org/teacher", "trust_remote_code": True}
    parsed = parser.parse_dict(yaml, allow_extra_keys=True)
    (model_args,) = [obj for obj in parsed if isinstance(obj, ModelArguments)]
    assert model_args.trust_remote_code is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
