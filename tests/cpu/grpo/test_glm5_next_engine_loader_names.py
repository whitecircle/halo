"""What GLM-5 Next's weight push forwards, held to the names both pinned engines' loaders map onto a served parameter
(``tests/common/engine_loader_names_glm5_next.json``, read off the loaders in the vLLM and SGLang images). An engine
drops a name it does not map behind a 200, so an experts, KDA conv or hyper-connection name that drifts reads as exact
on the next unchanged push."""

import json
import re

import pytest

from halo.models.moe.family_specs import GLM5_NEXT
from tests.common.repo import REPO_ROOT
from tests.common.weight_sync import ep_model, gathering_on_the_host, pushed_values

_RECORD = json.loads((REPO_ROOT / "tests/common/engine_loader_names_glm5_next.json").read_text(encoding="utf-8"))
_INDEX = re.compile(r"\.\d+(?=\.)")


@pytest.fixture(scope="module")
def pushed_names() -> set[str]:
    with gathering_on_the_host():
        return {_INDEX.sub(".N", name) for name in pushed_values(ep_model("glm5_next"))}


def test_the_record_spells_the_experts_the_layout_emits():
    experts = {name for name in _RECORD["accepted"] if ".mlp.experts." in name}
    gate, up, down = GLM5_NEXT.expert_layout.per_expert_keys
    assert experts == {f"model.language_model.layers.N.mlp.experts.N.{key}.weight" for key in (gate, up, down)}


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_every_pushed_name_is_one_the_engine_maps(engine, pushed_names):
    assert pushed_names <= set(_RECORD["accepted"]), sorted(pushed_names - set(_RECORD["accepted"]))
    assert not pushed_names & set(_RECORD["rejected"][engine])


def test_the_push_leaves_out_only_the_recorded_names(pushed_names):
    assert set(_RECORD["accepted"]) - pushed_names == set(_RECORD["left_out"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
