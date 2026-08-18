"""Regenerate the rollout-server config-parity fixtures.

Each fixture under ``fixtures/`` is a ``config.json`` the pinned vLLM server must parse: the one the
toolkit EXPORTS for that family, serialized by this image's transformers and put through every
export-side rewrite a checkpoint writer applies (``export_legacy_per_layer_config``,
``export_source_config_schema``). A ROSTER fixture builds it from the family's tiny config in
``tests/common/models.py``, through the vendor config module the fixture ships where transformers
has no class for the family. A HUB fixture — for a family whose export carries its SOURCE
checkpoint's schema — builds it from the RELEASE config at a pinned revision, plus the remote-code
modules that schema names. ``unparseable/`` holds the same configs WITHOUT the rewrites: the form
each rewrite exists to avoid, which the server's transformers must still refuse. ``check.py`` asserts
all of them against the server image at build time.

The roster is derived, not listed: every EP family whose layer class admits weight sync
(``_supports_weight_sync``) owes a fixture here, because the server has to parse that family's
checkpoint before a single tensor can be synced into it.
``tests/cpu/checkpoint/test_vllm_parity_fixtures.py`` fails when the roster or the rendered fixtures
drift. Regenerate in the training image (the hub fixtures need Hub access):

    python docker/vllm/parity/generate.py
"""

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from functools import cached_property

from huggingface_hub import snapshot_download
from transformers import CONFIG_MAPPING, AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.utils import CONFIG_NAME

# The export rewrites read their families off the EP roster.
import src.distributed.expert_parallel.layers.roster  # noqa: F401
from src.checkpoint.config_export import (
    export_legacy_per_layer_config,
    export_source_config_schema,
    restore_model_type,
)
from src.distributed.expert_parallel.expert_weights import ep_layer_classes
from src.models.moe_balancing import exports_source_config_schema, legacy_per_layer_config_keys
from tests.common.models import (
    TINY_BAILING_MOE_CONFIG,
    TINY_GEMMA4_MOE_CONFIG,
    TINY_GLM4_MOE_LITE_CONFIG,
    TINY_GPTOSS_CONFIG,
    TINY_LAGUNA_CONFIG,
    TINY_LFM2_MOE_CONFIG,
    TINY_QWEN3_MOE_CONFIG,
    TINY_QWEN35_MOE_CONFIG,
)

PARITY_ROOT = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class Fixture:
    """A family the roster carries a tiny config for: the export is that config, serialized here."""

    name: str
    model_type: str
    architecture: str
    config: dict
    sub_configs: dict = field(default_factory=dict)

    # A roster config is built in memory, so there is no source checkpoint to carry a schema from.
    source = None

    def build(self):
        kwargs = {"architectures": [self.architecture], **self.sub_configs}
        if self.sub_configs:
            kwargs["text_config"] = self.config
        else:
            kwargs.update(self.config)
        return CONFIG_MAPPING[self.model_type](**kwargs)


@dataclass(frozen=True)
class RemoteCodeFixture:
    """A roster family transformers ships no class for (Bailing/Ling): the same tiny config, built
    through the vendor config module the fixture ships for the server to import.

    That module is checked in beside the config it is shipped with — it is the vendor's own, not
    something this file renders — and declares neither ``model_type`` nor ``auto_map``. A
    checkpoint's ``config.json`` is what carries those two, and without them nothing resolves the
    family or finds the module, so the fixture supplies them.
    """

    name: str
    model_type: str
    architecture: str
    config: dict
    auto_map: dict

    source = None

    def build(self):
        config_class = get_class_from_dynamic_module(self.auto_map["AutoConfig"], fixture_dir("fixtures", self))
        return config_class(
            architectures=[self.architecture], model_type=self.model_type, auto_map=self.auto_map, **self.config
        )


@dataclass(frozen=True)
class HubFixture:
    """A family whose export carries its SOURCE checkpoint's schema: the RELEASE config at a pinned
    revision, put through that carry here.

    ``revision`` is pinned so a fixture never silently re-renders against a moved ``main`` — the
    gate would then certify a schema no checked-in code was reviewed against. The snapshot carries
    the config and every remote-code module, which is what the source-schema export reads.
    """

    name: str
    model_type: str
    repo: str
    revision: str

    @cached_property
    def source(self) -> str:
        return snapshot_download(self.repo, revision=self.revision, allow_patterns=[CONFIG_NAME, "*.py"])

    def build(self):
        return AutoConfig.from_pretrained(self.source)


FIXTURES = (
    Fixture("gpt_oss", "gpt_oss", "GptOssForCausalLM", TINY_GPTOSS_CONFIG),
    Fixture("qwen3_moe", "qwen3_moe", "Qwen3MoeForCausalLM", TINY_QWEN3_MOE_CONFIG),
    Fixture("glm4_moe_lite", "glm4_moe_lite", "Glm4MoeLiteForCausalLM", TINY_GLM4_MOE_LITE_CONFIG),
    Fixture(
        "qwen3_5_moe",
        "qwen3_5_moe",
        "Qwen3_5MoeForConditionalGeneration",
        TINY_QWEN35_MOE_CONFIG,
        sub_configs={"vision_config": None},
    ),
    Fixture(
        "gemma4",
        "gemma4",
        "Gemma4ForConditionalGeneration",
        TINY_GEMMA4_MOE_CONFIG,
        sub_configs={"vision_config": None, "audio_config": None},
    ),
    Fixture("laguna", "laguna", "LagunaForCausalLM", TINY_LAGUNA_CONFIG),
    Fixture("lfm2_moe", "lfm2_moe", "Lfm2MoeForCausalLM", TINY_LFM2_MOE_CONFIG),
    RemoteCodeFixture(
        "bailing_moe",
        "bailing_moe",
        "BailingMoeV2ForCausalLM",
        TINY_BAILING_MOE_CONFIG,
        auto_map={
            "AutoConfig": "configuration_bailing_moe_v2.BailingMoeV2Config",
            "AutoModel": "modeling_bailing_moe_v2.BailingMoeV2Model",
            "AutoModelForCausalLM": "modeling_bailing_moe_v2.BailingMoeV2ForCausalLM",
        },
    ),
    HubFixture("step3p7", "step3p7", "stepfun-ai/Step-3.7-Flash", "5f6244077ac62e04eec3f320501ff8c2b293373a"),
)
# The negative control is every fixture an export rewrite acts on, read off the registries the
# rewrites themselves read: a family that starts declaring either one gets its control without an
# edit here.
UNPARSEABLE = tuple(
    fixture
    for fixture in FIXTURES
    if legacy_per_layer_config_keys(fixture.model_type) or exports_source_config_schema(fixture.model_type)
)


def weight_sync_families() -> set[type]:
    """EP layer classes the weight sync admits — every family that owes a fixture here.

    A refused family (``_supports_weight_sync = False``, or every claimed ``model_type`` on the
    engine-side refusal list) is out: nothing syncs into it, so no server ever parses its export.
    """
    return {
        cls
        for cls in ep_layer_classes()
        if cls._supports_weight_sync and set(cls.HF_MODEL_TYPES) - set(cls._WEIGHT_SYNC_UNSUPPORTED_MODEL_TYPES)
    }


def uncovered_families() -> list[str]:
    """Names of the admitted families no fixture covers — empty is the contract this roster holds."""
    covered = {fixture.model_type for fixture in FIXTURES}
    return sorted(cls.__name__ for cls in weight_sync_families() if not covered & set(cls.HF_MODEL_TYPES))


def render(fixture, *, rewrite: bool) -> dict:
    """The ``config.json`` payload the toolkit writes for ``fixture`` — through the export rewrites
    by default, or the plain transformers serialization for the negative control."""
    with tempfile.TemporaryDirectory() as tmp:
        config = fixture.build()
        config.save_pretrained(tmp)
        restore_model_type(config, tmp)
        if rewrite:
            export_legacy_per_layer_config(tmp)
            export_source_config_schema(config, tmp, source=fixture.source)
        with open(os.path.join(tmp, CONFIG_NAME)) as f:
            payload = json.load(f)
    # Stable across patch releases of the writer; the pin decision lives in Dockerfile.vllm.
    payload.pop("transformers_version", None)
    return payload


def fixture_dir(role: str, fixture) -> str:
    return os.path.join(PARITY_ROOT, role, fixture.name)


def fixture_path(role: str, fixture) -> str:
    return os.path.join(fixture_dir(role, fixture), CONFIG_NAME)


def config_module(payload: dict) -> str | None:
    """The remote-code file a payload's ``auto_map`` names for ``AutoConfig`` — the only module
    beyond the config itself that the gate's parse imports. None for a natively-parsed family."""
    entry = (payload.get("auto_map") or {}).get("AutoConfig")
    return None if entry is None else f"{entry.split('.')[0]}.py"


def ship_config_module(fixture, directory: str, payload: dict) -> str | None:
    """Make sure the module ``payload``'s ``auto_map`` names sits beside it, and report which. None
    when the family parses natively. A fixture built from a source checkpoint takes the module from
    there, exactly as the export does; a vendored one already carries it. Anything else would ship a
    directory the gate cannot parse at all, so it raises."""
    module = config_module(payload)
    if module is None:
        return None
    destination = os.path.join(directory, module)
    if fixture.source is not None:
        shutil.copyfile(os.path.join(fixture.source, module), destination)
    elif not os.path.isfile(destination):
        raise ValueError(
            f"{fixture.name}: the rendered config names remote-code module {module!r}, which this "
            f"fixture neither vendors nor has a source to copy it from — the fixture would ship a "
            f"config the gate cannot parse."
        )
    return module


def write_all() -> None:
    uncovered = uncovered_families()
    if uncovered:
        raise SystemExit(
            f"{uncovered}: the weight sync admits these families but no fixture covers them — the "
            f"gate would certify a roster the server was never checked against. Add a fixture, or "
            f"refuse the family with _supports_weight_sync = False."
        )
    for role, fixtures, rewrite in (("fixtures", FIXTURES, True), ("unparseable", UNPARSEABLE, False)):
        for fixture in fixtures:
            directory = fixture_dir(role, fixture)
            path = fixture_path(role, fixture)
            os.makedirs(directory, exist_ok=True)
            payload = render(fixture, rewrite=rewrite)
            with open(path, "w") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
                f.write("\n")
            print(f"wrote {os.path.relpath(path, PARITY_ROOT)}")
            # Only the served fixtures get the module: the control must be refused exactly as the
            # unrewritten export is, and that export ships no remote-code modules either.
            module = ship_config_module(fixture, directory, payload) if role == "fixtures" else None
            if module is not None:
                print(f"shipped {role}/{fixture.name}/{module}")


if __name__ == "__main__":
    write_all()
