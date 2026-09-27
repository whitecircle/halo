"""A code-contests episode refuses an ``answer`` payload holding no tests, at reset.

Graded against no tests, a submission scores 0 inside its GRPO group, indistinguishable from a wrong
solution, so a malformed or empty payload fails the episode instead of training as signal.
"""

import pytest

from src.environments.envs.tasks.coding.code_contests import CodeContestsEnvironment
from tests.common.code_contests import StubSandbox, reset_episode

_TEST = {"input": "1", "output": "1"}


def _env() -> CodeContestsEnvironment:
    return CodeContestsEnvironment(language="python", sandbox=StubSandbox())


@pytest.mark.parametrize(
    ("answer", "refusal"),
    [
        ('{"tests": []}', "holds no tests"),
        ("[]", "holds no tests"),
        ({"checker": "exact"}, "holds no tests"),
        ("not json", "unparseable 'answer' payload"),
        ("5", "must be a dict or list of tests, got int"),
    ],
)
def test_an_answer_without_tests_is_refused(answer, refusal):
    with pytest.raises(ValueError, match=refusal):
        reset_episode(_env(), {"answer": answer})


@pytest.mark.parametrize("answer", [[_TEST], {"test_cases": [_TEST]}, '{"tests": [{"input": "1", "output": "1"}]}'])
def test_every_accepted_payload_shape_stores_its_tests(answer):
    trajectory = reset_episode(_env(), {"answer": answer})
    assert trajectory.info["_test_cases"] == [_TEST]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
