import pytest

from lm_eval.tasks.aime import utils


def test_extract_all_responses_preserves_every_generation():
    responses = [
        [
            "Reasoning... $12$",
            "Reasoning... \\boxed{12}",
            "12",
            "Reasoning... \\boxed{13}",
        ]
    ]

    assert utils.extract_all_responses(responses, [{}]) == [
        ["12", "12", "12", "13"]
    ]


def test_avg_at_k_uses_all_four_generations():
    result = utils.avg_at_k(
        references=["12", "7"],
        predictions=[
            ["12", "12", "12", "13"],
            ["7", "8", "9", "10"],
        ],
        k=[4],
    )

    assert result == {"avg@4": 0.5}


def test_avg_at_k_rejects_insufficient_generations():
    with pytest.raises(ValueError, match="requires at least 4 generations"):
        utils.avg_at_k(
            references=["12"],
            predictions=[["12", "12", "12"]],
            k=[4],
        )
