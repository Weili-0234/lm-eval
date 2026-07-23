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


def test_extract_answer_uses_final_section_after_thinking():
    response = (
        "<think>Try $3+4=7$ and then $8+4=12$.</think>\n"
        "Therefore the result is complete.\nAnswer: 12"
    )

    assert utils.extract_answer(response) == "12"


def test_extract_answer_accepts_terminal_equation():
    response = "<think>Long proof with $x=3$.</think>\nSum = 21 + 49 = 70"

    assert utils.extract_answer(response) == "70"


def test_extract_answer_rejects_reasoning_without_final_answer():
    response = "<think>We considered $12$ and $13$ but ran out of tokens"

    assert utils.extract_answer(response) == ""


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
