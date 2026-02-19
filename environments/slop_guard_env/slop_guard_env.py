"""RL environment for anti-slop prose generation using slop-guard as the reward signal.

Uses Eric Tramel's slop-guard (https://github.com/eric-tramel/slop-guard) — a
rule-based prose linter that scores text on a continuous 0-100 scale via
exponential decay over violation density. This produces a smooth, deterministic
reward suitable for RL training, unlike coarse integer rubrics.

The environment pairs slop-guard scoring with a length gate to prevent
degenerate short outputs from gaming the reward.
"""

from __future__ import annotations

import verifiers as vf
from datasets import Dataset

from slop_guard import HYPERPARAMETERS, Hyperparameters, _analyze

# ---------------------------------------------------------------------------
# Dataset: creative writing prompts
# ---------------------------------------------------------------------------

# Diverse prose-generation prompts spanning narrative, expository, and
# argumentative styles. The model must produce substantial prose (not lists,
# not one-liners) to score well on both the slop-guard and length components.
_PROMPTS = [
    "Write a short essay about what makes a city feel alive at night.",
    "Describe the experience of learning to cook a difficult dish for the first time.",
    "Write about a time when a small decision had unexpected consequences.",
    "Explain how a particular technology changed an industry, without using buzzwords.",
    "Describe the feeling of returning to a place you haven't visited in years.",
    "Write a character sketch of someone who works a job most people overlook.",
    "Explain a complex scientific concept to a curious twelve-year-old.",
    "Write about the difference between being alone and being lonely.",
    "Describe a landscape in a way that reveals something about the person observing it.",
    "Write about an ordinary object and why it matters to someone.",
    "Explain what makes a particular piece of music memorable, without resorting to clichés.",
    "Write about a mistake that taught you more than any success.",
    "Describe the atmosphere of a specific type of weather without naming it directly.",
    "Write about how a neighborhood changes over a decade.",
    "Explain a historical event from the perspective of someone who lived through it.",
    "Write about the experience of reading a book that changed how you think.",
    "Describe the tension in a room where two people disagree but neither speaks.",
    "Write about what happens when a routine is suddenly broken.",
    "Explain why a particular craft or trade deserves more respect.",
    "Write about a conversation that stayed with you long after it ended.",
]


def _build_dataset() -> Dataset:
    """Build the prompt dataset in verifiers format."""
    return Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": prompt}],
                "answer": "",
            }
            for prompt in _PROMPTS
        ]
    )


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

# Minimum word count to receive any slop-guard reward. Below this threshold
# the text is too short for meaningful pattern analysis.
_MIN_WORDS = 50

# Target word count at which the length ramp reaches 1.0.
_TARGET_WORDS = 300


async def slop_guard_reward(completion, parser) -> float:
    """Score the completion using slop-guard's continuous exponential-decay metric.

    Returns 0.0-1.0 where 1.0 = clean prose (score 100) and 0.0 = saturated slop.
    Completions under _MIN_WORDS words return 0.0 to discourage trivially short
    outputs that sidestep pattern detection.
    """
    text = parser.parse_answer(completion)
    word_count = len(text.split())
    if word_count < _MIN_WORDS:
        return 0.0
    result = _analyze(text, HYPERPARAMETERS)
    return result["score"] / 100.0


async def length_reward(completion, parser) -> float:
    """Ramp reward from 0.0 to 1.0 as word count approaches _TARGET_WORDS.

    Prevents the model from gaming slop-guard by writing the shortest possible
    response. The ramp is linear: 0 words -> 0.0, _TARGET_WORDS words -> 1.0,
    beyond _TARGET_WORDS -> 1.0 (capped).
    """
    text = parser.parse_answer(completion)
    word_count = len(text.split())
    return min(word_count / _TARGET_WORDS, 1.0)


# ---------------------------------------------------------------------------
# Environment entrypoint
# ---------------------------------------------------------------------------


def load_environment(
    use_think: bool = False,
    slop_weight: float = 0.8,
    length_weight: float = 0.2,
) -> vf.SingleTurnEnv:
    """Load the slop-guard RL environment.

    Args:
        use_think: If True, use ThinkParser to support chain-of-thought before
            the final answer. The slop-guard reward is computed only on the
            answer portion (after </think>).
        slop_weight: Weight for the slop-guard reward component (default 0.8).
        length_weight: Weight for the length reward component (default 0.2).

    Returns:
        A SingleTurnEnv configured with slop-guard as the primary reward signal.
    """
    dataset = _build_dataset()

    def extract_text(completion):
        try:
            return completion[-1]["content"]
        except Exception:
            return str(completion)

    parser = vf.ThinkParser(extract_text) if use_think else vf.Parser(extract_text)

    rubric = vf.Rubric(
        funcs=[slop_guard_reward, length_reward],
        weights=[slop_weight, length_weight],
    )

    return vf.SingleTurnEnv(
        dataset=dataset,
        parser=parser,
        rubric=rubric,
    )
