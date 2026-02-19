# slop-guard-env

RL environment that uses [slop-guard](https://github.com/eric-tramel/slop-guard) by [Eric Tramel](https://eric-tramel.github.io/) as a continuous reward signal for training models to produce less formulaic prose.

### Overview
- **Environment ID**: `slop-guard-env`
- **Type**: single-turn
- **Reward range**: 0.0–1.0 (continuous)
- **Tags**: creative-writing, rule-based, anti-slop

### How it works

The model receives a prose-generation prompt and must produce a substantial written response. The response is scored by two weighted reward components:

1. **slop-guard score** (weight: 0.8) — Runs ~80 compiled regex rules across 6 categories (vocabulary, phrases, structural patterns, tone markers, rhythm, syntactic repetition). Produces a continuous 0–100 score via exponential decay over violation density per 1,000 words, normalized to 0.0–1.0. Violations in "Claude categories" (contrast pairs, pithy fragments, setup-resolution) receive concentration amplification.

2. **length gate** (weight: 0.2) — Linear ramp from 0.0 to 1.0 as word count reaches 300. Prevents the model from gaming slop-guard by writing trivially short outputs.

### Why slop-guard over other approaches

| Property | slop-guard | Coarse regex (e.g. existing antislop env) | LLM judge |
|---|---|---|---|
| Score granularity | Continuous 0–100 | Integer 0–15 | Varies |
| Length normalization | Density per 1k words | Raw hit counts | N/A |
| Concentration penalty | Exponential compounding | None | N/A |
| Determinism | Fully deterministic | Fully deterministic | Non-deterministic |
| Latency | Milliseconds | Milliseconds | Seconds |
| Gameability | Hard (many orthogonal rules) | Easier (few coarse buckets) | Moderate |

### Quickstart

```bash
uv run vf-eval slop_guard_env
```

Configure model and sampling:
```bash
uv run vf-eval slop_guard_env -m gpt-4.1-mini -n 20 -r 3 -t 1024 -T 0.7
```

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `use_think` | bool | `False` | Enable ThinkParser for chain-of-thought; slop-guard scores only the answer portion |
| `slop_weight` | float | `0.8` | Weight for the slop-guard reward component |
| `length_weight` | float | `0.2` | Weight for the length reward component |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `slop_guard_reward` | slop-guard score normalized to 0.0–1.0 (higher = cleaner prose) |
| `length_reward` | Word count ramp, 0.0–1.0 |
| `reward` | Weighted sum of the above |

### Attribution

The slop-guard analysis engine (`slop_guard.py`) is vendored from [eric-tramel/slop-guard](https://github.com/eric-tramel/slop-guard) under the MIT License. See `LICENSE.slop-guard` for the full license text.
