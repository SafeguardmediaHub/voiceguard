"""
input_randomization.py — VoiceGuard V8 input randomization defense

Optional, opt-in defense that perturbs audio chunks before scoring to disrupt
gradient-aligned adversarial perturbations (FGSM/PGD-style attacks).

KEY DESIGN POINT — this module is OFF BY DEFAULT. When `enabled=False` (the
default), `randomize_chunk()` returns the input unchanged, making V8 scoring
fully deterministic and identical to the original server.py behavior.

When enabled, the operator chooses a mode:
  - 'gaussian':     add small white noise (cheapest, most common)
  - 'crop_pad':     random crop + zero pad (geometric defense)
  - 'multi_sample': average N independent gaussian-perturbed scores (slower but
                    more stable; recommended when consistency matters)

Honest scope notes:
  - Input randomization is a PROBABILISTIC defense. Adaptive attackers can
    expectation-over-transformation around it (Athalye et al. 2018). It is
    NOT a complete solution to adversarial robustness — it raises the cost
    of attacks but doesn't make V8 fully robust.
  - All three modes affect clean accuracy slightly. Operator must accept
    a small (typically 0.5-2pp) clean-EER regression in exchange for the
    adversarial robustness gain.
  - The 'multi_sample' mode is 5-10x slower than single-pass. Use only when
    score stability and reproducibility matter.

Typical operator workflow:
  - Default deployment: enabled=False (no change from V7 behavior)
  - Security-sensitive deployment: enabled=True, mode='gaussian', sigma=0.002
  - High-stakes deployment: enabled=True, mode='multi_sample', n_samples=5
"""
import torch
from typing import List, Optional


# ── Defaults ────────────────────────────────────────────────────────────────
# These are sensible starting values. Operator can override at call time.
DEFAULT_SIGMA = 0.002       # gaussian noise stddev — well below audible threshold
DEFAULT_CROP_FACTOR = 0.95  # crop_pad: keep 95% of samples (5% zero-padded)
DEFAULT_N_SAMPLES = 5       # multi_sample: average over 5 randomized scores


def randomize_chunk(
    chunk: torch.Tensor,
    enabled: bool = False,
    mode: str = 'gaussian',
    sigma: float = DEFAULT_SIGMA,
    crop_factor: float = DEFAULT_CROP_FACTOR,
    n_samples: int = DEFAULT_N_SAMPLES,
    seed: Optional[int] = None,
) -> List[torch.Tensor]:
    """Apply input randomization to an audio chunk before scoring.

    Args:
        chunk: Audio chunk tensor, shape (1, 1, T) or (1, T).
        enabled: Master switch. When False, returns [chunk] unchanged.
        mode: 'gaussian', 'crop_pad', or 'multi_sample'.
        sigma: Standard deviation for gaussian noise (modes: gaussian, multi_sample).
        crop_factor: Fraction of samples kept when crop_pad mode is used.
        n_samples: Number of randomized samples for multi_sample mode.
        seed: Optional torch seed for reproducible randomization (testing only).

    Returns:
        A list of one or more perturbed chunks. Caller scores each and averages.
        Length is 1 for all modes EXCEPT 'multi_sample' which returns N chunks.

    Examples:
        # Defense disabled — identical to original server.py
        chunks = randomize_chunk(audio, enabled=False)
        assert chunks == [audio]

        # Single gaussian-perturbed chunk
        chunks = randomize_chunk(audio, enabled=True, mode='gaussian')

        # Multi-sample averaging (more stable)
        chunks = randomize_chunk(audio, enabled=True, mode='multi_sample',
                                  n_samples=5)
    """
    # Defense disabled — pass through unchanged for full backward compatibility
    if not enabled:
        return [chunk]

    if seed is not None:
        torch.manual_seed(seed)

    if mode == 'gaussian':
        noise = torch.randn_like(chunk) * sigma
        perturbed = (chunk + noise).clamp(-1.0, 1.0)
        return [perturbed]

    elif mode == 'crop_pad':
        # Random crop then zero-pad back to original length
        T = chunk.shape[-1]
        crop_len = int(T * crop_factor)
        start_max = T - crop_len
        start = torch.randint(0, start_max + 1, (1,)).item()
        end = start + crop_len

        # Build cropped + padded tensor with same shape as input
        out = torch.zeros_like(chunk)
        if chunk.dim() == 3:
            out[..., start:end] = chunk[..., start:end]
        else:
            out[..., start:end] = chunk[..., start:end]
        return [out]

    elif mode == 'multi_sample':
        # Generate N independent gaussian-perturbed chunks; caller averages scores
        out = []
        for _ in range(n_samples):
            noise = torch.randn_like(chunk) * sigma
            perturbed = (chunk + noise).clamp(-1.0, 1.0)
            out.append(perturbed)
        return out

    else:
        raise ValueError(
            f"Unknown randomization mode: '{mode}'. "
            f"Valid modes: 'gaussian', 'crop_pad', 'multi_sample'"
        )


def score_chunks_and_aggregate(score_fn, chunks: List[torch.Tensor]) -> float:
    """Helper for the common pattern: score each randomized chunk, then aggregate.

    Args:
        score_fn: A callable that takes a chunk tensor and returns a scalar score.
        chunks: List of (possibly-randomized) chunks.

    Returns:
        Mean of per-chunk scores. For single-chunk lists this is just the score.

    Notes:
        For multi_sample mode, this gives an averaged, more stable score.
        For single-chunk modes (gaussian, crop_pad, disabled), returns the
        single score with no change in semantics.
    """
    if len(chunks) == 1:
        return score_fn(chunks[0])

    scores = [score_fn(c) for c in chunks]
    return sum(scores) / len(scores)
