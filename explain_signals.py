"""explain_signals.py — pure explainability helpers (no torch / no model imports).

Fast-tier testable: confidence, chain-of-custody id, and the SHAP contribution mapping.
"""
import uuid
from statistics import pstdev


def new_audit_id():
    """'aud_' + uuid4().hex — unique per detection, for chain of custody."""
    return "aud_" + uuid.uuid4().hex


def compute_confidence(chunk_scores, final_score):
    """0-1 confidence = decisiveness * agreement (rounded 3dp).
      decisiveness = |final_score - 0.5| * 2                  (0 at the fence, 1 at extremes)
      agreement    = 1 - min(1.0, pstdev(chunk_scores) / 0.5) (1 when chunks agree)
    Empty chunk_scores -> 0.0; a single chunk -> stdev 0 -> agreement 1."""
    if not chunk_scores:
        return 0.0
    decisiveness = abs(final_score - 0.5) * 2.0
    spread = pstdev(chunk_scores) if len(chunk_scores) > 1 else 0.0
    agreement = 1.0 - min(1.0, spread / 0.5)
    return round(decisiveness * agreement, 3)


def shap_from_contribs(contribs_row):
    """Map XGBoost pred_contribs output [c_aasist, c_wav2vec, c_rawnet, bias] to a named
    dict. Feature order matches detector.predict_ensemble's [aasist, wav2vec, rawnet]."""
    return {
        "aasist":  round(float(contribs_row[0]), 4),
        "wav2vec": round(float(contribs_row[1]), 4),
        "rawnet":  round(float(contribs_row[2]), 4),
        "base":    round(float(contribs_row[3]), 4),
    }
