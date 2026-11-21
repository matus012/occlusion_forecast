"""otraj — occlusion-aware trajectory prediction on Argoverse 2 (N1).

Terminology invariant (D-N1-1): OCCLUSION masking here = eval/train-time hidden
observation history. Forecast-MAE's PRETEXT masking (MAE pretraining objective)
is a different mechanism — the two terms are kept disjoint in all code and docs.
"""

__version__ = "0.1.0"
