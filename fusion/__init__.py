"""
fusion/ — Late fusion meta-learner for CDT embedding.

Combines four independent modality embeddings into a single consumer
behavioural embedding (CDT embedding). Encoders are frozen during
fusion training.
"""

from fusion.meta_learner import LateFusionMetaLearner

__all__ = ["LateFusionMetaLearner"]
