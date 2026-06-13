"""Campaign encoder — self-attention over campaign interaction sequences."""

from encoders.campaign.model import CampaignEncoder
from encoders.campaign.features import CampaignVocabulary, TOKEN_DIM

__all__ = ["CampaignEncoder", "CampaignVocabulary", "TOKEN_DIM"]
