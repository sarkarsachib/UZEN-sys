"""N6 autonomous research campaigns.

Public surface:

* :class:`CampaignSpec` / :class:`Axis` / :class:`Objective` -- immutable search
  definitions.
* :class:`CampaignEngine` -- deterministic batch scheduler with SQLite resume.
* :class:`BenchmarkCorpus` -- content-addressed replay and verification.
"""

from uzen.core.campaign.corpus import BenchmarkCorpus, CORPUS_SCHEMA, ENGINE_VERSION
from uzen.core.campaign.engine import CampaignEngine, dominates, pareto_front, score_trials
from uzen.core.campaign.evaluate import Evaluator, evaluate_candidate
from uzen.core.campaign.models import CampaignReport, Candidate, Trial
from uzen.core.campaign.spec import (
    ARCHITECTURE_MODE,
    KNOWN_METRICS,
    WEIGHT_MODE,
    Axis,
    CampaignSpec,
    Objective,
)
from uzen.core.campaign.store import CampaignStore
from uzen.core.campaign.transforms import apply_transforms

__all__ = [
    "ARCHITECTURE_MODE",
    "Axis",
    "BenchmarkCorpus",
    "CORPUS_SCHEMA",
    "CampaignEngine",
    "CampaignReport",
    "CampaignSpec",
    "CampaignStore",
    "Candidate",
    "ENGINE_VERSION",
    "Evaluator",
    "KNOWN_METRICS",
    "Objective",
    "Trial",
    "WEIGHT_MODE",
    "apply_transforms",
    "dominates",
    "evaluate_candidate",
    "pareto_front",
    "score_trials",
]
