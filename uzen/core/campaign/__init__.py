"""N6 autonomous research campaigns.

Public surface:

* :class:`CampaignSpec` / :class:`Axis` / :class:`Objective` -- immutable search
  definitions.
* :class:`CampaignEngine` -- deterministic batch scheduler with SQLite resume.
* :class:`BenchmarkCorpus` -- content-addressed replay and verification.
* :class:`DistributedTrialExecutor` / :class:`CampaignWorker` -- N7 trials over the
  N4 control-plane queue.
* :func:`submit_campaign_run` / :func:`campaign_run_request` -- N8 durable background
  campaign jobs, plus cancellation through :meth:`CampaignEngine.cancel`.
"""

from uzen.core.campaign.corpus import BenchmarkCorpus, CORPUS_SCHEMA, ENGINE_VERSION
from uzen.core.campaign.control import (
    CAMPAIGN_RUN_CANDIDATE,
    campaign_run_request,
    submit_campaign_run,
)
from uzen.core.campaign.distributed import (
    CAMPAIGN_RUN_KIND,
    CAMPAIGN_TRIAL_KIND,
    CampaignWorker,
    DistributedTrialExecutor,
    campaign_trial_request,
)
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
from uzen.errors import CampaignCancelled, CampaignTimeout

__all__ = [
    "ARCHITECTURE_MODE",
    "Axis",
    "BenchmarkCorpus",
    "CAMPAIGN_RUN_CANDIDATE",
    "CAMPAIGN_RUN_KIND",
    "CAMPAIGN_TRIAL_KIND",
    "CORPUS_SCHEMA",
    "CampaignCancelled",
    "CampaignEngine",
    "CampaignReport",
    "CampaignSpec",
    "CampaignStore",
    "CampaignTimeout",
    "CampaignWorker",
    "Candidate",
    "DistributedTrialExecutor",
    "campaign_run_request",
    "campaign_trial_request",
    "submit_campaign_run",
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
