from .heuristic_filter import HeuristicFilter
from .minhash_deduplication import MinHashDeduplication
from .pii_censoring import PIICensoring
from .pipeline_step import PipelineStep
from .step_report import StepReport
from .url_deduplication import URLDeduplication
from typing import Dict, Type

STEP_REGISTRY:Dict[str, Type[PipelineStep]] = {
    "URLDeduplication": URLDeduplication,
    "MinHashDeduplication": MinHashDeduplication,
    "PIICensoring": PIICensoring,
    "HeuristicFilter": HeuristicFilter
}

__all__ = ["PipelineStep", "StepReport", "URLDeduplication", "MinHashDeduplication", "PIICensoring", "HeuristicFilter", "STEP_REGISTRY"]