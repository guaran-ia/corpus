import os
from ..pipeline_step import PipelineStep
from ..utils.loader import load_jsonl
from ..utils.writer import JSONLWriter
from ..step_report import StepReport
from datetime import datetime
import json
from tqdm import tqdm

class HeuristicsFilter(PipelineStep):
    def __init__(self):
        self.step_report = StepReport(step_name=self.name)
        self.step_report.step_stats = {
            "duplicate_documents":0,
            "non_duplicate_documents":0,
            "unique_clusters":0
        }

class RemoveIDs(HeuristicsFilter):
    ""
    pass

class RemoveConditional(HeuristicsFilter):
    """Removes all documents based on a condition"""
    pass