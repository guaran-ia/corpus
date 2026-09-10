from src.pipeline.integration import Experiment

#This is a smoke test of the experiment, to check everything runs correc
non_synthetic_corpora_test_experiment = Experiment.from_yaml(
    "src/experiments/heuristics_filtering/sample_experiments/experiment_configs/non_synthetic_corpora.yaml"
)

report = non_synthetic_corpora_test_experiment.execute()

print(report)