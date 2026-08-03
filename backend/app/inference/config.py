from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class InferenceConfig:
    target_sample_rate: int = 16_000
    fragment_duration_seconds: float = 10.0
    n_mfcc: int = 20
    positive_label: int = 1
    decision_threshold: float = 0.0
    class_names: dict[int, str] = field(
        default_factory=lambda: {0: "human", 1: "ai_generated"}
    )
    model_id: str = "mfcc-svm-baseline"
    aggregation_strategy: str = "mean_decision_score"
    final_fragment_policy: str = "pad"
    usage_warning: str = (
        "La salida es una estimacion; el score no es una probabilidad calibrada "
        "y el resultado no constituye un veredicto forense."
    )

    @property
    def target_samples(self) -> int:
        return int(round(self.target_sample_rate * self.fragment_duration_seconds))
