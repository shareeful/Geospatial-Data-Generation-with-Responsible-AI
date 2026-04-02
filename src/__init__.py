def __getattr__(name):
    if name == "DataPreparation":
        from .phase1 import DataPreparation
        return DataPreparation
    elif name == "CGANTrainer":
        from .phase2 import CGANTrainer
        return CGANTrainer
    elif name == "Validator":
        from .phase3 import Validator
        return Validator
    elif name == "Explainer":
        from .phase4 import Explainer
        return Explainer
    elif name == "SpatialMetrics":
        from .metrics import SpatialMetrics
        return SpatialMetrics
    elif name == "ClassificationMetrics":
        from .metrics import ClassificationMetrics
        return ClassificationMetrics
    raise AttributeError(f"module 'src' has no attribute {name}")
