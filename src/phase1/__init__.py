def __getattr__(name):
    if name == "DataFusion":
        from .fusion import DataFusion
        return DataFusion
    elif name == "FairnessAssurance":
        from .fairness import FairnessAssurance
        return FairnessAssurance
    elif name == "PrivacyAssurance":
        from .privacy import PrivacyAssurance
        return PrivacyAssurance
    elif name == "SourcePreprocessor":
        from .preprocessing import SourcePreprocessor
        return SourcePreprocessor
    elif name == "DataPreparation":
        from .phase1_runner import DataPreparation
        return DataPreparation
    raise AttributeError(f"module 'src.phase1' has no attribute {name}")
