def __getattr__(name):
    if name == "Validator":
        from .validation import Validator
        return Validator
    elif name == "MembershipInferenceAttack":
        from .mia import MembershipInferenceAttack
        return MembershipInferenceAttack
    elif name == "SpatialFidelityEvaluator":
        from .spatial_fidelity import SpatialFidelityEvaluator
        return SpatialFidelityEvaluator
    elif name == "AblationRunner":
        from .ablation import AblationRunner
        return AblationRunner
    elif name == "FeedbackLoop":
        from .feedback import FeedbackLoop
        return FeedbackLoop
    raise AttributeError(f"module 'src.phase3' has no attribute {name}")
