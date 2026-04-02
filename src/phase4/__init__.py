def __getattr__(name):
    if name == "GWFAExplainer":
        from .gwfa import GWFAExplainer
        return GWFAExplainer
    elif name == "CSDExplainer":
        from .csd import CSDExplainer
        return CSDExplainer
    elif name == "AttributionDivergenceAnalyzer":
        from .attribution_divergence import AttributionDivergenceAnalyzer
        return AttributionDivergenceAnalyzer
    elif name == "AccountabilityDocGenerator":
        from .accountability import AccountabilityDocGenerator
        return AccountabilityDocGenerator
    elif name == "Explainer":
        from .phase4_runner import Explainer
        return Explainer
    raise AttributeError(f"module 'src.phase4' has no attribute {name}")
