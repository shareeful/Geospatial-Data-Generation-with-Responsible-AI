import numpy as np
from typing import Dict, Optional

from .gwfa import GWFAExplainer
from .csd import CSDExplainer
from .attribution_divergence import AttributionDivergenceAnalyzer
from .accountability import AccountabilityDocGenerator
from ..utils.logging_utils import setup_logger

logger = setup_logger("phase4")


class Explainer:
    def __init__(self, config: Dict):
        self.config = config
        exp_cfg = config.get("explainability", {})

        self.gwfa = GWFAExplainer(
            adaptive=exp_cfg.get("gwfa_adaptive", True),
            default_bandwidth=500.0,
        )
        self.csd = CSDExplainer()
        self.ad_analyzer = AttributionDivergenceAnalyzer(
            threshold=exp_cfg.get("ad_threshold", 0.05),
        )
        self.accountability = AccountabilityDocGenerator(
            output_dir=config.get("output", {}).get("docs_dir", "accountability_docs"),
        )

    def run_explainability(self, X_real: np.ndarray, y_real: np.ndarray,
                            X_synthetic: np.ndarray, y_synthetic: np.ndarray,
                            coordinates: np.ndarray,
                            classifier_real, classifier_synthetic,
                            feature_names: list = None) -> Dict:

        logger.info("=== Phase 4: Explainability and Accountability ===")

        logger.info("Task 4.1a: GWFA maps (real-trained)")
        gwfa_real = self.gwfa.compute_attributions(
            X_real, y_real.astype(float), coordinates,
            feature_names=feature_names,
        )

        logger.info("Task 4.1a: GWFA maps (synthetic-trained)")
        gwfa_synthetic = self.gwfa.compute_attributions(
            X_synthetic, y_synthetic.astype(float), coordinates,
            target_locations=gwfa_real["target_locations"],
            feature_names=feature_names,
        )

        sac_real = self.gwfa.spatial_attribution_consistency(
            gwfa_real["attributions"], gwfa_real["target_locations"]
        )
        sac_synthetic = self.gwfa.spatial_attribution_consistency(
            gwfa_synthetic["attributions"], gwfa_synthetic["target_locations"]
        )

        logger.info("Task 4.1b: CSD maps")
        predictions_real = classifier_real.predict(X_real)
        predictions_syn = classifier_synthetic.predict(X_real)

        csd_real = self.csd.compute_csd(X_real, predictions_real, classifier_real)
        csd_synthetic = self.csd.compute_csd(X_real, predictions_syn, classifier_synthetic)

        suds_locations = self.csd.identify_suds_locations(csd_real, coordinates)

        logger.info("Task 4.1c: Attribution Divergence")
        ad_result = self.ad_analyzer.compute(
            gwfa_real["attributions"], gwfa_synthetic["attributions"]
        )

        csd_comparison = self.ad_analyzer.compare_csd_distributions(csd_real, csd_synthetic)

        return {
            "gwfa_real": gwfa_real,
            "gwfa_synthetic": gwfa_synthetic,
            "sac_real": sac_real,
            "sac_synthetic": sac_synthetic,
            "csd_real": csd_real,
            "csd_synthetic": csd_synthetic,
            "suds_locations": suds_locations,
            "attribution_divergence": ad_result,
            "csd_comparison": csd_comparison,
        }

    def generate_accountability_docs(self, model_info: Dict, performance: Dict,
                                       fairness_info: Dict, dataset_info: Dict,
                                       class_distribution: Dict,
                                       provenance_hashes: list,
                                       pca_info: Dict, privacy_info: Dict) -> Dict[str, str]:

        sample_provenance = [
            {"sample_idx": i, "hash": h, "traceable": True}
            for i, h in enumerate(provenance_hashes[:100])
        ]

        return self.accountability.generate_all(
            model_info=model_info,
            performance=performance,
            fairness_info=fairness_info,
            dataset_info=dataset_info,
            class_distribution=class_distribution,
            provenance_info={"method": "PCA loading matrix + epoch + sample hash"},
            sample_provenance=sample_provenance,
            pca_info=pca_info,
            privacy_info=privacy_info,
        )
