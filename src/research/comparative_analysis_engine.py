"""Comparative Analysis Engine for Medical AI Research.

Implements automated comparative studies between different AI models
and approaches for medical image analysis.
"""

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformance:
    """Container for model performance metrics."""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    inference_time: float
    training_time: Optional[float] = None
    memory_usage: Optional[float] = None


@dataclass 
class ComparativeStudyConfig:
    """Configuration for comparative studies."""
    study_name: str
    models_to_compare: List[str]
    metrics: List[str]
    significance_threshold: float = 0.05
    num_runs: int = 3
    output_dir: str = "comparative_studies"


class ComparativeAnalysisEngine:
    """Engine for conducting automated comparative studies."""
    
    def __init__(self, config: ComparativeStudyConfig):
        self.config = config
        self.results: List[Dict[str, Any]] = []
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def run_comparative_study(self, 
                            model_configs: List[Dict[str, Any]],
                            test_data: Any) -> Dict[str, Any]:
        """Run comprehensive comparative study."""
        logger.info(f"Starting comparative study: {self.config.study_name}")
        
        study_results = {
            "study_id": f"study_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "study_name": self.config.study_name,
            "timestamp": datetime.now().isoformat(),
            "models": [],
            "statistical_analysis": {},
            "recommendations": []
        }
        
        # Run experiments for each model
        all_performances = []
        for model_config in model_configs:
            performances = self._run_model_experiments(model_config, test_data)
            all_performances.extend(performances)
            
            # Calculate aggregate metrics
            avg_performance = self._aggregate_performance(performances)
            study_results["models"].append({
                "model_name": model_config["name"],
                "config": model_config,
                "performance": avg_performance.__dict__,
                "runs": [p.__dict__ for p in performances]
            })
            
        # Conduct statistical analysis
        study_results["statistical_analysis"] = self._conduct_statistical_analysis(
            all_performances
        )
        
        # Generate recommendations
        study_results["recommendations"] = self._generate_recommendations(
            study_results
        )
        
        # Save results
        self._save_study_results(study_results)
        
        return study_results
    
    def _run_model_experiments(self, 
                              model_config: Dict[str, Any], 
                              test_data: Any) -> List[ModelPerformance]:
        """Run multiple experiments for a single model."""
        performances = []
        
        for run_idx in range(self.config.num_runs):
            logger.info(f"Running experiment {run_idx + 1}/{self.config.num_runs} "
                       f"for {model_config['name']}")
            
            # Simulate model training and evaluation
            # In real implementation, this would load/train actual models
            performance = self._simulate_model_performance(
                model_config, test_data, run_idx
            )
            
            performances.append(performance)
            
        return performances
    
    def _simulate_model_performance(self, 
                                   model_config: Dict[str, Any],
                                   test_data: Any,
                                   run_idx: int) -> ModelPerformance:
        """Simulate model performance for demonstration."""
        # This would be replaced with actual model evaluation in production
        base_performance = model_config.get("base_performance", {
            "accuracy": 0.85,
            "precision": 0.83,
            "recall": 0.87,
            "f1_score": 0.85,
            "auc_roc": 0.89
        })
        
        # Add some realistic noise
        np.random.seed(42 + run_idx)
        noise_factor = 0.02
        
        return ModelPerformance(
            model_name=model_config["name"],
            accuracy=base_performance["accuracy"] + np.random.normal(0, noise_factor),
            precision=base_performance["precision"] + np.random.normal(0, noise_factor),
            recall=base_performance["recall"] + np.random.normal(0, noise_factor),
            f1_score=base_performance["f1_score"] + np.random.normal(0, noise_factor),
            auc_roc=base_performance["auc_roc"] + np.random.normal(0, noise_factor),
            inference_time=np.random.uniform(0.1, 2.0),
            training_time=np.random.uniform(300, 3600),
            memory_usage=np.random.uniform(1000, 8000)
        )
    
    def _aggregate_performance(self, 
                              performances: List[ModelPerformance]) -> ModelPerformance:
        """Aggregate performance across multiple runs."""
        if not performances:
            raise ValueError("No performances to aggregate")
            
        return ModelPerformance(
            model_name=performances[0].model_name,
            accuracy=np.mean([p.accuracy for p in performances]),
            precision=np.mean([p.precision for p in performances]),
            recall=np.mean([p.recall for p in performances]),
            f1_score=np.mean([p.f1_score for p in performances]),
            auc_roc=np.mean([p.auc_roc for p in performances]),
            inference_time=np.mean([p.inference_time for p in performances]),
            training_time=np.mean([p.training_time for p in performances if p.training_time]),
            memory_usage=np.mean([p.memory_usage for p in performances if p.memory_usage])
        )
    
    def _conduct_statistical_analysis(self, 
                                    performances: List[ModelPerformance]) -> Dict[str, Any]:
        """Conduct statistical significance testing."""
        # Group performances by model
        model_groups = {}
        for perf in performances:
            if perf.model_name not in model_groups:
                model_groups[perf.model_name] = []
            model_groups[perf.model_name].append(perf)
        
        analysis = {
            "model_comparison": {},
            "significance_tests": {},
            "effect_sizes": {}
        }
        
        models = list(model_groups.keys())
        
        # Pairwise comparisons
        for i, model_a in enumerate(models):
            for model_b in models[i+1:]:
                key = f"{model_a}_vs_{model_b}"
                
                # Get accuracy scores for comparison
                scores_a = [p.accuracy for p in model_groups[model_a]]
                scores_b = [p.accuracy for p in model_groups[model_b]]
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(scores_a, scores_b)
                
                # Calculate effect size (Cohen's d)
                pooled_std = np.sqrt(((np.std(scores_a, ddof=1) ** 2) + 
                                    (np.std(scores_b, ddof=1) ** 2)) / 2)
                effect_size = (np.mean(scores_a) - np.mean(scores_b)) / pooled_std
                
                analysis["significance_tests"][key] = {
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "significant": p_value < self.config.significance_threshold
                }
                
                analysis["effect_sizes"][key] = {
                    "cohens_d": float(effect_size),
                    "magnitude": self._interpret_effect_size(effect_size)
                }
        
        return analysis
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _generate_recommendations(self, study_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on results."""
        recommendations = []
        
        # Find best performing model
        best_model = max(study_results["models"], 
                        key=lambda m: m["performance"]["accuracy"])
        
        recommendations.append(
            f"Best performing model: {best_model['model_name']} "
            f"(Accuracy: {best_model['performance']['accuracy']:.3f})"
        )
        
        # Check for significant differences
        sig_tests = study_results["statistical_analysis"]["significance_tests"]
        significant_differences = [k for k, v in sig_tests.items() if v["significant"]]
        
        if significant_differences:
            recommendations.append(
                f"Found {len(significant_differences)} statistically significant "
                "performance differences between models"
            )
        else:
            recommendations.append(
                "No statistically significant differences found between models"
            )
        
        # Performance vs efficiency trade-offs
        fastest_model = min(study_results["models"], 
                          key=lambda m: m["performance"]["inference_time"])
        
        if fastest_model["model_name"] != best_model["model_name"]:
            recommendations.append(
                f"For real-time applications, consider {fastest_model['model_name']} "
                f"(Inference time: {fastest_model['performance']['inference_time']:.3f}s)"
            )
        
        return recommendations
    
    def _save_study_results(self, study_results: Dict[str, Any]):
        """Save study results to JSON and generate visualizations."""
        # Save JSON results
        results_file = self.output_dir / f"{study_results['study_id']}_results.json"
        with open(results_file, 'w') as f:
            json.dump(study_results, f, indent=2, default=str)
        
        # Generate visualizations
        self._generate_comparison_plots(study_results)
        
        logger.info(f"Study results saved to {results_file}")
    
    def _generate_comparison_plots(self, study_results: Dict[str, Any]):
        """Generate comparison visualizations."""
        models_data = study_results["models"]
        
        # Performance comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Model Comparison: {study_results['study_name']}")
        
        metrics = ["accuracy", "precision", "recall", "f1_score"]
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            model_names = [m["model_name"] for m in models_data]
            values = [m["performance"][metric] for m in models_data]
            
            bars = ax.bar(model_names, values)
            ax.set_title(f"{metric.replace('_', ' ').title()}")
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plot_file = self.output_dir / f"{study_results['study_id']}_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comparison plots saved to {plot_file}")


def run_example_study():
    """Run an example comparative study."""
    config = ComparativeStudyConfig(
        study_name="CNN vs Transfer Learning Comparison",
        models_to_compare=["Simple_CNN", "VGG16_Transfer", "ResNet50_Transfer"],
        metrics=["accuracy", "precision", "recall", "f1_score", "auc_roc"],
        num_runs=5
    )
    
    engine = ComparativeAnalysisEngine(config)
    
    # Define model configurations for comparison
    model_configs = [
        {
            "name": "Simple_CNN",
            "type": "custom",
            "base_performance": {
                "accuracy": 0.82, "precision": 0.80, "recall": 0.84,
                "f1_score": 0.82, "auc_roc": 0.86
            }
        },
        {
            "name": "VGG16_Transfer",
            "type": "transfer_learning",
            "base_performance": {
                "accuracy": 0.89, "precision": 0.87, "recall": 0.91,
                "f1_score": 0.89, "auc_roc": 0.93
            }
        },
        {
            "name": "ResNet50_Transfer", 
            "type": "transfer_learning",
            "base_performance": {
                "accuracy": 0.91, "precision": 0.89, "recall": 0.93,
                "f1_score": 0.91, "auc_roc": 0.95
            }
        }
    ]
    
    # Run comparative study
    results = engine.run_comparative_study(model_configs, test_data=None)
    
    print(f"Comparative study completed: {results['study_id']}")
    print("\nRecommendations:")
    for rec in results["recommendations"]:
        print(f"- {rec}")


if __name__ == "__main__":
    run_example_study()