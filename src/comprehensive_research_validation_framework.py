"""
Comprehensive Research Validation Framework
==========================================

Publication-ready research validation system providing:
- Rigorous statistical testing
- Reproducible benchmark protocols
- Cross-validation frameworks
- Performance comparison studies
- Clinical validation metrics
- Peer-review quality documentation

Key Features:
1. Multi-domain benchmark suite
2. Statistical significance testing
3. Cross-validation protocols
4. Clinical performance metrics
5. Comparative effectiveness research
6. Reproducibility validation
7. Publication-ready documentation
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import pickle
from datetime import datetime
import warnings

# Statistical testing
from scipy import stats
from scipy.stats import ttest_ind, wilcoxon, mannwhitneyu, chi2_contingency
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.power import ttest_power
import statsmodels.api as sm

# Machine learning validation
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score, 
    roc_curve, precision_recall_curve, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Multiple testing correction
from statsmodels.stats.multitest import multipletests


@dataclass
class BenchmarkConfiguration:
    """Configuration for benchmark studies."""
    name: str
    description: str
    algorithms: List[str]
    metrics: List[str]
    dataset_sizes: List[int]
    cross_validation_folds: int = 5
    statistical_tests: List[str] = field(default_factory=lambda: ['ttest', 'wilcoxon', 'mannwhitneyu'])
    significance_level: float = 0.05
    multiple_testing_correction: str = 'bonferroni'
    random_seed: int = 42
    bootstrap_iterations: int = 1000


@dataclass
class ValidationResult:
    """Comprehensive validation result."""
    algorithm_name: str
    dataset_name: str
    metrics: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    statistical_tests: Dict[str, Dict[str, float]]
    cross_validation_scores: Dict[str, List[float]]
    training_time: float
    inference_time: float
    memory_usage: float
    reproducibility_score: float
    clinical_relevance_score: float


@dataclass
class ComparativeStudyResults:
    """Results from comparative research study."""
    study_name: str
    algorithms_compared: List[str]
    datasets_used: List[str]
    validation_results: List[ValidationResult]
    statistical_summary: Dict[str, Any]
    effect_sizes: Dict[str, Dict[str, float]]
    power_analysis: Dict[str, float]
    clinical_significance: Dict[str, Any]
    recommendations: List[str]
    limitations: List[str]


class StatisticalTestSuite:
    """Comprehensive statistical testing suite for medical AI validation."""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.logger = logging.getLogger(__name__)
        
    def compare_algorithm_performance(self, 
                                    results1: List[float], 
                                    results2: List[float],
                                    algorithm1_name: str = "Algorithm A",
                                    algorithm2_name: str = "Algorithm B",
                                    metric_name: str = "Performance") -> Dict[str, Any]:
        """Compare performance between two algorithms with multiple statistical tests."""
        
        # Ensure equal length
        min_length = min(len(results1), len(results2))
        results1 = results1[:min_length]
        results2 = results2[:min_length]
        
        comparison_results = {
            'algorithm1_name': algorithm1_name,
            'algorithm2_name': algorithm2_name,
            'metric_name': metric_name,
            'sample_size': min_length,
            'descriptive_stats': {},
            'statistical_tests': {},
            'effect_size': {},
            'confidence_intervals': {},
            'interpretation': {}
        }
        
        # Descriptive statistics
        comparison_results['descriptive_stats'] = {
            f'{algorithm1_name}_mean': np.mean(results1),
            f'{algorithm1_name}_std': np.std(results1),
            f'{algorithm1_name}_median': np.median(results1),
            f'{algorithm2_name}_mean': np.mean(results2),
            f'{algorithm2_name}_std': np.std(results2),
            f'{algorithm2_name}_median': np.median(results2),
            'mean_difference': np.mean(results1) - np.mean(results2)
        }
        
        # Normality tests
        shapiro1 = stats.shapiro(results1)
        shapiro2 = stats.shapiro(results2)
        
        comparison_results['normality_tests'] = {
            f'{algorithm1_name}_shapiro_p': shapiro1.pvalue,
            f'{algorithm2_name}_shapiro_p': shapiro2.pvalue,
            'both_normal': shapiro1.pvalue > 0.05 and shapiro2.pvalue > 0.05
        }
        
        # Statistical tests
        
        # 1. Paired t-test (if data is paired and normal)
        if comparison_results['normality_tests']['both_normal']:
            try:
                ttest_stat, ttest_p = ttest_ind(results1, results2)
                comparison_results['statistical_tests']['ttest'] = {
                    'statistic': ttest_stat,
                    'p_value': ttest_p,
                    'significant': ttest_p < self.significance_level
                }
            except Exception as e:
                self.logger.warning(f"T-test failed: {e}")
        
        # 2. Mann-Whitney U test (non-parametric)
        try:
            mw_stat, mw_p = mannwhitneyu(results1, results2, alternative='two-sided')
            comparison_results['statistical_tests']['mann_whitney'] = {
                'statistic': mw_stat,
                'p_value': mw_p,
                'significant': mw_p < self.significance_level
            }
        except Exception as e:
            self.logger.warning(f"Mann-Whitney U test failed: {e}")
        
        # 3. Wilcoxon signed-rank test (if paired)
        if len(results1) == len(results2):
            try:
                wilcox_stat, wilcox_p = wilcoxon(results1, results2)
                comparison_results['statistical_tests']['wilcoxon'] = {
                    'statistic': wilcox_stat,
                    'p_value': wilcox_p,
                    'significant': wilcox_p < self.significance_level
                }
            except Exception as e:
                self.logger.warning(f"Wilcoxon test failed: {e}")
        
        # Effect size calculations
        
        # Cohen's d
        pooled_std = np.sqrt((np.var(results1) + np.var(results2)) / 2)
        if pooled_std > 0:
            cohens_d = (np.mean(results1) - np.mean(results2)) / pooled_std
            comparison_results['effect_size']['cohens_d'] = cohens_d
            
            # Interpret Cohen's d
            if abs(cohens_d) < 0.2:
                effect_interpretation = "negligible"
            elif abs(cohens_d) < 0.5:
                effect_interpretation = "small"
            elif abs(cohens_d) < 0.8:
                effect_interpretation = "medium"
            else:
                effect_interpretation = "large"
            
            comparison_results['effect_size']['cohens_d_interpretation'] = effect_interpretation
        
        # Confidence intervals for difference in means
        try:
            diff_mean = np.mean(results1) - np.mean(results2)
            diff_se = np.sqrt(np.var(results1)/len(results1) + np.var(results2)/len(results2))
            ci_lower = diff_mean - 1.96 * diff_se
            ci_upper = diff_mean + 1.96 * diff_se
            
            comparison_results['confidence_intervals']['mean_difference_95ci'] = (ci_lower, ci_upper)
        except Exception as e:
            self.logger.warning(f"Confidence interval calculation failed: {e}")
        
        # Clinical significance assessment
        comparison_results['interpretation'] = self._interpret_clinical_significance(
            comparison_results, metric_name
        )
        
        return comparison_results
    
    def _interpret_clinical_significance(self, comparison_results: Dict[str, Any], 
                                       metric_name: str) -> Dict[str, str]:
        """Interpret clinical significance of statistical results."""
        
        interpretation = {}
        
        # Statistical significance
        significant_tests = [
            test for test, results in comparison_results['statistical_tests'].items()
            if results.get('significant', False)
        ]
        
        if significant_tests:
            interpretation['statistical_significance'] = (
                f"Statistically significant difference detected "
                f"(tests: {', '.join(significant_tests)})"
            )
        else:
            interpretation['statistical_significance'] = "No statistically significant difference detected"
        
        # Clinical significance based on metric type and effect size
        cohens_d = comparison_results['effect_size'].get('cohens_d', 0)
        mean_diff = comparison_results['descriptive_stats']['mean_difference']
        
        if 'accuracy' in metric_name.lower():
            if abs(mean_diff) >= 0.05:  # 5% difference in accuracy
                interpretation['clinical_significance'] = "Clinically meaningful difference in accuracy"
            else:
                interpretation['clinical_significance'] = "Difference may not be clinically meaningful"
                
        elif 'auc' in metric_name.lower():
            if abs(mean_diff) >= 0.03:  # 3% difference in AUC
                interpretation['clinical_significance'] = "Clinically meaningful difference in discrimination"
            else:
                interpretation['clinical_significance'] = "Difference may not be clinically meaningful"
                
        elif 'time' in metric_name.lower():
            if abs(mean_diff) >= 0.1:  # 100ms difference
                interpretation['clinical_significance'] = "Clinically meaningful difference in processing time"
            else:
                interpretation['clinical_significance'] = "Difference may not be clinically meaningful"
        
        # Overall recommendation
        if significant_tests and abs(cohens_d) >= 0.5:
            interpretation['recommendation'] = "Strong evidence for algorithm superiority"
        elif significant_tests:
            interpretation['recommendation'] = "Moderate evidence for algorithm difference"
        else:
            interpretation['recommendation'] = "Insufficient evidence for algorithm superiority"
        
        return interpretation
    
    def multiple_comparison_correction(self, p_values: List[float], 
                                     method: str = 'bonferroni') -> Tuple[List[bool], List[float]]:
        """Apply multiple testing correction to p-values."""
        
        try:
            rejected, p_corrected, _, _ = multipletests(
                p_values, alpha=self.significance_level, method=method
            )
            return rejected.tolist(), p_corrected.tolist()
        except Exception as e:
            self.logger.error(f"Multiple testing correction failed: {e}")
            return [False] * len(p_values), p_values
    
    def power_analysis(self, effect_size: float, sample_size: int, 
                      alpha: float = 0.05) -> Dict[str, float]:
        """Perform statistical power analysis."""
        
        try:
            # Calculate power for given parameters
            power = ttest_power(effect_size, sample_size, alpha)
            
            # Calculate required sample size for 80% power
            required_n_80 = sm.stats.tt_solve_power(
                effect_size=effect_size, power=0.8, alpha=alpha
            )
            
            # Calculate required sample size for 90% power
            required_n_90 = sm.stats.tt_solve_power(
                effect_size=effect_size, power=0.9, alpha=alpha
            )
            
            return {
                'current_power': power,
                'sample_size_for_80_power': required_n_80,
                'sample_size_for_90_power': required_n_90,
                'adequate_power': power >= 0.8
            }
        except Exception as e:
            self.logger.error(f"Power analysis failed: {e}")
            return {'current_power': 0.0, 'adequate_power': False}


class CrossValidationFramework:
    """Robust cross-validation framework for medical AI models."""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.logger = logging.getLogger(__name__)
        
    def perform_cross_validation(self, 
                                model: Any,
                                X: np.ndarray, 
                                y: np.ndarray,
                                cv_strategy: str = 'stratified',
                                n_folds: int = 5,
                                metrics: List[str] = None) -> Dict[str, Any]:
        """Perform comprehensive cross-validation with multiple metrics."""
        
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        
        # Select cross-validation strategy
        if cv_strategy == 'stratified':
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_seed)
        elif cv_strategy == 'kfold':
            cv = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_seed)
        elif cv_strategy == 'timeseries':
            cv = TimeSeriesSplit(n_splits=n_folds)
        else:
            raise ValueError(f"Unknown CV strategy: {cv_strategy}")
        
        cv_results = {
            'cv_strategy': cv_strategy,
            'n_folds': n_folds,
            'metrics': {},
            'fold_results': [],
            'training_times': [],
            'inference_times': []
        }
        
        # Initialize metric storage
        for metric in metrics:
            cv_results['metrics'][metric] = []
        
        # Perform cross-validation
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            fold_start_time = time.time()
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            train_start = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - train_start
            
            # Make predictions
            inference_start = time.time()
            y_pred = model.predict(X_val)
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_val)[:, 1]
            else:
                y_proba = y_pred
            inference_time = time.time() - inference_start
            
            # Calculate metrics
            fold_metrics = self._calculate_fold_metrics(y_val, y_pred, y_proba, metrics)
            
            # Store results
            for metric in metrics:
                if metric in fold_metrics:
                    cv_results['metrics'][metric].append(fold_metrics[metric])
            
            cv_results['training_times'].append(training_time)
            cv_results['inference_times'].append(inference_time)
            
            fold_result = {
                'fold': fold_idx + 1,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'metrics': fold_metrics,
                'training_time': training_time,
                'inference_time': inference_time
            }
            cv_results['fold_results'].append(fold_result)
            
            self.logger.debug(f"Fold {fold_idx + 1}/{n_folds} completed in {time.time() - fold_start_time:.2f}s")
        
        # Calculate summary statistics
        cv_results['summary_statistics'] = self._calculate_cv_summary_statistics(cv_results)
        
        return cv_results
    
    def _calculate_fold_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               y_proba: np.ndarray, metrics: List[str]) -> Dict[str, float]:
        """Calculate metrics for a single fold."""
        
        fold_metrics = {}
        
        try:
            if 'accuracy' in metrics:
                fold_metrics['accuracy'] = accuracy_score(y_true, y_pred)
            
            if any(m in metrics for m in ['precision', 'recall', 'f1']):
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, average='binary', zero_division=0
                )
                if 'precision' in metrics:
                    fold_metrics['precision'] = precision
                if 'recall' in metrics:
                    fold_metrics['recall'] = recall
                if 'f1' in metrics:
                    fold_metrics['f1'] = f1
            
            if 'auc' in metrics:
                try:
                    fold_metrics['auc'] = roc_auc_score(y_true, y_proba)
                except ValueError:
                    fold_metrics['auc'] = 0.5  # Default for cases with only one class
            
            if 'sensitivity' in metrics:
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                fold_metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            if 'specificity' in metrics:
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                fold_metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
                
        except Exception as e:
            self.logger.error(f"Error calculating fold metrics: {e}")
            # Return zeros for all metrics if calculation fails
            for metric in metrics:
                fold_metrics[metric] = 0.0
        
        return fold_metrics
    
    def _calculate_cv_summary_statistics(self, cv_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics for cross-validation results."""
        
        summary = {}
        
        for metric, values in cv_results['metrics'].items():
            if values:  # Only calculate if we have values
                summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'q25': np.percentile(values, 25),
                    'q75': np.percentile(values, 75),
                    'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
                }
        
        # Training and inference time statistics
        if cv_results['training_times']:
            summary['training_time'] = {
                'mean': np.mean(cv_results['training_times']),
                'std': np.std(cv_results['training_times'])
            }
        
        if cv_results['inference_times']:
            summary['inference_time'] = {
                'mean': np.mean(cv_results['inference_times']),
                'std': np.std(cv_results['inference_times'])
            }
        
        return summary


class BenchmarkSuite:
    """Comprehensive benchmark suite for medical AI algorithms."""
    
    def __init__(self, config: BenchmarkConfiguration):
        self.config = config
        self.statistical_tester = StatisticalTestSuite(config.significance_level)
        self.cv_framework = CrossValidationFramework(config.random_seed)
        self.logger = logging.getLogger(__name__)
        
        # Set random seeds for reproducibility
        np.random.seed(config.random_seed)
        
    def run_comprehensive_benchmark(self, 
                                  algorithms: Dict[str, Any],
                                  datasets: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> ComparativeStudyResults:
        """Run comprehensive benchmark study across algorithms and datasets."""
        
        study_name = f"{self.config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Starting comprehensive benchmark study: {study_name}")
        
        validation_results = []
        
        # Run benchmarks for each algorithm-dataset combination
        for dataset_name, (X, y) in datasets.items():
            self.logger.info(f"Processing dataset: {dataset_name}")
            
            for algorithm_name, algorithm in algorithms.items():
                self.logger.info(f"  Testing algorithm: {algorithm_name}")
                
                try:
                    result = self._benchmark_algorithm(
                        algorithm, X, y, algorithm_name, dataset_name
                    )
                    validation_results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"Error benchmarking {algorithm_name} on {dataset_name}: {e}")
        
        # Perform statistical analysis
        statistical_summary = self._perform_statistical_analysis(validation_results)
        
        # Calculate effect sizes
        effect_sizes = self._calculate_effect_sizes(validation_results)
        
        # Power analysis
        power_analysis = self._perform_power_analysis(validation_results)
        
        # Clinical significance assessment
        clinical_significance = self._assess_clinical_significance(validation_results)
        
        # Generate recommendations and limitations
        recommendations = self._generate_recommendations(
            validation_results, statistical_summary, clinical_significance
        )
        limitations = self._identify_limitations(validation_results)
        
        study_results = ComparativeStudyResults(
            study_name=study_name,
            algorithms_compared=list(algorithms.keys()),
            datasets_used=list(datasets.keys()),
            validation_results=validation_results,
            statistical_summary=statistical_summary,
            effect_sizes=effect_sizes,
            power_analysis=power_analysis,
            clinical_significance=clinical_significance,
            recommendations=recommendations,
            limitations=limitations
        )
        
        self.logger.info(f"Benchmark study completed: {study_name}")
        
        return study_results
    
    def _benchmark_algorithm(self, algorithm: Any, X: np.ndarray, y: np.ndarray,
                           algorithm_name: str, dataset_name: str) -> ValidationResult:
        """Benchmark a single algorithm on a dataset."""
        
        start_time = time.time()
        
        # Perform cross-validation
        cv_results = self.cv_framework.perform_cross_validation(
            algorithm, X, y, 
            cv_strategy='stratified',
            n_folds=self.config.cross_validation_folds,
            metrics=self.config.metrics
        )
        
        total_time = time.time() - start_time
        
        # Extract performance metrics
        metrics = {}
        confidence_intervals = {}
        
        for metric_name, values in cv_results['metrics'].items():
            if values:
                metrics[metric_name] = np.mean(values)
                
                # Bootstrap confidence intervals
                ci_lower, ci_upper = self._bootstrap_confidence_interval(values)
                confidence_intervals[metric_name] = (ci_lower, ci_upper)
        
        # Calculate reproducibility score
        reproducibility_score = self._calculate_reproducibility_score(cv_results)
        
        # Calculate clinical relevance score
        clinical_relevance_score = self._calculate_clinical_relevance_score(metrics)
        
        # Memory usage estimation (simplified)
        memory_usage = len(str(algorithm)) * 0.001  # Rough estimation
        
        return ValidationResult(
            algorithm_name=algorithm_name,
            dataset_name=dataset_name,
            metrics=metrics,
            confidence_intervals=confidence_intervals,
            statistical_tests={},  # Will be filled in comparative analysis
            cross_validation_scores=cv_results['metrics'],
            training_time=cv_results['summary_statistics'].get('training_time', {}).get('mean', 0),
            inference_time=cv_results['summary_statistics'].get('inference_time', {}).get('mean', 0),
            memory_usage=memory_usage,
            reproducibility_score=reproducibility_score,
            clinical_relevance_score=clinical_relevance_score
        )
    
    def _bootstrap_confidence_interval(self, values: List[float], 
                                     confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval."""
        
        if not values:
            return (0.0, 0.0)
        
        bootstrap_samples = []
        n_bootstrap = self.config.bootstrap_iterations
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=len(values), replace=True)
            bootstrap_samples.append(np.mean(sample))
        
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_samples, lower_percentile)
        ci_upper = np.percentile(bootstrap_samples, upper_percentile)
        
        return ci_lower, ci_upper
    
    def _calculate_reproducibility_score(self, cv_results: Dict[str, Any]) -> float:
        """Calculate reproducibility score based on cross-validation consistency."""
        
        reproducibility_scores = []
        
        for metric_name, values in cv_results['metrics'].items():
            if values and len(values) > 1:
                # Lower coefficient of variation indicates higher reproducibility
                cv_coefficient = np.std(values) / np.mean(values) if np.mean(values) != 0 else 1
                reproducibility = max(0, 1 - cv_coefficient)
                reproducibility_scores.append(reproducibility)
        
        return np.mean(reproducibility_scores) if reproducibility_scores else 0.5
    
    def _calculate_clinical_relevance_score(self, metrics: Dict[str, float]) -> float:
        """Calculate clinical relevance score based on performance metrics."""
        
        # Weight different metrics by clinical importance
        clinical_weights = {
            'accuracy': 0.2,
            'sensitivity': 0.3,  # High weight for detecting positive cases
            'specificity': 0.25, # Important for avoiding false positives
            'precision': 0.15,
            'auc': 0.1
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric, value in metrics.items():
            if metric in clinical_weights:
                weight = clinical_weights[metric]
                weighted_score += value * weight
                total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.5
    
    def _perform_statistical_analysis(self, validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis across all results."""
        
        statistical_summary = {
            'pairwise_comparisons': {},
            'overall_rankings': {},
            'significance_matrix': {}
        }
        
        # Group results by dataset
        dataset_groups = {}
        for result in validation_results:
            if result.dataset_name not in dataset_groups:
                dataset_groups[result.dataset_name] = []
            dataset_groups[result.dataset_name].append(result)
        
        # Perform pairwise comparisons within each dataset
        for dataset_name, results in dataset_groups.items():
            dataset_comparisons = {}
            
            for i, result1 in enumerate(results):
                for j, result2 in enumerate(results[i+1:], i+1):
                    comparison_key = f"{result1.algorithm_name}_vs_{result2.algorithm_name}"
                    
                    # Compare each metric
                    metric_comparisons = {}
                    for metric in self.config.metrics:
                        if (metric in result1.cross_validation_scores and 
                            metric in result2.cross_validation_scores):
                            
                            scores1 = result1.cross_validation_scores[metric]
                            scores2 = result2.cross_validation_scores[metric]
                            
                            if scores1 and scores2:
                                comparison = self.statistical_tester.compare_algorithm_performance(
                                    scores1, scores2, 
                                    result1.algorithm_name, result2.algorithm_name,
                                    metric
                                )
                                metric_comparisons[metric] = comparison
                    
                    dataset_comparisons[comparison_key] = metric_comparisons
            
            statistical_summary['pairwise_comparisons'][dataset_name] = dataset_comparisons
        
        # Calculate overall rankings
        statistical_summary['overall_rankings'] = self._calculate_overall_rankings(validation_results)
        
        return statistical_summary
    
    def _calculate_overall_rankings(self, validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """Calculate overall algorithm rankings across datasets and metrics."""
        
        rankings = {}
        
        # Group by algorithm
        algorithm_scores = {}
        for result in validation_results:
            if result.algorithm_name not in algorithm_scores:
                algorithm_scores[result.algorithm_name] = []
            
            # Use primary metric (first in config) for ranking
            primary_metric = self.config.metrics[0] if self.config.metrics else 'accuracy'
            if primary_metric in result.metrics:
                algorithm_scores[result.algorithm_name].append(result.metrics[primary_metric])
        
        # Calculate mean scores and rank
        mean_scores = {}
        for algorithm, scores in algorithm_scores.items():
            mean_scores[algorithm] = np.mean(scores) if scores else 0
        
        # Sort by mean score (descending)
        sorted_algorithms = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)
        
        rankings['by_primary_metric'] = {
            'metric': self.config.metrics[0] if self.config.metrics else 'accuracy',
            'rankings': [(rank + 1, alg, score) for rank, (alg, score) in enumerate(sorted_algorithms)]
        }
        
        return rankings
    
    def _calculate_effect_sizes(self, validation_results: List[ValidationResult]) -> Dict[str, Dict[str, float]]:
        """Calculate effect sizes for algorithm comparisons."""
        
        effect_sizes = {}
        
        # Group by dataset
        dataset_groups = {}
        for result in validation_results:
            if result.dataset_name not in dataset_groups:
                dataset_groups[result.dataset_name] = []
            dataset_groups[result.dataset_name].append(result)
        
        for dataset_name, results in dataset_groups.items():
            dataset_effects = {}
            
            for i, result1 in enumerate(results):
                for j, result2 in enumerate(results[i+1:], i+1):
                    comparison_key = f"{result1.algorithm_name}_vs_{result2.algorithm_name}"
                    
                    # Calculate Cohen's d for primary metric
                    primary_metric = self.config.metrics[0] if self.config.metrics else 'accuracy'
                    
                    if (primary_metric in result1.cross_validation_scores and 
                        primary_metric in result2.cross_validation_scores):
                        
                        scores1 = result1.cross_validation_scores[primary_metric]
                        scores2 = result2.cross_validation_scores[primary_metric]
                        
                        if scores1 and scores2:
                            pooled_std = np.sqrt((np.var(scores1) + np.var(scores2)) / 2)
                            if pooled_std > 0:
                                cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std
                                dataset_effects[comparison_key] = cohens_d
            
            effect_sizes[dataset_name] = dataset_effects
        
        return effect_sizes
    
    def _perform_power_analysis(self, validation_results: List[ValidationResult]) -> Dict[str, float]:
        """Perform power analysis for the study."""
        
        power_results = {}
        
        # Calculate average effect size across comparisons
        all_effect_sizes = []
        
        # Simplified power analysis
        n_folds = self.config.cross_validation_folds
        
        if validation_results:
            # Estimate effect size from validation results
            primary_metric = self.config.metrics[0] if self.config.metrics else 'accuracy'
            metric_values = []
            
            for result in validation_results:
                if primary_metric in result.metrics:
                    metric_values.append(result.metrics[primary_metric])
            
            if len(metric_values) >= 2:
                effect_size = np.std(metric_values)
                power_analysis_result = self.statistical_tester.power_analysis(
                    effect_size, n_folds
                )
                power_results = power_analysis_result
        
        return power_results
    
    def _assess_clinical_significance(self, validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """Assess clinical significance of results."""
        
        clinical_assessment = {
            'clinically_superior_algorithms': [],
            'minimal_clinically_important_differences': {},
            'clinical_recommendations': []
        }
        
        # Define minimal clinically important differences (MCIDs)
        mcids = {
            'accuracy': 0.05,  # 5% improvement in accuracy
            'sensitivity': 0.10,  # 10% improvement in sensitivity
            'specificity': 0.05,  # 5% improvement in specificity
            'auc': 0.03  # 3% improvement in AUC
        }
        
        clinical_assessment['minimal_clinically_important_differences'] = mcids
        
        # Find algorithms that exceed MCIDs
        algorithm_performance = {}
        for result in validation_results:
            if result.algorithm_name not in algorithm_performance:
                algorithm_performance[result.algorithm_name] = {}
            
            for metric, value in result.metrics.items():
                if metric not in algorithm_performance[result.algorithm_name]:
                    algorithm_performance[result.algorithm_name][metric] = []
                algorithm_performance[result.algorithm_name][metric].append(value)
        
        # Calculate mean performance
        for algorithm, metrics in algorithm_performance.items():
            for metric, values in metrics.items():
                algorithm_performance[algorithm][metric] = np.mean(values)
        
        # Find clinically superior algorithms
        for metric, mcid in mcids.items():
            if metric in self.config.metrics:
                metric_performances = []
                for algorithm, metrics in algorithm_performance.items():
                    if metric in metrics:
                        metric_performances.append((algorithm, metrics[metric]))
                
                if metric_performances:
                    # Sort by performance
                    metric_performances.sort(key=lambda x: x[1], reverse=True)
                    best_performance = metric_performances[0][1]
                    
                    # Find algorithms within MCID of best
                    superior_algorithms = []
                    for algorithm, performance in metric_performances:
                        if (best_performance - performance) <= mcid:
                            superior_algorithms.append({
                                'algorithm': algorithm,
                                'metric': metric,
                                'performance': performance,
                                'difference_from_best': best_performance - performance
                            })
                    
                    if superior_algorithms:
                        clinical_assessment['clinically_superior_algorithms'].extend(superior_algorithms)
        
        return clinical_assessment
    
    def _generate_recommendations(self, validation_results: List[ValidationResult],
                                statistical_summary: Dict[str, Any],
                                clinical_significance: Dict[str, Any]) -> List[str]:
        """Generate evidence-based recommendations."""
        
        recommendations = []
        
        # Overall best performing algorithm
        if validation_results:
            primary_metric = self.config.metrics[0] if self.config.metrics else 'accuracy'
            
            algorithm_scores = {}
            for result in validation_results:
                if result.algorithm_name not in algorithm_scores:
                    algorithm_scores[result.algorithm_name] = []
                if primary_metric in result.metrics:
                    algorithm_scores[result.algorithm_name].append(result.metrics[primary_metric])
            
            mean_scores = {alg: np.mean(scores) for alg, scores in algorithm_scores.items() if scores}
            
            if mean_scores:
                best_algorithm = max(mean_scores.items(), key=lambda x: x[1])
                recommendations.append(
                    f"Based on {primary_metric}, {best_algorithm[0]} shows the best overall "
                    f"performance (mean {primary_metric}: {best_algorithm[1]:.3f})"
                )
        
        # Clinical significance recommendations
        superior_algorithms = clinical_significance.get('clinically_superior_algorithms', [])
        if superior_algorithms:
            unique_algorithms = list(set(alg['algorithm'] for alg in superior_algorithms))
            recommendations.append(
                f"Clinically superior algorithms: {', '.join(unique_algorithms)}"
            )
        
        # Reproducibility recommendations
        high_repro_algorithms = [
            result.algorithm_name for result in validation_results
            if result.reproducibility_score > 0.8
        ]
        if high_repro_algorithms:
            recommendations.append(
                f"Algorithms with high reproducibility (>0.8): {', '.join(set(high_repro_algorithms))}"
            )
        
        # Performance-efficiency trade-off
        efficient_algorithms = [
            result.algorithm_name for result in validation_results
            if result.inference_time < 0.1  # Less than 100ms
        ]
        if efficient_algorithms:
            recommendations.append(
                f"Algorithms suitable for real-time applications: {', '.join(set(efficient_algorithms))}"
            )
        
        return recommendations
    
    def _identify_limitations(self, validation_results: List[ValidationResult]) -> List[str]:
        """Identify study limitations."""
        
        limitations = []
        
        # Sample size limitations
        total_folds = sum(1 for _ in validation_results) * self.config.cross_validation_folds
        if total_folds < 50:
            limitations.append("Limited sample size may affect statistical power")
        
        # Dataset diversity
        unique_datasets = len(set(result.dataset_name for result in validation_results))
        if unique_datasets < 3:
            limitations.append("Limited dataset diversity may affect generalizability")
        
        # Metric limitations
        if len(self.config.metrics) < 3:
            limitations.append("Limited number of evaluation metrics")
        
        # Reproducibility concerns
        low_repro_count = sum(1 for result in validation_results if result.reproducibility_score < 0.7)
        if low_repro_count > len(validation_results) * 0.3:
            limitations.append("Some algorithms show low reproducibility")
        
        # Computational resource limitations
        limitations.append("Computational constraints may limit algorithm complexity")
        
        return limitations


def create_demo_algorithms() -> Dict[str, Any]:
    """Create demo algorithms for benchmarking."""
    
    algorithms = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(probability=True, random_state=42),
    }
    
    return algorithms


def create_demo_datasets() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Create demo datasets for benchmarking."""
    
    np.random.seed(42)
    
    datasets = {}
    
    # Dataset 1: Balanced binary classification
    n_samples = 1000
    n_features = 20
    X1 = np.random.randn(n_samples, n_features)
    y1 = (X1[:, 0] + X1[:, 1] + np.random.randn(n_samples) * 0.1 > 0).astype(int)
    datasets['balanced_binary'] = (X1, y1)
    
    # Dataset 2: Imbalanced binary classification
    n_positive = 200
    n_negative = 800
    X2_pos = np.random.randn(n_positive, n_features) + 1
    X2_neg = np.random.randn(n_negative, n_features) - 0.5
    X2 = np.vstack([X2_pos, X2_neg])
    y2 = np.hstack([np.ones(n_positive), np.zeros(n_negative)])
    datasets['imbalanced_binary'] = (X2, y2)
    
    # Dataset 3: High-dimensional
    n_samples_hd = 500
    n_features_hd = 100
    X3 = np.random.randn(n_samples_hd, n_features_hd)
    y3 = (np.sum(X3[:, :5], axis=1) + np.random.randn(n_samples_hd) * 0.2 > 0).astype(int)
    datasets['high_dimensional'] = (X3, y3)
    
    return datasets


def main():
    """Demonstrate comprehensive research validation framework."""
    print("🔍 Comprehensive Research Validation Framework")
    print("=" * 50)
    
    # Create benchmark configuration
    config = BenchmarkConfiguration(
        name="Medical_AI_Benchmark",
        description="Comprehensive validation of medical AI algorithms",
        algorithms=['RandomForest', 'LogisticRegression', 'SVM'],
        metrics=['accuracy', 'precision', 'recall', 'f1', 'auc', 'sensitivity', 'specificity'],
        dataset_sizes=[500, 1000],
        cross_validation_folds=5,
        random_seed=42
    )
    
    print(f"📋 Benchmark Configuration:")
    print(f"   Algorithms: {', '.join(config.algorithms)}")
    print(f"   Metrics: {', '.join(config.metrics)}")
    print(f"   CV Folds: {config.cross_validation_folds}")
    
    # Initialize benchmark suite
    benchmark_suite = BenchmarkSuite(config)
    
    # Create demo algorithms and datasets
    print("\n🤖 Preparing algorithms and datasets...")
    algorithms = create_demo_algorithms()
    datasets = create_demo_datasets()
    
    print(f"   Algorithms created: {list(algorithms.keys())}")
    print(f"   Datasets created: {list(datasets.keys())}")
    
    # Run comprehensive benchmark
    print("\n🚀 Running comprehensive benchmark study...")
    study_results = benchmark_suite.run_comprehensive_benchmark(algorithms, datasets)
    
    # Display results
    print(f"\n📊 Study Results: {study_results.study_name}")
    print(f"   Algorithms compared: {len(study_results.algorithms_compared)}")
    print(f"   Datasets used: {len(study_results.datasets_used)}")
    print(f"   Total validation results: {len(study_results.validation_results)}")
    
    # Show performance summary
    print("\n🏆 Performance Summary:")
    rankings = study_results.statistical_summary.get('overall_rankings', {})
    if 'by_primary_metric' in rankings:
        metric_name = rankings['by_primary_metric']['metric']
        print(f"   Rankings by {metric_name}:")
        for rank, algorithm, score in rankings['by_primary_metric']['rankings']:
            print(f"     {rank}. {algorithm}: {score:.3f}")
    
    # Show statistical significance
    print("\n📈 Statistical Analysis:")
    pairwise_comparisons = study_results.statistical_summary.get('pairwise_comparisons', {})
    
    significant_comparisons = 0
    total_comparisons = 0
    
    for dataset_name, comparisons in pairwise_comparisons.items():
        for comparison_name, metrics in comparisons.items():
            for metric_name, comparison_data in metrics.items():
                total_comparisons += 1
                if comparison_data.get('statistical_tests', {}).get('mann_whitney', {}).get('significant', False):
                    significant_comparisons += 1
    
    if total_comparisons > 0:
        significance_rate = significant_comparisons / total_comparisons
        print(f"   Significant comparisons: {significant_comparisons}/{total_comparisons} ({significance_rate:.1%})")
    
    # Show clinical significance
    print("\n🏥 Clinical Significance:")
    clinical_significance = study_results.clinical_significance
    superior_algorithms = clinical_significance.get('clinically_superior_algorithms', [])
    
    if superior_algorithms:
        unique_superior = list(set(alg['algorithm'] for alg in superior_algorithms))
        print(f"   Clinically superior algorithms: {', '.join(unique_superior)}")
    else:
        print("   No algorithms show clinically meaningful superiority")
    
    # Show recommendations
    print("\n💡 Recommendations:")
    for i, recommendation in enumerate(study_results.recommendations, 1):
        print(f"   {i}. {recommendation}")
    
    # Show limitations
    print("\n⚠️  Study Limitations:")
    for i, limitation in enumerate(study_results.limitations, 1):
        print(f"   {i}. {limitation}")
    
    # Power analysis
    power_analysis = study_results.power_analysis
    if power_analysis:
        print(f"\n⚡ Power Analysis:")
        current_power = power_analysis.get('current_power', 0)
        adequate_power = power_analysis.get('adequate_power', False)
        print(f"   Current statistical power: {current_power:.3f}")
        print(f"   Adequate power (≥0.8): {'Yes' if adequate_power else 'No'}")
    
    print("\n✅ Comprehensive research validation framework demonstration complete!")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()