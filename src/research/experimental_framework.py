# Experimental Framework for Research Studies
# Provides controlled experimental environment with statistical validation

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, asdict
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import mlflow
import mlflow.tensorflow
from pathlib import Path
import time
import psutil
import GPUtil
from contextlib import contextmanager


@dataclass
class ExperimentConfig:
    """Configuration for controlled experiments."""
    name: str
    description: str
    models: List[str]
    datasets: List[str]
    random_seeds: List[int] = None
    num_runs: int = 3
    statistical_test: str = 'wilcoxon'  # wilcoxon, ttest, mannwhitney
    significance_level: float = 0.05
    metrics: List[str] = None
    
    def __post_init__(self):
        if self.random_seeds is None:
            self.random_seeds = [42, 123, 456]
        if self.metrics is None:
            self.metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']


@dataclass 
class ExperimentResult:
    """Container for experiment results."""
    model_name: str
    dataset_name: str
    run_id: int
    seed: int
    metrics: Dict[str, float]
    training_time: float
    memory_usage: float
    gpu_usage: float
    confusion_matrix: np.ndarray
    predictions: np.ndarray
    true_labels: np.ndarray


class ResourceMonitor:
    """Monitor system resources during training."""
    
    def __init__(self):
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        
    @contextmanager
    def monitor(self):
        """Context manager for resource monitoring."""
        start_time = time.time()
        start_memory = psutil.virtual_memory().used / 1024**3  # GB
        
        try:
            gpu_available = len(GPUtil.getGPUs()) > 0
            if gpu_available:
                start_gpu = GPUtil.getGPUs()[0].memoryUsed
            else:
                start_gpu = 0
                
            yield self
            
        finally:
            end_time = time.time()
            end_memory = psutil.virtual_memory().used / 1024**3
            
            if gpu_available:
                end_gpu = GPUtil.getGPUs()[0].memoryUsed
            else:
                end_gpu = 0
                
            self.training_time = end_time - start_time
            self.memory_usage = end_memory - start_memory
            self.gpu_usage = end_gpu - start_gpu


class ExperimentRunner:
    """
    Runs controlled experiments with statistical validation.
    Implements best practices for reproducible ML research.
    """
    
    def __init__(self, config: ExperimentConfig, output_dir: str = "experiments"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[ExperimentResult] = []
        
        # Setup MLflow
        mlflow.set_experiment(config.name)
        
    def set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        
    def run_single_experiment(self, 
                            model_factory: Callable, 
                            train_data: Tuple[np.ndarray, np.ndarray],
                            val_data: Tuple[np.ndarray, np.ndarray],
                            test_data: Tuple[np.ndarray, np.ndarray],
                            model_name: str,
                            dataset_name: str,
                            run_id: int,
                            seed: int) -> ExperimentResult:
        """Run a single experiment with resource monitoring."""
        
        self.set_seed(seed)
        
        with mlflow.start_run(nested=True):
            # Log experiment parameters
            mlflow.log_params({
                'model_name': model_name,
                'dataset': dataset_name,
                'run_id': run_id,
                'seed': seed
            })
            
            # Initialize resource monitor
            monitor = ResourceMonitor()
            
            with monitor.monitor():
                # Create and train model
                model = model_factory()
                
                # Compile model
                model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy', 'precision', 'recall']
                )
                
                # Train model
                X_train, y_train = train_data
                X_val, y_val = val_data
                
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50,
                    batch_size=32,
                    verbose=0,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(patience=10),
                        tf.keras.callbacks.ReduceLROnPlateau(patience=5)
                    ]
                )
                
                # Evaluate on test set
                X_test, y_test = test_data
                predictions = model.predict(X_test)
                predicted_labels = (predictions > 0.5).astype(int).flatten()
                
                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(y_test, predicted_labels),
                    'precision': precision_score(y_test, predicted_labels),
                    'recall': recall_score(y_test, predicted_labels),
                    'f1': f1_score(y_test, predicted_labels),
                    'auc': roc_auc_score(y_test, predictions)
                }
                
                # Log metrics to MLflow
                mlflow.log_metrics(metrics)
                mlflow.log_metrics({
                    'training_time': monitor.training_time,
                    'memory_usage': monitor.memory_usage,
                    'gpu_usage': monitor.gpu_usage
                })
                
                # Save model
                mlflow.tensorflow.log_model(model, f"model_{model_name}_{run_id}")
                
            # Create confusion matrix
            cm = confusion_matrix(y_test, predicted_labels)
            
            return ExperimentResult(
                model_name=model_name,
                dataset_name=dataset_name,
                run_id=run_id,
                seed=seed,
                metrics=metrics,
                training_time=monitor.training_time,
                memory_usage=monitor.memory_usage,
                gpu_usage=monitor.gpu_usage,
                confusion_matrix=cm,
                predictions=predictions,
                true_labels=y_test
            )
    
    def run_comparative_study(self,
                            model_factories: Dict[str, Callable],
                            data_loaders: Dict[str, Callable]) -> Dict[str, Any]:
        """
        Run comprehensive comparative study across models and datasets.
        """
        
        print(f"Starting comparative study: {self.config.name}")
        print(f"Models: {list(model_factories.keys())}")
        print(f"Datasets: {list(data_loaders.keys())}")
        print(f"Runs per configuration: {self.config.num_runs}")
        
        total_experiments = len(model_factories) * len(data_loaders) * self.config.num_runs
        current_experiment = 0
        
        for model_name, model_factory in model_factories.items():
            for dataset_name, data_loader in data_loaders.items():
                
                # Load dataset
                train_data, val_data, test_data = data_loader()
                
                for run_id in range(self.config.num_runs):
                    current_experiment += 1
                    seed = self.config.random_seeds[run_id % len(self.config.random_seeds)]
                    
                    print(f"Experiment {current_experiment}/{total_experiments}: "
                          f"{model_name} on {dataset_name} (run {run_id+1})")
                    
                    try:
                        result = self.run_single_experiment(
                            model_factory, train_data, val_data, test_data,
                            model_name, dataset_name, run_id, seed
                        )
                        self.results.append(result)
                        
                    except Exception as e:
                        print(f"Experiment failed: {e}")
                        continue
        
        # Analyze results
        analysis = self.analyze_results()
        
        # Save comprehensive report
        self.save_results()
        
        return analysis
    
    def analyze_results(self) -> Dict[str, Any]:
        """Perform statistical analysis of experimental results."""
        
        if not self.results:
            return {"error": "No results to analyze"}
        
        # Convert results to DataFrame
        data = []
        for result in self.results:
            row = {
                'model': result.model_name,
                'dataset': result.dataset_name,
                'run_id': result.run_id,
                'seed': result.seed,
                **result.metrics,
                'training_time': result.training_time,
                'memory_usage': result.memory_usage,
                'gpu_usage': result.gpu_usage
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Statistical analysis
        analysis = {
            'summary_statistics': {},
            'statistical_tests': {},
            'best_models': {},
            'significant_differences': {}
        }
        
        # Summary statistics by model
        for metric in self.config.metrics:
            analysis['summary_statistics'][metric] = df.groupby('model')[metric].agg([
                'mean', 'std', 'min', 'max', 'count'
            ]).to_dict()
        
        # Pairwise statistical tests
        models = df['model'].unique()
        for metric in self.config.metrics:
            analysis['statistical_tests'][metric] = {}
            
            for i, model1 in enumerate(models):
                for model2 in models[i+1:]:
                    data1 = df[df['model'] == model1][metric].values
                    data2 = df[df['model'] == model2][metric].values
                    
                    if len(data1) > 1 and len(data2) > 1:
                        if self.config.statistical_test == 'wilcoxon':
                            stat, p_value = stats.wilcoxon(data1, data2, alternative='two-sided')
                        elif self.config.statistical_test == 'ttest':
                            stat, p_value = stats.ttest_ind(data1, data2)
                        elif self.config.statistical_test == 'mannwhitney':
                            stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                        
                        analysis['statistical_tests'][metric][f"{model1}_vs_{model2}"] = {
                            'statistic': float(stat),
                            'p_value': float(p_value),
                            'significant': p_value < self.config.significance_level,
                            'effect_size': abs(np.mean(data1) - np.mean(data2)) / np.sqrt((np.std(data1)**2 + np.std(data2)**2) / 2)
                        }
        
        # Identify best models per metric
        for metric in self.config.metrics:
            best_model = df.groupby('model')[metric].mean().idxmax()
            best_score = df.groupby('model')[metric].mean().max()
            analysis['best_models'][metric] = {
                'model': best_model,
                'score': float(best_score)
            }
        
        return analysis
    
    def generate_visualizations(self):
        """Generate comprehensive visualization suite."""
        
        if not self.results:
            return
        
        # Convert to DataFrame
        data = []
        for result in self.results:
            row = {'model': result.model_name, **result.metrics}
            data.append(row)
        df = pd.DataFrame(data)
        
        # Create visualization directory
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Box plots for each metric
        for metric in self.config.metrics:
            plt.figure(figsize=(12, 8))
            sns.boxplot(data=df, x='model', y=metric)
            plt.title(f'{metric.title()} Comparison Across Models')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(viz_dir / f'{metric}_boxplot.png', dpi=300)
            plt.close()
        
        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        corr_matrix = df[self.config.metrics].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Metric Correlation Matrix')
        plt.tight_layout()
        plt.savefig(viz_dir / 'correlation_heatmap.png', dpi=300)
        plt.close()
        
        # Performance radar chart
        self._create_radar_chart(df, viz_dir)
        
    def _create_radar_chart(self, df: pd.DataFrame, viz_dir: Path):
        """Create radar chart comparing models across metrics."""
        
        from math import pi
        
        # Prepare data
        models = df['model'].unique()
        metrics = self.config.metrics
        
        # Normalize metrics to 0-1 scale for visualization
        df_norm = df.copy()
        for metric in metrics:
            df_norm[metric] = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
        angles += angles[:1]  # Complete the circle
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        for idx, model in enumerate(models):
            values = df_norm[df_norm['model'] == model][metrics].mean().values.flatten().tolist()
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[idx])
            ax.fill(angles, values, alpha=0.25, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Comparison (Normalized)', size=16, y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'performance_radar.png', dpi=300)
        plt.close()
    
    def save_results(self):
        """Save comprehensive experimental results."""
        
        # Save raw results
        results_data = [asdict(result) for result in self.results]
        
        # Convert numpy arrays to lists for JSON serialization
        for result in results_data:
            result['confusion_matrix'] = result['confusion_matrix'].tolist()
            result['predictions'] = result['predictions'].tolist()
            result['true_labels'] = result['true_labels'].tolist()
        
        with open(self.output_dir / 'raw_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save analysis
        analysis = self.analyze_results()
        with open(self.output_dir / 'statistical_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Generate and save visualizations
        self.generate_visualizations()
        
        # Save experiment config
        with open(self.output_dir / 'experiment_config.json', 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        print(f"Results saved to {self.output_dir}")


def create_synthetic_dataset(num_samples: int = 1000, 
                           image_size: Tuple[int, int] = (224, 224)) -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic chest X-ray dataset for controlled experiments."""
    
    # Generate synthetic images
    X = np.random.randn(num_samples, *image_size, 3).astype(np.float32)
    
    # Add realistic patterns
    for i in range(num_samples):
        # Add lung-like structures
        center_x, center_y = image_size[0] // 2, image_size[1] // 2
        y, x = np.ogrid[:image_size[0], :image_size[1]]
        
        # Create two lung regions
        lung1_mask = (x - center_x + 60)**2 + (y - center_y)**2 < 80**2
        lung2_mask = (x - center_x - 60)**2 + (y - center_y)**2 < 80**2
        
        X[i][lung1_mask] += 0.5
        X[i][lung2_mask] += 0.5
    
    # Generate labels (balanced)
    y = np.random.binomial(1, 0.5, num_samples)
    
    # Add pneumonia patterns to positive cases
    for i in range(num_samples):
        if y[i] == 1:
            # Add opacity patterns for pneumonia
            patch_x = np.random.randint(50, image_size[0] - 50)
            patch_y = np.random.randint(50, image_size[1] - 50)
            patch_size = np.random.randint(20, 60)
            
            X[i][patch_x:patch_x+patch_size, patch_y:patch_y+patch_size] += 0.8
    
    # Normalize
    X = (X - X.mean()) / (X.std() + 1e-8)
    
    return X, y


if __name__ == "__main__":
    # Demonstration of experimental framework
    
    from novel_architectures import build_research_baseline_models
    
    # Create experiment configuration
    config = ExperimentConfig(
        name="pneumonia_detection_comparative_study",
        description="Comparative study of novel CNN architectures for pneumonia detection",
        models=['efficientnet_baseline', 'dual_path_cnn', 'hierarchical_attention'],
        datasets=['synthetic_balanced'],
        num_runs=3,
        statistical_test='wilcoxon'
    )
    
    # Initialize experiment runner
    runner = ExperimentRunner(config, "experiments/comparative_study")
    
    # Define model factories
    def create_simple_cnn():
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    
    model_factories = {
        'simple_cnn': create_simple_cnn,
    }
    
    # Define data loader
    def load_synthetic_data():
        X, y = create_synthetic_dataset(1000)
        
        # Split data
        train_size = int(0.6 * len(X))
        val_size = int(0.2 * len(X))
        
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
        X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    data_loaders = {
        'synthetic_balanced': load_synthetic_data
    }
    
    # Run comparative study
    results = runner.run_comparative_study(model_factories, data_loaders)
    
    print("\nExperimental study completed!")
    print(f"Results saved to: {runner.output_dir}")