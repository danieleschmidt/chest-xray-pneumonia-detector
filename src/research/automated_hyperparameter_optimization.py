"""Automated Hyperparameter Optimization for Medical AI Models.

Implements advanced hyperparameter optimization techniques including
Bayesian optimization, genetic algorithms, and neural architecture search.
"""

import json
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple

import numpy as np
from sklearn.model_selection import cross_val_score
from scipy.optimize import minimize
import optuna

logger = logging.getLogger(__name__)


@dataclass
class HyperparameterSpace:
    """Definition of hyperparameter search space."""
    name: str
    param_type: str  # 'categorical', 'uniform', 'log_uniform', 'int_uniform'
    bounds: List[Any]
    default: Any = None


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""
    study_name: str
    optimization_method: str = "bayesian"  # bayesian, genetic, grid, random
    n_trials: int = 100
    timeout_seconds: Optional[int] = None
    n_jobs: int = 1
    random_seed: int = 42
    early_stopping_patience: int = 10
    cv_folds: int = 5
    scoring_metric: str = "accuracy"
    direction: str = "maximize"  # maximize or minimize


@dataclass
class OptimizationResult:
    """Results from hyperparameter optimization."""
    study_name: str
    best_params: Dict[str, Any]
    best_score: float
    n_trials: int
    optimization_time: float
    convergence_history: List[float]
    parameter_importance: Dict[str, float] = field(default_factory=dict)


class AutomatedHyperparameterOptimizer:
    """Advanced hyperparameter optimization engine."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.results_dir = Path("hyperparameter_optimization")
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Set random seeds for reproducibility
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
    
    def optimize(self,
                objective_function: Callable,
                hyperparameter_space: List[HyperparameterSpace],
                model_config: Dict[str, Any]) -> OptimizationResult:
        """Run hyperparameter optimization."""
        logger.info(f"Starting hyperparameter optimization: {self.config.study_name}")
        start_time = time.time()
        
        if self.config.optimization_method == "bayesian":
            result = self._bayesian_optimization(
                objective_function, hyperparameter_space, model_config
            )
        elif self.config.optimization_method == "genetic":
            result = self._genetic_optimization(
                objective_function, hyperparameter_space, model_config
            )
        elif self.config.optimization_method == "grid":
            result = self._grid_search_optimization(
                objective_function, hyperparameter_space, model_config
            )
        elif self.config.optimization_method == "random":
            result = self._random_search_optimization(
                objective_function, hyperparameter_space, model_config
            )
        else:
            raise ValueError(f"Unknown optimization method: {self.config.optimization_method}")
        
        optimization_time = time.time() - start_time
        result.optimization_time = optimization_time
        
        # Save results
        self._save_optimization_results(result)
        
        logger.info(f"Optimization completed in {optimization_time:.2f} seconds")
        logger.info(f"Best score: {result.best_score:.4f}")
        
        return result
    
    def _bayesian_optimization(self,
                              objective_function: Callable,
                              hyperparameter_space: List[HyperparameterSpace],
                              model_config: Dict[str, Any]) -> OptimizationResult:
        """Bayesian optimization using Optuna."""
        
        def optuna_objective(trial):
            """Optuna objective function wrapper."""
            params = {}
            
            for hp in hyperparameter_space:
                if hp.param_type == "categorical":
                    params[hp.name] = trial.suggest_categorical(hp.name, hp.bounds)
                elif hp.param_type == "uniform":
                    params[hp.name] = trial.suggest_float(hp.name, hp.bounds[0], hp.bounds[1])
                elif hp.param_type == "log_uniform":
                    params[hp.name] = trial.suggest_float(
                        hp.name, hp.bounds[0], hp.bounds[1], log=True
                    )
                elif hp.param_type == "int_uniform":
                    params[hp.name] = trial.suggest_int(hp.name, hp.bounds[0], hp.bounds[1])
            
            # Merge with model config
            full_config = {**model_config, **params}
            
            try:
                score = objective_function(full_config)
                return score
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return float('-inf') if self.config.direction == "maximize" else float('inf')
        
        # Create Optuna study
        direction = "maximize" if self.config.direction == "maximize" else "minimize"
        study = optuna.create_study(
            direction=direction,
            study_name=self.config.study_name,
            sampler=optuna.samplers.TPESampler(seed=self.config.random_seed)
        )
        
        # Run optimization
        study.optimize(
            optuna_objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout_seconds,
            n_jobs=self.config.n_jobs
        )
        
        # Extract results
        best_params = study.best_params
        best_score = study.best_value
        convergence_history = [trial.value for trial in study.trials 
                              if trial.value is not None]
        
        # Calculate parameter importance
        try:
            importance = optuna.importance.get_param_importances(study)
        except:
            importance = {}
        
        return OptimizationResult(
            study_name=self.config.study_name,
            best_params=best_params,
            best_score=best_score,
            n_trials=len(study.trials),
            optimization_time=0.0,  # Will be set by caller
            convergence_history=convergence_history,
            parameter_importance=importance
        )
    
    def _genetic_optimization(self,
                             objective_function: Callable,
                             hyperparameter_space: List[HyperparameterSpace],
                             model_config: Dict[str, Any]) -> OptimizationResult:
        """Genetic algorithm optimization."""
        population_size = min(50, self.config.n_trials // 2)
        n_generations = self.config.n_trials // population_size
        mutation_rate = 0.1
        crossover_rate = 0.7
        
        # Initialize population
        population = []
        for _ in range(population_size):
            individual = self._generate_random_individual(hyperparameter_space)
            population.append(individual)
        
        best_scores = []
        best_individual = None
        best_score = float('-inf') if self.config.direction == "maximize" else float('inf')
        
        for generation in range(n_generations):
            logger.info(f"Generation {generation + 1}/{n_generations}")
            
            # Evaluate population
            scores = []
            for individual in population:
                full_config = {**model_config, **individual}
                try:
                    score = objective_function(full_config)
                    scores.append(score)
                    
                    # Track best
                    is_better = (score > best_score if self.config.direction == "maximize" 
                               else score < best_score)
                    if is_better:
                        best_score = score
                        best_individual = individual.copy()
                        
                except Exception as e:
                    scores.append(float('-inf') if self.config.direction == "maximize" 
                                else float('inf'))
            
            best_scores.append(best_score)
            
            # Selection, crossover, and mutation
            population = self._evolve_population(
                population, scores, mutation_rate, crossover_rate, hyperparameter_space
            )
        
        return OptimizationResult(
            study_name=self.config.study_name,
            best_params=best_individual,
            best_score=best_score,
            n_trials=population_size * n_generations,
            optimization_time=0.0,
            convergence_history=best_scores
        )
    
    def _generate_random_individual(self, 
                                   hyperparameter_space: List[HyperparameterSpace]) -> Dict[str, Any]:
        """Generate random individual for genetic algorithm."""
        individual = {}
        
        for hp in hyperparameter_space:
            if hp.param_type == "categorical":
                individual[hp.name] = random.choice(hp.bounds)
            elif hp.param_type == "uniform":
                individual[hp.name] = random.uniform(hp.bounds[0], hp.bounds[1])
            elif hp.param_type == "log_uniform":
                log_low = np.log(hp.bounds[0])
                log_high = np.log(hp.bounds[1])
                individual[hp.name] = np.exp(random.uniform(log_low, log_high))
            elif hp.param_type == "int_uniform":
                individual[hp.name] = random.randint(hp.bounds[0], hp.bounds[1])
        
        return individual
    
    def _evolve_population(self,
                          population: List[Dict[str, Any]],
                          scores: List[float],
                          mutation_rate: float,
                          crossover_rate: float,
                          hyperparameter_space: List[HyperparameterSpace]) -> List[Dict[str, Any]]:
        """Evolve population using selection, crossover, and mutation."""
        # Tournament selection
        new_population = []
        
        for _ in range(len(population)):
            # Tournament selection
            tournament_size = 3
            tournament_indices = random.sample(range(len(population)), tournament_size)
            
            if self.config.direction == "maximize":
                winner_idx = max(tournament_indices, key=lambda i: scores[i])
            else:
                winner_idx = min(tournament_indices, key=lambda i: scores[i])
            
            new_population.append(population[winner_idx].copy())
        
        # Crossover and mutation
        for i in range(0, len(new_population) - 1, 2):
            if random.random() < crossover_rate:
                self._crossover(new_population[i], new_population[i + 1], hyperparameter_space)
            
            if random.random() < mutation_rate:
                self._mutate(new_population[i], hyperparameter_space)
            if random.random() < mutation_rate:
                self._mutate(new_population[i + 1], hyperparameter_space)
        
        return new_population
    
    def _crossover(self,
                  parent1: Dict[str, Any],
                  parent2: Dict[str, Any],
                  hyperparameter_space: List[HyperparameterSpace]):
        """Crossover between two parents."""
        for hp in hyperparameter_space:
            if random.random() < 0.5:
                parent1[hp.name], parent2[hp.name] = parent2[hp.name], parent1[hp.name]
    
    def _mutate(self,
               individual: Dict[str, Any],
               hyperparameter_space: List[HyperparameterSpace]):
        """Mutate an individual."""
        hp = random.choice(hyperparameter_space)
        
        if hp.param_type == "categorical":
            individual[hp.name] = random.choice(hp.bounds)
        elif hp.param_type == "uniform":
            # Add Gaussian noise
            current = individual[hp.name]
            noise = random.gauss(0, (hp.bounds[1] - hp.bounds[0]) * 0.1)
            individual[hp.name] = np.clip(current + noise, hp.bounds[0], hp.bounds[1])
        elif hp.param_type == "log_uniform":
            individual[hp.name] = random.uniform(hp.bounds[0], hp.bounds[1])
        elif hp.param_type == "int_uniform":
            individual[hp.name] = random.randint(hp.bounds[0], hp.bounds[1])
    
    def _grid_search_optimization(self,
                                 objective_function: Callable,
                                 hyperparameter_space: List[HyperparameterSpace],
                                 model_config: Dict[str, Any]) -> OptimizationResult:
        """Grid search optimization."""
        # Generate grid points
        grid_points = self._generate_grid_points(hyperparameter_space)
        
        best_params = None
        best_score = float('-inf') if self.config.direction == "maximize" else float('inf')
        convergence_history = []
        
        for i, params in enumerate(grid_points[:self.config.n_trials]):
            full_config = {**model_config, **params}
            
            try:
                score = objective_function(full_config)
                convergence_history.append(score)
                
                is_better = (score > best_score if self.config.direction == "maximize" 
                           else score < best_score)
                if is_better:
                    best_score = score
                    best_params = params
                    
            except Exception as e:
                logger.warning(f"Grid point {i} failed: {e}")
        
        return OptimizationResult(
            study_name=self.config.study_name,
            best_params=best_params,
            best_score=best_score,
            n_trials=len(convergence_history),
            optimization_time=0.0,
            convergence_history=convergence_history
        )
    
    def _generate_grid_points(self, 
                             hyperparameter_space: List[HyperparameterSpace]) -> List[Dict[str, Any]]:
        """Generate grid points for grid search."""
        # Simplified grid generation - in practice, this would be more sophisticated
        grid_points = []
        n_points_per_dim = max(2, int(self.config.n_trials ** (1/len(hyperparameter_space))))
        
        # Generate points for each hyperparameter
        param_grids = {}
        for hp in hyperparameter_space:
            if hp.param_type == "categorical":
                param_grids[hp.name] = hp.bounds
            elif hp.param_type in ["uniform", "log_uniform"]:
                if hp.param_type == "log_uniform":
                    points = np.logspace(np.log10(hp.bounds[0]), np.log10(hp.bounds[1]), n_points_per_dim)
                else:
                    points = np.linspace(hp.bounds[0], hp.bounds[1], n_points_per_dim)
                param_grids[hp.name] = points.tolist()
            elif hp.param_type == "int_uniform":
                points = np.linspace(hp.bounds[0], hp.bounds[1], n_points_per_dim, dtype=int)
                param_grids[hp.name] = np.unique(points).tolist()
        
        # Generate Cartesian product (simplified)
        import itertools
        param_names = list(param_grids.keys())
        param_values = list(param_grids.values())
        
        for combo in itertools.product(*param_values):
            grid_points.append(dict(zip(param_names, combo)))
        
        return grid_points
    
    def _random_search_optimization(self,
                                   objective_function: Callable,
                                   hyperparameter_space: List[HyperparameterSpace],
                                   model_config: Dict[str, Any]) -> OptimizationResult:
        """Random search optimization."""
        best_params = None
        best_score = float('-inf') if self.config.direction == "maximize" else float('inf')
        convergence_history = []
        
        for trial in range(self.config.n_trials):
            params = self._generate_random_individual(hyperparameter_space)
            full_config = {**model_config, **params}
            
            try:
                score = objective_function(full_config)
                convergence_history.append(score)
                
                is_better = (score > best_score if self.config.direction == "maximize" 
                           else score < best_score)
                if is_better:
                    best_score = score
                    best_params = params
                    
            except Exception as e:
                logger.warning(f"Trial {trial} failed: {e}")
        
        return OptimizationResult(
            study_name=self.config.study_name,
            best_params=best_params,
            best_score=best_score,
            n_trials=len(convergence_history),
            optimization_time=0.0,
            convergence_history=convergence_history
        )
    
    def _save_optimization_results(self, result: OptimizationResult):
        """Save optimization results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result.study_name}_{timestamp}_results.json"
        filepath = self.results_dir / filename
        
        # Convert result to dict for JSON serialization
        result_dict = {
            "study_name": result.study_name,
            "best_params": result.best_params,
            "best_score": result.best_score,
            "n_trials": result.n_trials,
            "optimization_time": result.optimization_time,
            "convergence_history": result.convergence_history,
            "parameter_importance": result.parameter_importance
        }
        
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        
        logger.info(f"Optimization results saved to {filepath}")


def create_medical_ai_hyperparameter_space() -> List[HyperparameterSpace]:
    """Create hyperparameter space for medical AI models."""
    return [
        HyperparameterSpace("learning_rate", "log_uniform", [1e-5, 1e-1]),
        HyperparameterSpace("batch_size", "categorical", [16, 32, 64, 128]),
        HyperparameterSpace("dropout_rate", "uniform", [0.1, 0.7]),
        HyperparameterSpace("l2_regularization", "log_uniform", [1e-6, 1e-2]),
        HyperparameterSpace("optimizer", "categorical", ["adam", "rmsprop", "sgd"]),
        HyperparameterSpace("activation", "categorical", ["relu", "elu", "swish"]),
        HyperparameterSpace("num_filters", "int_uniform", [32, 256]),
        HyperparameterSpace("filter_size", "categorical", [3, 5, 7]),
        HyperparameterSpace("num_dense_layers", "int_uniform", [1, 4]),
        HyperparameterSpace("dense_units", "int_uniform", [64, 512])
    ]


def example_objective_function(config: Dict[str, Any]) -> float:
    """Example objective function for hyperparameter optimization."""
    # Simulate model training and evaluation
    # In practice, this would train and evaluate an actual model
    
    # Simulate some realistic performance based on hyperparameters
    base_score = 0.85
    
    # Learning rate impact
    lr_penalty = abs(np.log10(config["learning_rate"]) + 3) * 0.01
    
    # Batch size impact
    batch_penalty = 0.005 if config["batch_size"] < 32 else 0
    
    # Dropout regularization
    dropout_bonus = 0.01 if 0.2 <= config["dropout_rate"] <= 0.5 else -0.01
    
    # Add some noise to simulate real training variability
    noise = np.random.normal(0, 0.02)
    
    score = base_score - lr_penalty - batch_penalty + dropout_bonus + noise
    
    # Simulate training time (longer for complex configs)
    time.sleep(0.1)  # Simulate computation time
    
    return max(0, min(1, score))


def run_optimization_example():
    """Run example hyperparameter optimization."""
    config = OptimizationConfig(
        study_name="medical_ai_optimization",
        optimization_method="bayesian",
        n_trials=20,
        scoring_metric="accuracy",
        direction="maximize"
    )
    
    optimizer = AutomatedHyperparameterOptimizer(config)
    hyperparameter_space = create_medical_ai_hyperparameter_space()
    
    model_config = {
        "model_type": "cnn",
        "input_shape": (150, 150, 3),
        "num_classes": 2
    }
    
    result = optimizer.optimize(
        objective_function=example_objective_function,
        hyperparameter_space=hyperparameter_space,
        model_config=model_config
    )
    
    print(f"Optimization completed!")
    print(f"Best score: {result.best_score:.4f}")
    print(f"Best parameters: {result.best_params}")
    print(f"Optimization time: {result.optimization_time:.2f} seconds")


if __name__ == "__main__":
    run_optimization_example()