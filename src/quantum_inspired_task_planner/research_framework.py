"""Research framework for quantum-inspired task planning algorithms."""

import math
import time
import json
import logging
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

from .advanced_scheduler import QuantumMLScheduler
from .simple_optimization import SimpleQuantumAnnealer

logger = logging.getLogger(__name__)


@dataclass
class ExperimentalResult:
    """Results from a single experimental run."""
    algorithm_name: str
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    execution_time: float
    memory_usage: float
    convergence_iterations: int
    success_rate: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BenchmarkSuite:
    """Comprehensive benchmark suite for scheduling algorithms."""
    name: str
    description: str
    task_configurations: List[Dict]
    resource_constraints: Dict[str, float]
    expected_outcomes: Dict[str, float]
    complexity_level: str  # 'simple', 'medium', 'complex'


class QuantumSchedulingResearchFramework:
    """Research framework for comparing and optimizing scheduling algorithms."""
    
    def __init__(self):
        self.experimental_results: List[ExperimentalResult] = []
        self.benchmark_suites: Dict[str, BenchmarkSuite] = {}
        self.baseline_algorithms = {}
        self.research_datasets = {}
        self._initialize_benchmark_suites()
        self._initialize_baseline_algorithms()
    
    def _initialize_benchmark_suites(self) -> None:
        """Initialize standard benchmark suites for algorithm comparison."""
        
        # Simple scheduling benchmark
        simple_suite = BenchmarkSuite(
            name="simple_scheduling",
            description="Basic task scheduling with minimal dependencies",
            task_configurations=[
                {"name": f"task_{i}", "priority": "medium", "duration": 3600}
                for i in range(10)
            ],
            resource_constraints={"cpu": 100.0, "memory": 1000.0},
            expected_outcomes={"completion_time": 36000, "utilization": 0.8},
            complexity_level="simple"
        )
        
        # Complex dependency benchmark
        complex_suite = BenchmarkSuite(
            name="complex_dependencies",
            description="Complex task graph with multiple dependency chains",
            task_configurations=[
                {"name": f"task_{i}", "priority": "high" if i < 5 else "medium", 
                 "duration": 1800 + (i * 300), "dependencies": [f"task_{j}" for j in range(max(0, i-2), i)]}
                for i in range(20)
            ],
            resource_constraints={"cpu": 200.0, "memory": 2000.0},
            expected_outcomes={"completion_time": 72000, "utilization": 0.85},
            complexity_level="complex"
        )
        
        # Resource-constrained benchmark
        resource_suite = BenchmarkSuite(
            name="resource_constrained",
            description="Heavy resource contention scenario",
            task_configurations=[
                {"name": f"resource_task_{i}", "priority": "critical" if i < 3 else "medium",
                 "duration": 2400, "resources": {"cpu": 25.0, "memory": 200.0}}
                for i in range(15)
            ],
            resource_constraints={"cpu": 100.0, "memory": 800.0},
            expected_outcomes={"completion_time": 43200, "utilization": 0.95},
            complexity_level="medium"
        )
        
        self.benchmark_suites = {
            "simple": simple_suite,
            "complex": complex_suite,
            "resource_constrained": resource_suite
        }
    
    def _initialize_baseline_algorithms(self) -> None:
        """Initialize baseline algorithms for comparison."""
        
        # First-Come-First-Serve (FCFS)
        def fcfs_scheduler(tasks: List[Dict]) -> List[str]:
            return [task["name"] for task in sorted(tasks, key=lambda t: t.get("created_at", 0))]
        
        # Priority-based scheduling
        def priority_scheduler(tasks: List[Dict]) -> List[str]:
            priority_values = {"critical": 4, "high": 3, "medium": 2, "low": 1}
            return [task["name"] for task in sorted(
                tasks, key=lambda t: priority_values.get(t.get("priority", "medium"), 2), reverse=True
            )]
        
        # Shortest Job First (SJF)
        def sjf_scheduler(tasks: List[Dict]) -> List[str]:
            return [task["name"] for task in sorted(tasks, key=lambda t: t.get("duration", 3600))]
        
        self.baseline_algorithms = {
            "fcfs": fcfs_scheduler,
            "priority": priority_scheduler,
            "sjf": sjf_scheduler
        }
    
    def run_comparative_study(self, algorithm_configs: Dict[str, Dict]) -> Dict[str, List[ExperimentalResult]]:
        """Run comprehensive comparative study across multiple algorithms."""
        results = defaultdict(list)
        
        print("🔬 Starting Comparative Algorithm Study")
        print("=" * 50)
        
        for benchmark_name, benchmark_suite in self.benchmark_suites.items():
            print(f"\n📊 Running Benchmark: {benchmark_name}")
            print(f"Description: {benchmark_suite.description}")
            print(f"Complexity: {benchmark_suite.complexity_level}")
            print(f"Tasks: {len(benchmark_suite.task_configurations)}")
            
            # Test each algorithm configuration
            for algorithm_name, config in algorithm_configs.items():
                print(f"\n🧪 Testing Algorithm: {algorithm_name}")
                
                # Run multiple trials for statistical significance
                trial_results = []
                for trial in range(config.get("trials", 3)):
                    print(f"  Trial {trial + 1}/{config.get('trials', 3)}")
                    
                    result = self._run_single_experiment(
                        algorithm_name=algorithm_name,
                        algorithm_config=config,
                        benchmark_suite=benchmark_suite,
                        trial_number=trial
                    )
                    
                    trial_results.append(result)
                    results[f"{benchmark_name}_{algorithm_name}"].append(result)
                
                # Calculate trial statistics
                avg_time = statistics.mean([r.execution_time for r in trial_results])
                std_time = statistics.stdev([r.execution_time for r in trial_results]) if len(trial_results) > 1 else 0
                avg_success = statistics.mean([r.success_rate for r in trial_results])
                
                print(f"  Results: {avg_time:.2f}±{std_time:.2f}s, Success: {avg_success:.1%}")
        
        return dict(results)
    
    def _run_single_experiment(self, algorithm_name: str, algorithm_config: Dict, 
                             benchmark_suite: BenchmarkSuite, trial_number: int) -> ExperimentalResult:
        """Run a single experimental trial."""
        start_time = time.time()
        
        # Initialize scheduler based on algorithm type
        if algorithm_name.startswith("quantum"):
            scheduler = QuantumMLScheduler(max_parallel_tasks=algorithm_config.get("parallel_tasks", 4))
        else:
            # Use baseline algorithm
            scheduler = QuantumMLScheduler(max_parallel_tasks=4)  # For fair comparison
        
        # Setup experiment
        task_ids = []
        for task_config in benchmark_suite.task_configurations:
            task_id = scheduler.add_intelligent_task(
                name=task_config["name"],
                description=task_config.get("description", ""),
                priority=self._parse_priority(task_config.get("priority", "medium")),
                estimated_duration=timedelta(seconds=task_config.get("duration", 3600)),
                dependencies=task_config.get("dependencies", []),
                resource_requirements=task_config.get("resources", {})
            )
            task_ids.append(task_id)
        
        # Execute scheduling algorithm
        performance_metrics = {}
        
        if algorithm_name in self.baseline_algorithms:
            # Run baseline algorithm
            task_order = self.baseline_algorithms[algorithm_name](benchmark_suite.task_configurations)
            performance_metrics = self._simulate_baseline_execution(scheduler, task_order)
        else:
            # Run quantum-inspired algorithm
            performance_metrics = self._execute_quantum_algorithm(scheduler, algorithm_config)
        
        execution_time = time.time() - start_time
        
        # Calculate success rate
        completed_tasks = len([t for t in scheduler.tasks.values() if t.status.value == "completed"])
        success_rate = completed_tasks / len(task_ids) if task_ids else 0.0
        
        return ExperimentalResult(
            algorithm_name=algorithm_name,
            parameters=algorithm_config,
            performance_metrics=performance_metrics,
            execution_time=execution_time,
            memory_usage=self._estimate_memory_usage(scheduler),
            convergence_iterations=performance_metrics.get("iterations", 0),
            success_rate=success_rate
        )
    
    def _parse_priority(self, priority_str: str):
        """Parse priority string to enum."""
        from .quantum_scheduler import TaskPriority
        priority_map = {
            "critical": TaskPriority.CRITICAL,
            "high": TaskPriority.HIGH,
            "medium": TaskPriority.MEDIUM,
            "low": TaskPriority.LOW
        }
        return priority_map.get(priority_str, TaskPriority.MEDIUM)
    
    def _simulate_baseline_execution(self, scheduler, task_order: List[str]) -> Dict[str, float]:
        """Simulate execution of baseline algorithm."""
        metrics = {"total_time": 0, "utilization": 0, "iterations": 0}
        
        # Simple simulation of task execution
        current_time = 0
        for task_name in task_order:
            # Find matching task
            matching_task = None
            for task in scheduler.tasks.values():
                if task.name == task_name:
                    matching_task = task
                    break
            
            if matching_task:
                task_duration = matching_task.estimated_duration.total_seconds()
                current_time += task_duration
                matching_task.status = scheduler.TaskStatus.COMPLETED if hasattr(scheduler, 'TaskStatus') else None
        
        metrics["total_time"] = current_time
        metrics["utilization"] = min(1.0, len(task_order) / 10.0)  # Simplified calculation
        
        return metrics
    
    def _execute_quantum_algorithm(self, scheduler: QuantumMLScheduler, config: Dict) -> Dict[str, float]:
        """Execute quantum-inspired scheduling algorithm."""
        metrics = {"total_time": 0, "utilization": 0, "iterations": 0}
        
        iterations = 0
        max_iterations = config.get("max_iterations", 100)
        
        while iterations < max_iterations:
            # Get next tasks using quantum algorithm
            next_tasks = scheduler.get_intelligent_next_tasks()
            
            if not next_tasks:
                break
            
            # Simulate starting tasks
            for task in next_tasks:
                scheduler.start_task(task.id)
            
            # Simulate task completion (simplified)
            for task in next_tasks:
                scheduler.complete_task_with_learning(task.id)
            
            iterations += 1
        
        # Calculate metrics
        completed_count = len([t for t in scheduler.tasks.values() if t.status.value == "completed"])
        total_duration = sum(t.estimated_duration.total_seconds() for t in scheduler.tasks.values())
        
        metrics["total_time"] = total_duration / max(1, scheduler.max_parallel_tasks)  # Parallel execution estimate
        metrics["utilization"] = completed_count / len(scheduler.tasks) if scheduler.tasks else 0
        metrics["iterations"] = iterations
        
        return metrics
    
    def _estimate_memory_usage(self, scheduler) -> float:
        """Estimate memory usage of scheduler (simplified)."""
        base_memory = 1.0  # MB
        task_memory = len(scheduler.tasks) * 0.001  # 1KB per task
        history_memory = len(getattr(scheduler, 'scheduling_history', [])) * 0.0005
        
        return base_memory + task_memory + history_memory
    
    def analyze_statistical_significance(self, results: Dict[str, List[ExperimentalResult]]) -> Dict[str, Dict]:
        """Analyze statistical significance of experimental results."""
        analysis = {}
        
        # Group results by benchmark
        benchmarks = set(key.split('_')[0] for key in results.keys())
        
        for benchmark in benchmarks:
            benchmark_results = {k: v for k, v in results.items() if k.startswith(benchmark)}
            
            if len(benchmark_results) < 2:
                continue
            
            analysis[benchmark] = {}
            
            # Compare execution times
            algorithm_times = {}
            for key, result_list in benchmark_results.items():
                algorithm_name = '_'.join(key.split('_')[1:])
                times = [r.execution_time for r in result_list]
                algorithm_times[algorithm_name] = {
                    'mean': statistics.mean(times),
                    'std': statistics.stdev(times) if len(times) > 1 else 0,
                    'median': statistics.median(times),
                    'min': min(times),
                    'max': max(times)
                }
            
            analysis[benchmark]['execution_time'] = algorithm_times
            
            # Compare success rates
            algorithm_success = {}
            for key, result_list in benchmark_results.items():
                algorithm_name = '_'.join(key.split('_')[1:])
                success_rates = [r.success_rate for r in result_list]
                algorithm_success[algorithm_name] = {
                    'mean': statistics.mean(success_rates),
                    'std': statistics.stdev(success_rates) if len(success_rates) > 1 else 0,
                    'median': statistics.median(success_rates)
                }
            
            analysis[benchmark]['success_rate'] = algorithm_success
            
            # Identify best performing algorithm
            best_algorithm = min(algorithm_times.keys(), key=lambda k: algorithm_times[k]['mean'])
            analysis[benchmark]['best_algorithm'] = {
                'name': best_algorithm,
                'improvement_factor': max(algorithm_times[k]['mean'] for k in algorithm_times.keys()) / algorithm_times[best_algorithm]['mean']
            }
        
        return analysis
    
    def generate_research_report(self, results: Dict[str, List[ExperimentalResult]], 
                               statistical_analysis: Dict) -> str:
        """Generate comprehensive research report."""
        report = []
        report.append("# Quantum-Inspired Task Scheduling Research Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append("")
        
        total_experiments = sum(len(result_list) for result_list in results.values())
        unique_algorithms = len(set('_'.join(key.split('_')[1:]) for key in results.keys()))
        unique_benchmarks = len(set(key.split('_')[0] for key in results.keys()))
        
        report.append(f"- Total experiments conducted: {total_experiments}")
        report.append(f"- Algorithms tested: {unique_algorithms}")
        report.append(f"- Benchmark suites: {unique_benchmarks}")
        report.append("")
        
        # Key Findings
        report.append("## Key Findings")
        report.append("")
        
        for benchmark, analysis in statistical_analysis.items():
            report.append(f"### {benchmark.replace('_', ' ').title()} Benchmark")
            
            if 'best_algorithm' in analysis:
                best_alg = analysis['best_algorithm']
                report.append(f"- **Best performing algorithm**: {best_alg['name']}")
                report.append(f"- **Performance improvement**: {best_alg['improvement_factor']:.2f}x over worst algorithm")
            
            if 'execution_time' in analysis:
                exec_times = analysis['execution_time']
                fastest_alg = min(exec_times.keys(), key=lambda k: exec_times[k]['mean'])
                report.append(f"- **Fastest execution**: {fastest_alg} ({exec_times[fastest_alg]['mean']:.2f}s average)")
            
            report.append("")
        
        # Detailed Results
        report.append("## Detailed Results")
        report.append("")
        
        for benchmark, analysis in statistical_analysis.items():
            report.append(f"### {benchmark.replace('_', ' ').title()}")
            report.append("")
            report.append("| Algorithm | Avg Time (s) | Std Dev | Success Rate | Memory (MB) |")
            report.append("|-----------|--------------|---------|--------------|-------------|")
            
            if 'execution_time' in analysis:
                for alg_name, time_stats in analysis['execution_time'].items():
                    success_stats = analysis.get('success_rate', {}).get(alg_name, {})
                    
                    # Find corresponding results for memory usage
                    memory_usage = 0
                    for key, result_list in results.items():
                        if key.endswith(alg_name) and key.startswith(benchmark):
                            memory_usage = statistics.mean([r.memory_usage for r in result_list])
                            break
                    
                    report.append(f"| {alg_name} | {time_stats['mean']:.2f} | {time_stats['std']:.2f} | "
                                f"{success_stats.get('mean', 0):.1%} | {memory_usage:.2f} |")
            
            report.append("")
        
        # Research Conclusions
        report.append("## Research Conclusions")
        report.append("")
        report.append("### Algorithm Performance Insights")
        report.append("")
        
        # Analyze cross-benchmark performance
        all_algorithms = set()
        for analysis in statistical_analysis.values():
            if 'execution_time' in analysis:
                all_algorithms.update(analysis['execution_time'].keys())
        
        for algorithm in all_algorithms:
            avg_performance = []
            for benchmark, analysis in statistical_analysis.items():
                if 'execution_time' in analysis and algorithm in analysis['execution_time']:
                    avg_performance.append(analysis['execution_time'][algorithm]['mean'])
            
            if avg_performance:
                overall_avg = statistics.mean(avg_performance)
                report.append(f"- **{algorithm}**: Overall average execution time: {overall_avg:.2f}s")
        
        report.append("")
        report.append("### Recommendations")
        report.append("")
        
        # Generate recommendations based on results
        quantum_algorithms = [alg for alg in all_algorithms if 'quantum' in alg.lower()]
        baseline_algorithms = [alg for alg in all_algorithms if alg in ['fcfs', 'priority', 'sjf']]
        
        if quantum_algorithms and baseline_algorithms:
            # Compare quantum vs baseline
            quantum_avg = statistics.mean([
                statistics.mean([
                    analysis['execution_time'][alg]['mean']
                    for alg in quantum_algorithms
                    if alg in analysis['execution_time']
                ])
                for analysis in statistical_analysis.values()
                if 'execution_time' in analysis
            ])
            
            baseline_avg = statistics.mean([
                statistics.mean([
                    analysis['execution_time'][alg]['mean']
                    for alg in baseline_algorithms
                    if alg in analysis['execution_time']
                ])
                for analysis in statistical_analysis.values()
                if 'execution_time' in analysis
            ])
            
            if quantum_avg < baseline_avg:
                improvement = (baseline_avg - quantum_avg) / baseline_avg * 100
                report.append(f"1. Quantum-inspired algorithms show {improvement:.1f}% improvement over baseline methods")
            else:
                report.append("1. Baseline algorithms currently outperform quantum-inspired methods")
        
        report.append("2. Further research needed on parameter optimization for quantum algorithms")
        report.append("3. Consider hybrid approaches combining quantum and classical techniques")
        report.append("")
        
        # Future Work
        report.append("## Future Research Directions")
        report.append("")
        report.append("1. **Parameter Sensitivity Analysis**: Systematic study of hyperparameter impact")
        report.append("2. **Real-world Validation**: Testing with actual production workloads")
        report.append("3. **Scalability Studies**: Performance analysis with larger task sets")
        report.append("4. **Hardware Acceleration**: Implementation on quantum hardware")
        report.append("5. **Multi-objective Optimization**: Balancing multiple performance criteria")
        report.append("")
        
        return "\n".join(report)
    
    def export_experimental_data(self, results: Dict[str, List[ExperimentalResult]], 
                                filename: Optional[str] = None) -> str:
        """Export experimental data for external analysis."""
        if filename is None:
            filename = f"quantum_scheduling_experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert results to serializable format
        export_data = {
            "metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "total_experiments": sum(len(result_list) for result_list in results.values()),
                "benchmarks": list(set(key.split('_')[0] for key in results.keys())),
                "algorithms": list(set('_'.join(key.split('_')[1:]) for key in results.keys()))
            },
            "experiments": {}
        }
        
        for key, result_list in results.items():
            export_data["experiments"][key] = [
                {
                    "algorithm_name": result.algorithm_name,
                    "parameters": result.parameters,
                    "performance_metrics": result.performance_metrics,
                    "execution_time": result.execution_time,
                    "memory_usage": result.memory_usage,
                    "convergence_iterations": result.convergence_iterations,
                    "success_rate": result.success_rate,
                    "timestamp": result.timestamp.isoformat()
                }
                for result in result_list
            ]
        
        # Write to file
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filename
    
    def create_visualization_data(self, statistical_analysis: Dict) -> Dict:
        """Create data structure optimized for visualization."""
        viz_data = {
            "performance_comparison": {},
            "algorithm_rankings": {},
            "benchmark_characteristics": {}
        }
        
        # Performance comparison data
        for benchmark, analysis in statistical_analysis.items():
            if 'execution_time' in analysis:
                viz_data["performance_comparison"][benchmark] = {
                    "algorithms": list(analysis['execution_time'].keys()),
                    "execution_times": [
                        analysis['execution_time'][alg]['mean'] 
                        for alg in analysis['execution_time'].keys()
                    ],
                    "error_bars": [
                        analysis['execution_time'][alg]['std'] 
                        for alg in analysis['execution_time'].keys()
                    ]
                }
        
        # Algorithm rankings across benchmarks
        all_algorithms = set()
        for analysis in statistical_analysis.values():
            if 'execution_time' in analysis:
                all_algorithms.update(analysis['execution_time'].keys())
        
        for algorithm in all_algorithms:
            rankings = []
            for benchmark, analysis in statistical_analysis.items():
                if 'execution_time' in analysis and algorithm in analysis['execution_time']:
                    # Rank algorithm in this benchmark (1 = best)
                    sorted_algs = sorted(
                        analysis['execution_time'].keys(),
                        key=lambda k: analysis['execution_time'][k]['mean']
                    )
                    rank = sorted_algs.index(algorithm) + 1
                    rankings.append(rank)
            
            if rankings:
                viz_data["algorithm_rankings"][algorithm] = {
                    "average_rank": statistics.mean(rankings),
                    "rank_std": statistics.stdev(rankings) if len(rankings) > 1 else 0,
                    "rankings": rankings
                }
        
        return viz_data