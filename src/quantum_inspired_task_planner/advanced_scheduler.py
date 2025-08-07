"""Advanced quantum-inspired scheduler with ML-based optimization."""

import logging
import math
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import json

from .quantum_scheduler import QuantumScheduler, QuantumTask, TaskPriority, TaskStatus
from .simple_optimization import SimpleQuantumAnnealer

logger = logging.getLogger(__name__)


@dataclass
class SchedulingMetrics:
    """Comprehensive scheduling performance metrics."""
    avg_task_completion_time: float = 0.0
    resource_utilization_efficiency: float = 0.0
    priority_satisfaction_rate: float = 0.0
    dependency_resolution_time: float = 0.0
    quantum_coherence_score: float = 0.0
    scheduling_overhead: float = 0.0
    parallel_execution_efficiency: float = 0.0


@dataclass
class AdaptiveParameters:
    """Self-adaptive scheduling parameters."""
    learning_rate: float = 0.01
    momentum: float = 0.9
    temperature_adaptation_rate: float = 0.05
    priority_decay_factor: float = 0.95
    entanglement_strength: float = 0.5
    superposition_threshold: float = 0.3


class QuantumMLScheduler(QuantumScheduler):
    """Advanced quantum scheduler with machine learning optimization."""
    
    def __init__(self, max_parallel_tasks: int = 8):
        super().__init__(max_parallel_tasks)
        self.adaptive_params = AdaptiveParameters()
        self.scheduling_history: List[Dict] = []
        self.performance_metrics = SchedulingMetrics()
        self.pattern_recognition_model = {}
        self.quantum_coherence_matrix = {}
        self.optimization_engine = SimpleQuantumAnnealer()
        
    def add_intelligent_task(self, name: str, description: str = "",
                           priority: TaskPriority = TaskPriority.MEDIUM,
                           dependencies: Optional[List[str]] = None,
                           estimated_duration: Optional[timedelta] = None,
                           resource_requirements: Optional[Dict[str, float]] = None,
                           deadline: Optional[datetime] = None,
                           tags: Optional[List[str]] = None) -> str:
        """Create task with ML-based estimation and intelligent defaults."""
        
        # Use historical data to estimate duration if not provided
        if estimated_duration is None:
            estimated_duration = self._predict_task_duration(name, description, tags or [])
        
        # Automatically detect dependencies based on pattern analysis
        if dependencies is None:
            dependencies = self._predict_dependencies(name, description, tags or [])
        
        # Create enhanced task
        task = QuantumTask(
            name=name,
            description=description,
            priority=priority,
            dependencies=set(dependencies),
            estimated_duration=estimated_duration,
            resource_requirements=resource_requirements or {}
        )
        
        # Enhanced properties for ML scheduler
        task.deadline = deadline
        task.tags = set(tags or [])
        task.predicted_success_probability = self._calculate_success_probability(task)
        task.complexity_score = self._calculate_complexity_score(task)
        
        return self.add_task(task)
    
    def _predict_task_duration(self, name: str, description: str, tags: List[str]) -> timedelta:
        """Predict task duration based on historical patterns."""
        # Simple ML-based duration prediction
        base_duration = timedelta(hours=1)  # Default
        
        # Analyze historical tasks with similar characteristics
        similar_tasks = self._find_similar_tasks(name, description, tags)
        
        if similar_tasks:
            avg_duration = sum(
                (task.completed_at - task.started_at).total_seconds() 
                for task in similar_tasks 
                if task.completed_at and task.started_at
            ) / len(similar_tasks)
            
            return timedelta(seconds=avg_duration)
        
        # Use complexity-based estimation
        complexity_multiplier = len(description.split()) * 0.1 + len(tags) * 0.05 + 1.0
        return timedelta(seconds=base_duration.total_seconds() * complexity_multiplier)
    
    def _predict_dependencies(self, name: str, description: str, tags: List[str]) -> List[str]:
        """Predict task dependencies using pattern recognition."""
        predicted_deps = []
        
        # Pattern-based dependency detection
        for existing_task in self.tasks.values():
            similarity_score = self._calculate_task_similarity(
                (name, description, tags),
                (existing_task.name, existing_task.description, list(existing_task.tags))
            )
            
            if similarity_score > 0.7 and existing_task.status != TaskStatus.COMPLETED:
                predicted_deps.append(existing_task.id)
        
        return predicted_deps
    
    def _calculate_task_similarity(self, task1_data: Tuple, task2_data: Tuple) -> float:
        """Calculate similarity between two tasks."""
        name1, desc1, tags1 = task1_data
        name2, desc2, tags2 = task2_data
        
        # Simple text similarity
        name_similarity = self._text_similarity(name1, name2)
        desc_similarity = self._text_similarity(desc1, desc2)
        
        # Tag overlap
        set1, set2 = set(tags1), set(tags2)
        tag_similarity = len(set1.intersection(set2)) / len(set1.union(set2)) if set1.union(set2) else 0
        
        return (name_similarity + desc_similarity + tag_similarity) / 3
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity calculation."""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1.union(words2):
            return 0.0
        
        return len(words1.intersection(words2)) / len(words1.union(words2))
    
    def _find_similar_tasks(self, name: str, description: str, tags: List[str]) -> List[QuantumTask]:
        """Find historically similar tasks."""
        similar_tasks = []
        
        for task in self.tasks.values():
            similarity = self._calculate_task_similarity(
                (name, description, tags),
                (task.name, task.description, list(getattr(task, 'tags', [])))
            )
            
            if similarity > 0.5:
                similar_tasks.append(task)
        
        return similar_tasks
    
    def _calculate_success_probability(self, task: QuantumTask) -> float:
        """Calculate probability of successful task completion."""
        base_probability = 0.8  # Default 80% success rate
        
        # Adjust based on complexity
        complexity_penalty = getattr(task, 'complexity_score', 1.0) * 0.1
        
        # Adjust based on dependencies
        dependency_penalty = len(task.dependencies) * 0.05
        
        # Adjust based on priority
        priority_bonus = task.priority.value * 0.02
        
        probability = base_probability - complexity_penalty - dependency_penalty + priority_bonus
        return max(0.1, min(0.99, probability))  # Keep between 10% and 99%
    
    def _calculate_complexity_score(self, task: QuantumTask) -> float:
        """Calculate task complexity score."""
        base_complexity = 1.0
        
        # Description length factor
        desc_factor = len(task.description.split()) * 0.05
        
        # Dependencies factor
        dep_factor = len(task.dependencies) * 0.2
        
        # Resource requirements factor
        resource_factor = sum(task.resource_requirements.values()) * 0.01
        
        return base_complexity + desc_factor + dep_factor + resource_factor
    
    def get_intelligent_next_tasks(self) -> List[QuantumTask]:
        """Get next tasks using ML-enhanced selection."""
        available_tasks = []
        
        for task in self.tasks.values():
            if (task.status == TaskStatus.PENDING and 
                self._are_dependencies_satisfied(task) and
                len(self.running_tasks) < self.max_parallel_tasks):
                available_tasks.append(task)
        
        if not available_tasks:
            return []
        
        # ML-enhanced scoring
        scored_tasks = []
        for task in available_tasks:
            score = self._calculate_ml_priority_score(task)
            scored_tasks.append((task, score))
        
        # Sort by ML score
        scored_tasks.sort(key=lambda x: x[1], reverse=True)
        
        # Select optimal subset using quantum optimization
        optimal_tasks = self._quantum_task_selection([t[0] for t in scored_tasks])
        
        return optimal_tasks
    
    def _calculate_ml_priority_score(self, task: QuantumTask) -> float:
        """Calculate ML-enhanced priority score."""
        base_score = self._calculate_priority_score(task)
        
        # ML enhancements
        success_probability = getattr(task, 'predicted_success_probability', 0.8)
        complexity_score = getattr(task, 'complexity_score', 1.0)
        
        # Deadline urgency
        deadline_factor = 1.0
        if hasattr(task, 'deadline') and task.deadline:
            time_to_deadline = (task.deadline - datetime.now()).total_seconds()
            if time_to_deadline > 0:
                deadline_factor = max(0.1, 1.0 - (time_to_deadline / (7 * 24 * 3600)))  # 1 week normalization
        
        # Historical performance factor
        historical_factor = self._get_historical_performance_factor(task)
        
        # Quantum coherence contribution
        coherence_factor = self._calculate_quantum_coherence_factor(task)
        
        ml_score = (base_score * success_probability * deadline_factor * 
                   historical_factor * coherence_factor) / complexity_score
        
        return ml_score
    
    def _get_historical_performance_factor(self, task: QuantumTask) -> float:
        """Get performance factor based on historical data."""
        similar_tasks = self._find_similar_tasks(
            task.name, 
            task.description, 
            list(getattr(task, 'tags', []))
        )
        
        if not similar_tasks:
            return 1.0
        
        # Calculate average completion rate
        completed_count = sum(1 for t in similar_tasks if t.status == TaskStatus.COMPLETED)
        completion_rate = completed_count / len(similar_tasks)
        
        return 0.5 + completion_rate * 0.5  # Factor between 0.5 and 1.0
    
    def _calculate_quantum_coherence_factor(self, task: QuantumTask) -> float:
        """Calculate quantum coherence factor for task scheduling."""
        coherence_score = 1.0
        
        # Entanglement effects
        if task.entangled_tasks:
            entangled_running = sum(1 for et_id in task.entangled_tasks 
                                  if et_id in self.running_tasks)
            if entangled_running > 0:
                coherence_score *= (1.0 + entangled_running * 0.2)
        
        # Superposition effects
        coherence_score *= task.superposition_weight
        
        return coherence_score
    
    def _quantum_task_selection(self, candidate_tasks: List[QuantumTask]) -> List[QuantumTask]:
        """Select optimal task subset using quantum optimization."""
        if len(candidate_tasks) <= self.max_parallel_tasks:
            return candidate_tasks
        
        # Create optimization problem
        def selection_cost(task_indices: List[int]) -> float:
            if len(task_indices) > self.max_parallel_tasks:
                return float('inf')  # Invalid solution
            
            total_cost = 0.0
            selected_tasks = [candidate_tasks[i] for i in task_indices if i < len(candidate_tasks)]
            
            for task in selected_tasks:
                # Cost based on inverse priority score
                priority_score = self._calculate_ml_priority_score(task)
                total_cost -= priority_score  # Minimize negative priority
            
            return total_cost
        
        # Use quantum annealing for selection
        task_ids = [str(i) for i in range(len(candidate_tasks))]
        
        try:
            result = self.optimization_engine.anneal(
                lambda schedule: selection_cost([int(task_id) for task_id in schedule[:self.max_parallel_tasks]]),
                task_ids,
                max_iterations=500
            )
            
            selected_indices = [int(task_id) for task_id in result.optimal_schedule[:self.max_parallel_tasks]]
            return [candidate_tasks[i] for i in selected_indices if i < len(candidate_tasks)]
            
        except Exception as e:
            logger.warning(f"Quantum optimization failed: {e}, using fallback selection")
            return candidate_tasks[:self.max_parallel_tasks]
    
    def adaptive_learning_update(self, completed_task: QuantumTask) -> None:
        """Update ML model based on completed task performance."""
        # Record performance data
        actual_duration = None
        if completed_task.started_at and completed_task.completed_at:
            actual_duration = (completed_task.completed_at - completed_task.started_at).total_seconds()
        
        performance_record = {
            'task_id': completed_task.id,
            'predicted_duration': completed_task.estimated_duration.total_seconds(),
            'actual_duration': actual_duration,
            'success_probability': getattr(completed_task, 'predicted_success_probability', 0.8),
            'complexity_score': getattr(completed_task, 'complexity_score', 1.0),
            'completion_time': datetime.now().isoformat(),
            'priority': completed_task.priority.value
        }
        
        self.scheduling_history.append(performance_record)
        
        # Update adaptive parameters based on performance
        self._update_adaptive_parameters(performance_record)
        
        # Update pattern recognition model
        self._update_pattern_model(completed_task)
    
    def _update_adaptive_parameters(self, performance_record: Dict) -> None:
        """Update scheduling parameters based on performance feedback."""
        if performance_record['actual_duration'] and performance_record['predicted_duration']:
            prediction_error = abs(
                performance_record['actual_duration'] - performance_record['predicted_duration']
            ) / performance_record['predicted_duration']
            
            # Adjust learning rate based on prediction accuracy
            if prediction_error > 0.3:  # High error
                self.adaptive_params.learning_rate *= 1.1  # Increase learning
                self.adaptive_params.temperature_adaptation_rate *= 1.05
            else:  # Good prediction
                self.adaptive_params.learning_rate *= 0.99  # Slight decrease for stability
        
        # Keep parameters in reasonable bounds
        self.adaptive_params.learning_rate = max(0.001, min(0.1, self.adaptive_params.learning_rate))
        self.adaptive_params.temperature_adaptation_rate = max(0.01, min(0.2, self.adaptive_params.temperature_adaptation_rate))
    
    def _update_pattern_model(self, completed_task: QuantumTask) -> None:
        """Update pattern recognition model with completed task data."""
        task_signature = f"{completed_task.name}_{completed_task.priority.name}"
        
        if task_signature not in self.pattern_recognition_model:
            self.pattern_recognition_model[task_signature] = {
                'count': 0,
                'avg_duration': 0.0,
                'success_rate': 0.0,
                'common_tags': set(),
                'typical_dependencies': []
            }
        
        model_entry = self.pattern_recognition_model[task_signature]
        model_entry['count'] += 1
        
        # Update duration average
        if completed_task.started_at and completed_task.completed_at:
            actual_duration = (completed_task.completed_at - completed_task.started_at).total_seconds()
            model_entry['avg_duration'] = (
                (model_entry['avg_duration'] * (model_entry['count'] - 1) + actual_duration) / 
                model_entry['count']
            )
        
        # Update success rate (assume completion = success for now)
        model_entry['success_rate'] = (
            (model_entry['success_rate'] * (model_entry['count'] - 1) + 1.0) / 
            model_entry['count']
        )
        
        # Update common patterns
        if hasattr(completed_task, 'tags'):
            model_entry['common_tags'].update(completed_task.tags)
    
    def complete_task_with_learning(self, task_id: str) -> bool:
        """Complete task and trigger adaptive learning."""
        success = self.complete_task(task_id)
        
        if success:
            task = self.get_task(task_id)
            if task:
                self.adaptive_learning_update(task)
                self._update_performance_metrics()
        
        return success
    
    def _update_performance_metrics(self) -> None:
        """Update comprehensive performance metrics."""
        completed_tasks = [t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]
        
        if not completed_tasks:
            return
        
        # Calculate average completion time
        completion_times = []
        for task in completed_tasks:
            if task.started_at and task.completed_at:
                duration = (task.completed_at - task.started_at).total_seconds()
                completion_times.append(duration)
        
        if completion_times:
            self.performance_metrics.avg_task_completion_time = sum(completion_times) / len(completion_times)
        
        # Calculate priority satisfaction rate
        high_priority_completed = sum(1 for t in completed_tasks if t.priority.value >= 3)
        total_high_priority = sum(1 for t in self.tasks.values() if t.priority.value >= 3)
        
        if total_high_priority > 0:
            self.performance_metrics.priority_satisfaction_rate = high_priority_completed / total_high_priority
        
        # Calculate quantum coherence score
        total_entanglements = sum(len(t.entangled_tasks) for t in self.tasks.values())
        if total_entanglements > 0:
            successful_entanglements = sum(
                len(t.entangled_tasks) for t in completed_tasks
            )
            self.performance_metrics.quantum_coherence_score = successful_entanglements / total_entanglements
    
    def get_scheduling_insights(self) -> Dict:
        """Get comprehensive scheduling insights and recommendations."""
        insights = {
            'performance_metrics': {
                'avg_completion_time': self.performance_metrics.avg_task_completion_time,
                'priority_satisfaction_rate': self.performance_metrics.priority_satisfaction_rate,
                'quantum_coherence_score': self.performance_metrics.quantum_coherence_score
            },
            'adaptive_parameters': {
                'learning_rate': self.adaptive_params.learning_rate,
                'temperature_adaptation_rate': self.adaptive_params.temperature_adaptation_rate,
                'entanglement_strength': self.adaptive_params.entanglement_strength
            },
            'pattern_insights': self._analyze_scheduling_patterns(),
            'optimization_recommendations': self._generate_optimization_recommendations()
        }
        
        return insights
    
    def _analyze_scheduling_patterns(self) -> Dict:
        """Analyze scheduling patterns from historical data."""
        patterns = {
            'most_common_task_types': {},
            'peak_completion_times': [],
            'dependency_patterns': {},
            'priority_distribution': defaultdict(int)
        }
        
        for task in self.tasks.values():
            patterns['priority_distribution'][task.priority.name] += 1
            
            if task.completed_at:
                hour = task.completed_at.hour
                patterns['peak_completion_times'].append(hour)
        
        return patterns
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on performance data."""
        recommendations = []
        
        if self.performance_metrics.avg_task_completion_time > 3600:  # More than 1 hour
            recommendations.append("Consider breaking down large tasks into smaller subtasks")
        
        if self.performance_metrics.priority_satisfaction_rate < 0.7:
            recommendations.append("Increase focus on high-priority tasks")
        
        if self.performance_metrics.quantum_coherence_score < 0.3:
            recommendations.append("Enhance task entanglement for better parallel execution")
        
        if len(self.running_tasks) < self.max_parallel_tasks * 0.7:
            recommendations.append("Increase parallel task execution to improve throughput")
        
        return recommendations