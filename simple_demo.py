#!/usr/bin/env python3
"""Simple demonstration of quantum task planner functionality."""

import sys
import os
from datetime import timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from quantum_inspired_task_planner.quantum_scheduler import QuantumScheduler, TaskPriority
from quantum_inspired_task_planner.advanced_scheduler import QuantumMLScheduler


def demo_basic_scheduler():
    """Demonstrate basic quantum scheduler functionality."""
    print("🔮 Basic Quantum Scheduler Demo")
    print("-" * 30)
    
    scheduler = QuantumScheduler(max_parallel_tasks=3)
    
    # Create sample tasks
    tasks = [
        ("Data Processing", TaskPriority.HIGH, 45),
        ("Model Training", TaskPriority.CRITICAL, 120),
        ("Report Generation", TaskPriority.MEDIUM, 30),
        ("Data Validation", TaskPriority.HIGH, 60),
        ("Cleanup Tasks", TaskPriority.LOW, 15)
    ]
    
    task_ids = []
    for name, priority, duration in tasks:
        task_id = scheduler.create_task(
            name=name,
            priority=priority,
            estimated_duration=timedelta(minutes=duration)
        )
        task_ids.append((task_id, name))
        print(f"✅ Created task: {name} (ID: {task_id[:8]}...)")
    
    print(f"\n📊 Task Statistics: {scheduler.get_task_statistics()}")
    
    # Get next recommended tasks
    print("\n🎯 Next Recommended Tasks:")
    next_tasks = scheduler.get_next_tasks()
    for i, task in enumerate(next_tasks, 1):
        score = scheduler._calculate_priority_score(task)
        print(f"  {i}. {task.name} (Score: {score:.2f})")
    
    # Start and complete some tasks
    print("\n⚡ Executing Tasks:")
    for task in next_tasks[:2]:  # Start first 2 tasks
        success = scheduler.start_task(task.id)
        if success:
            print(f"  🚀 Started: {task.name}")
            # Simulate completion
            scheduler.complete_task(task.id)
            print(f"  ✅ Completed: {task.name}")
    
    print(f"\n📈 Final Statistics: {scheduler.get_task_statistics()}")
    return scheduler


def demo_ml_scheduler():
    """Demonstrate ML-enhanced quantum scheduler."""
    print("\n🧠 ML-Enhanced Quantum Scheduler Demo")
    print("-" * 40)
    
    ml_scheduler = QuantumMLScheduler(max_parallel_tasks=4)
    
    # Create intelligent tasks with tags and descriptions
    intelligent_tasks = [
        {
            "name": "Image Classification Model",
            "description": "Train deep learning model for image recognition",
            "priority": TaskPriority.CRITICAL,
            "tags": ["ml", "training", "images"]
        },
        {
            "name": "Data Preprocessing",
            "description": "Clean and normalize dataset for training",
            "priority": TaskPriority.HIGH,
            "tags": ["data", "preprocessing", "ml"]
        },
        {
            "name": "Model Evaluation",
            "description": "Evaluate model performance on test set",
            "priority": TaskPriority.HIGH,
            "tags": ["ml", "evaluation", "testing"]
        },
        {
            "name": "Hyperparameter Tuning",
            "description": "Optimize model hyperparameters",
            "priority": TaskPriority.MEDIUM,
            "tags": ["ml", "optimization", "tuning"]
        }
    ]
    
    task_ids = []
    for task_config in intelligent_tasks:
        task_id = ml_scheduler.add_intelligent_task(**task_config)
        task_ids.append(task_id)
        print(f"🧠 Created intelligent task: {task_config['name']} (ID: {task_id[:8]}...)")
    
    # Get ML-enhanced recommendations
    print("\n🤖 ML-Enhanced Task Recommendations:")
    next_tasks = ml_scheduler.get_intelligent_next_tasks()
    
    for i, task in enumerate(next_tasks, 1):
        score = ml_scheduler._calculate_ml_priority_score(task)
        success_prob = getattr(task, 'predicted_success_probability', 0.8)
        complexity = getattr(task, 'complexity_score', 1.0)
        
        print(f"  {i}. {task.name}")
        print(f"     ML Score: {score:.3f} | Success Prob: {success_prob:.1%} | Complexity: {complexity:.2f}")
    
    # Execute tasks with learning
    print("\n🎓 Executing with Adaptive Learning:")
    executed = 0
    for task in next_tasks:
        if executed >= 2:  # Limit execution for demo
            break
            
        success = ml_scheduler.start_task(task.id)
        if success:
            print(f"  🚀 Started: {task.name}")
            # Simulate completion with learning
            success = ml_scheduler.complete_task_with_learning(task.id)
            if success:
                print(f"  ✅ Completed with learning: {task.name}")
                executed += 1
    
    # Show learning insights
    print("\n📊 Learning Insights:")
    insights = ml_scheduler.get_scheduling_insights()
    
    print(f"  Performance Metrics:")
    for metric, value in insights['performance_metrics'].items():
        print(f"    - {metric}: {value:.3f}")
    
    print(f"  Adaptive Parameters:")
    for param, value in insights['adaptive_parameters'].items():
        print(f"    - {param}: {value:.3f}")
    
    if insights['optimization_recommendations']:
        print(f"  Recommendations:")
        for rec in insights['optimization_recommendations']:
            print(f"    - {rec}")
    
    return ml_scheduler


def demo_quantum_optimization():
    """Demonstrate quantum optimization capabilities."""
    print("\n⚛️  Quantum Optimization Demo")
    print("-" * 30)
    
    from quantum_inspired_task_planner.simple_optimization import SimpleQuantumAnnealer
    
    # Create annealer
    annealer = SimpleQuantumAnnealer(
        initial_temperature=50.0,
        cooling_rate=0.9,
        min_temperature=0.1
    )
    
    # Define a simple scheduling problem
    tasks = ["task_a", "task_b", "task_c", "task_d", "task_e"]
    
    def cost_function(schedule):
        """Simple cost function - earlier tasks have lower cost."""
        total_cost = 0
        for i, task in enumerate(schedule):
            # Priority weights (lower is better)
            weights = {"task_a": 1, "task_b": 3, "task_c": 2, "task_d": 1, "task_e": 2}
            total_cost += weights.get(task, 1) * (i + 1)  # Position penalty
        return total_cost
    
    print(f"🎯 Optimizing schedule for tasks: {tasks}")
    print(f"📊 Initial cost: {cost_function(tasks)}")
    
    # Run optimization
    result = annealer.anneal(cost_function, tasks, max_iterations=500)
    
    print(f"\n⚡ Optimization Results:")
    print(f"  Optimal Schedule: {result.optimal_schedule}")
    print(f"  Final Cost: {result.energy:.2f}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Execution Time: {result.execution_time:.3f}s")
    print(f"  Converged: {'Yes' if result.convergence_achieved else 'No'}")
    
    return result


def main():
    """Run all demonstrations."""
    print("🚀 Quantum Task Planner Comprehensive Demo")
    print("=" * 50)
    
    try:
        # Basic scheduler demo
        basic_scheduler = demo_basic_scheduler()
        
        # ML scheduler demo
        ml_scheduler = demo_ml_scheduler()
        
        # Quantum optimization demo
        optimization_result = demo_quantum_optimization()
        
        print("\n🎉 All Demos Completed Successfully!")
        print("\n📈 Summary:")
        print(f"  Basic Scheduler Tasks: {len(basic_scheduler.tasks)}")
        print(f"  ML Scheduler Tasks: {len(ml_scheduler.tasks)}")
        print(f"  Quantum Optimization Improvement: {optimization_result.energy:.1f} cost units")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)