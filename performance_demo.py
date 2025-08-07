#!/usr/bin/env python3
"""Performance optimization demonstration."""

import sys
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from quantum_inspired_task_planner.advanced_scheduler import QuantumMLScheduler
from quantum_inspired_task_planner.performance_optimizer import (
    PerformanceOptimizer, OptimizationStrategy, PerformanceProfiler, 
    QuantumCacheManager, AutoScaler, PerformanceMetrics
)
from quantum_inspired_task_planner.quantum_scheduler import TaskPriority


def demo_performance_profiler():
    """Demonstrate performance profiling capabilities."""
    print("⚡ Performance Profiler Demo")
    print("-" * 30)
    
    profiler = PerformanceProfiler()
    
    # Simulate various operations with profiling
    operations = [
        ("task_creation", 0.05),
        ("task_scheduling", 0.02),
        ("task_execution", 0.5),
        ("data_processing", 0.8),  # Simulate bottleneck
        ("cache_lookup", 0.001)
    ]
    
    print("🔍 Profiling operations...")
    for op_name, duration in operations:
        for _ in range(10):
            with profiler.profile_operation(op_name):
                time.sleep(duration / 10)  # Simulate work
    
    # Get performance summary
    summary = profiler.get_performance_summary()
    
    print(f"\n📊 Performance Summary:")
    print(f"  Total operations: {summary['total_operations']}")
    print(f"  Bottlenecks detected: {len(summary['bottlenecks'])}")
    
    print(f"\n⚠️  Operation Performance:")
    for op_name, stats in summary['operations'].items():
        print(f"  {op_name}:")
        print(f"    Count: {stats['count']}")
        print(f"    Avg time: {stats['avg_time']:.4f}s")
        print(f"    Min/Max: {stats['min_time']:.4f}s / {stats['max_time']:.4f}s")
    
    if summary['bottlenecks']:
        print(f"\n🚨 Bottlenecks detected:")
        for op_name, avg_time in summary['bottlenecks'].items():
            print(f"  {op_name}: {avg_time:.4f}s average")
    
    return profiler


def demo_quantum_cache():
    """Demonstrate quantum-inspired caching."""
    print("\n🧠 Quantum Cache Demo")
    print("-" * 25)
    
    cache = QuantumCacheManager(max_size=100)
    
    # Simulate cache usage patterns
    print("💾 Testing cache operations...")
    
    # Store related data
    cache.put("user_123_profile", {"name": "John Doe", "role": "admin"})
    cache.put("user_123_tasks", [{"id": 1, "name": "Task A"}])
    cache.put("user_123_settings", {"theme": "dark", "notifications": True})
    
    # Access patterns that create coherence
    print("🔗 Creating quantum coherence...")
    for _ in range(5):
        cache.get("user_123_profile")
        cache.get("user_123_tasks")
        time.sleep(0.01)
    
    # Test coherence detection
    coherent_keys = cache.get_coherent_keys("user_123_profile")
    print(f"Keys coherent with 'user_123_profile': {coherent_keys}")
    
    # Cache statistics
    stats = cache.get_cache_stats()
    print(f"\n📈 Cache Statistics:")
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Total size estimate: {stats['total_size_estimate']} bytes")
    print(f"  Average coherence: {stats['avg_coherence']:.3f}")
    print(f"  Most accessed: {stats['most_accessed']}")
    
    return cache


def demo_auto_scaler():
    """Demonstrate auto-scaling capabilities."""
    print("\n📈 Auto-Scaler Demo")
    print("-" * 20)
    
    scaler = AutoScaler(min_capacity=2, max_capacity=16)
    
    print("🎯 Simulating varying load conditions...")
    
    # Simulate different load scenarios
    scenarios = [
        ("Low load", PerformanceMetrics(throughput=1.0, cpu_utilization=20.0, queue_length=2)),
        ("Medium load", PerformanceMetrics(throughput=5.0, cpu_utilization=50.0, queue_length=10)),
        ("High load", PerformanceMetrics(throughput=10.0, cpu_utilization=85.0, queue_length=25)),
        ("Overload", PerformanceMetrics(throughput=15.0, cpu_utilization=95.0, queue_length=50)),
        ("Returning to normal", PerformanceMetrics(throughput=3.0, cpu_utilization=40.0, queue_length=5))
    ]
    
    for scenario_name, metrics in scenarios:
        print(f"\n📊 Scenario: {scenario_name}")
        print(f"  Metrics: CPU={metrics.cpu_utilization}%, Queue={metrics.queue_length}, Throughput={metrics.throughput}")
        
        scaler.add_metrics(metrics)
        decision = scaler.make_scaling_decision(OptimizationStrategy.BALANCED)
        
        print(f"  Decision: {decision.action}")
        print(f"  Target capacity: {decision.current_capacity} → {decision.target_capacity}")
        print(f"  Confidence: {decision.confidence:.2f}")
        print(f"  Reasoning: {'; '.join(decision.reasoning)}")
        
        # Simulate time passing for cooldown
        scaler.last_scaling_action -= scaler.cooldown_period
    
    return scaler


def demo_integrated_performance_optimizer():
    """Demonstrate integrated performance optimizer."""
    print("\n🚀 Integrated Performance Optimizer Demo")
    print("-" * 42)
    
    # Initialize components
    scheduler = QuantumMLScheduler(max_parallel_tasks=4)
    optimizer = PerformanceOptimizer(scheduler)
    
    # Create test workload
    print("📝 Creating test workload...")
    task_ids = []
    for i in range(20):
        priority = TaskPriority.HIGH if i < 5 else TaskPriority.MEDIUM
        task_id = scheduler.add_intelligent_task(
            name=f"Performance Test Task {i+1}",
            description=f"Task for performance testing {i+1}",
            priority=priority,
            tags=["performance", "test"]
        )
        task_ids.append(task_id)
    
    print(f"✅ Created {len(task_ids)} test tasks")
    
    # Start performance optimization
    print("\n⚡ Starting performance optimization...")
    optimizer.start_optimization(OptimizationStrategy.BALANCED)
    
    # Simulate task execution with profiling
    print("🔄 Simulating task execution...")
    
    def simulate_task_work(task_id):
        """Simulate task work with profiling."""
        with optimizer.profiler.profile_operation("task_execution"):
            time.sleep(0.1)  # Simulate work
            return f"Completed {task_id[:8]}..."
    
    # Execute some tasks in parallel
    task_functions = [lambda tid=task_id: simulate_task_work(tid) for task_id in task_ids[:10]]
    
    start_time = time.time()
    results = optimizer.execute_parallel_tasks(task_functions, max_workers=8)
    execution_time = time.time() - start_time
    
    print(f"✅ Executed {len(results)} tasks in {execution_time:.2f}s")
    print(f"  Throughput: {len(results)/execution_time:.2f} tasks/second")
    
    # Let optimizer collect metrics
    time.sleep(2)
    
    # Get performance report
    print("\n📊 Performance Report:")
    report = optimizer.get_performance_report()
    
    print(f"  Current capacity: {report['current_capacity']['thread_pool_workers']} workers")
    print(f"  Total profiled operations: {report['profiler_summary']['total_operations']}")
    print(f"  Cache entries: {report['cache_statistics']['total_entries']}")
    print(f"  Cache coherence: {report['cache_statistics']['avg_coherence']:.3f}")
    
    if report['recent_scaling_decisions']:
        print(f"  Recent scaling decisions: {len(report['recent_scaling_decisions'])}")
        for decision in report['recent_scaling_decisions'][-3:]:  # Show last 3
            print(f"    {decision['action']}: {decision['target_capacity']} workers ({decision['confidence']:.2f} confidence)")
    
    if report['performance_recommendations']:
        print(f"\n💡 Performance Recommendations:")
        for rec in report['performance_recommendations']:
            print(f"  - {rec}")
    
    # Stop optimization
    optimizer.stop_optimization()
    
    return optimizer


def demo_parallel_execution_benchmark():
    """Benchmark parallel execution performance."""
    print("\n🏁 Parallel Execution Benchmark")
    print("-" * 35)
    
    def cpu_intensive_task(n: int) -> int:
        """Simulate CPU-intensive work."""
        result = 0
        for i in range(n * 1000):
            result += i ** 2
        return result
    
    task_counts = [1, 2, 4, 8, 16]
    work_size = 1000
    
    print(f"🔢 Benchmarking with work size: {work_size}")
    
    for task_count in task_counts:
        print(f"\n📊 Testing {task_count} tasks:")
        
        # Sequential execution
        start_time = time.time()
        sequential_results = [cpu_intensive_task(work_size) for _ in range(task_count)]
        sequential_time = time.time() - start_time
        
        # Parallel execution
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=8) as executor:
            parallel_results = list(executor.map(cpu_intensive_task, [work_size] * task_count))
        parallel_time = time.time() - start_time
        
        # Calculate speedup
        speedup = sequential_time / parallel_time if parallel_time > 0 else 0
        
        print(f"  Sequential: {sequential_time:.3f}s")
        print(f"  Parallel:   {parallel_time:.3f}s")
        print(f"  Speedup:    {speedup:.2f}x")
        print(f"  Efficiency: {speedup/min(task_count, 8):.1%}")


def main():
    """Run comprehensive performance optimization demonstration."""
    print("⚡ Quantum Task Scheduler - Performance Optimization Demo")
    print("=" * 60)
    
    try:
        # Individual component demos
        profiler = demo_performance_profiler()
        cache = demo_quantum_cache()
        scaler = demo_auto_scaler()
        
        # Integrated optimizer demo
        optimizer = demo_integrated_performance_optimizer()
        
        # Parallel execution benchmark
        demo_parallel_execution_benchmark()
        
        print("\n🎉 Performance optimization demonstration completed!")
        print("\n📝 Key Features Demonstrated:")
        print("  ✅ Performance profiling and bottleneck detection")
        print("  ✅ Quantum-inspired caching with coherence tracking")
        print("  ✅ Intelligent auto-scaling based on metrics")
        print("  ✅ Parallel task execution optimization")
        print("  ✅ Real-time performance monitoring")
        print("  ✅ Dynamic resource allocation")
        print("  ✅ Performance recommendations generation")
        print("  ✅ Comprehensive performance reporting")
        
    except Exception as e:
        print(f"\n❌ Performance demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)