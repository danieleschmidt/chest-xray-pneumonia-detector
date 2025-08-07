#!/usr/bin/env python3
"""Research demonstration for quantum-inspired task scheduling."""

import sys
import os
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from quantum_inspired_task_planner.research_framework import QuantumSchedulingResearchFramework


def main():
    """Run comprehensive research demonstration."""
    print("🚀 Quantum Task Scheduling Research Demo")
    print("=" * 50)
    
    # Initialize research framework
    research = QuantumSchedulingResearchFramework()
    
    # Configure algorithms to test
    algorithm_configs = {
        "quantum_ml": {
            "type": "quantum_ml",
            "parallel_tasks": 4,
            "max_iterations": 100,
            "trials": 3
        },
        "quantum_annealing": {
            "type": "quantum_annealing",
            "parallel_tasks": 4,
            "cooling_rate": 0.95,
            "trials": 3
        },
        "fcfs": {
            "type": "baseline",
            "trials": 3
        },
        "priority": {
            "type": "baseline", 
            "trials": 3
        },
        "sjf": {
            "type": "baseline",
            "trials": 3
        }
    }
    
    # Run comparative study
    print("🔬 Starting Comparative Research Study...")
    results = research.run_comparative_study(algorithm_configs)
    
    # Analyze statistical significance
    print("\n📊 Analyzing Statistical Significance...")
    statistical_analysis = research.analyze_statistical_significance(results)
    
    # Generate research report
    print("\n📝 Generating Research Report...")
    report = research.generate_research_report(results, statistical_analysis)
    
    # Save report
    report_file = "quantum_scheduling_research_report.md"
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"✅ Research report saved: {report_file}")
    
    # Export experimental data
    print("\n💾 Exporting Experimental Data...")
    data_file = research.export_experimental_data(results)
    print(f"✅ Experimental data exported: {data_file}")
    
    # Create visualization data
    print("\n📈 Creating Visualization Data...")
    viz_data = research.create_visualization_data(statistical_analysis)
    
    viz_file = "visualization_data.json"
    with open(viz_file, 'w') as f:
        json.dump(viz_data, f, indent=2)
    print(f"✅ Visualization data saved: {viz_file}")
    
    # Print summary
    print("\n🎯 Research Summary:")
    print("-" * 30)
    
    total_experiments = sum(len(result_list) for result_list in results.values())
    print(f"Total experiments: {total_experiments}")
    print(f"Algorithms tested: {len(algorithm_configs)}")
    print(f"Benchmark suites: {len(research.benchmark_suites)}")
    
    # Show best performing algorithms per benchmark
    print("\n🏆 Best Performing Algorithms:")
    for benchmark, analysis in statistical_analysis.items():
        if 'best_algorithm' in analysis:
            best = analysis['best_algorithm']
            print(f"  {benchmark}: {best['name']} ({best['improvement_factor']:.2f}x improvement)")
    
    print(f"\n📄 Full results in: {report_file}")
    print("🚀 Research demonstration complete!")


if __name__ == "__main__":
    main()