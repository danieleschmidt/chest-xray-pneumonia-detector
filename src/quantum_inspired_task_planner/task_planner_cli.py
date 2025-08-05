"""Command-line interface for the quantum-inspired task planner."""

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Optional

from .quantum_scheduler import QuantumScheduler, QuantumTask, TaskPriority, TaskStatus
from .resource_allocator import QuantumResourceAllocator, ResourceType
from .quantum_optimization import QuantumAnnealer


def create_scheduler_cli() -> argparse.ArgumentParser:
    """Create the main CLI parser for the task planner."""
    parser = argparse.ArgumentParser(
        description="Quantum-Inspired Task Planner CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Task management commands
    task_parser = subparsers.add_parser("task", help="Task management")
    task_subparsers = task_parser.add_subparsers(dest="task_action")
    
    # Add task
    add_parser = task_subparsers.add_parser("add", help="Add a new task")
    add_parser.add_argument("name", help="Task name")
    add_parser.add_argument("--description", default="", help="Task description")
    add_parser.add_argument("--priority", choices=["low", "medium", "high", "critical"],
                           default="medium", help="Task priority")
    add_parser.add_argument("--duration", type=int, default=60,
                           help="Estimated duration in minutes")
    add_parser.add_argument("--dependencies", nargs="*", default=[],
                           help="Task dependency IDs")
    
    # List tasks
    list_parser = task_subparsers.add_parser("list", help="List all tasks")
    list_parser.add_argument("--status", choices=["pending", "running", "completed", "blocked"],
                            help="Filter by task status")
    list_parser.add_argument("--format", choices=["table", "json"], default="table",
                            help="Output format")
    
    # Start task
    start_parser = task_subparsers.add_parser("start", help="Start a task")
    start_parser.add_argument("task_id", help="Task ID to start")
    
    # Complete task
    complete_parser = task_subparsers.add_parser("complete", help="Complete a task")
    complete_parser.add_argument("task_id", help="Task ID to complete")
    
    # Schedule optimization
    schedule_parser = subparsers.add_parser("schedule", help="Schedule optimization")
    schedule_subparsers = schedule_parser.add_subparsers(dest="schedule_action")
    
    # Optimize schedule
    optimize_parser = schedule_subparsers.add_parser("optimize", help="Optimize task schedule")
    optimize_parser.add_argument("--algorithm", choices=["annealing", "variational"],
                                default="annealing", help="Optimization algorithm")
    optimize_parser.add_argument("--max-iterations", type=int, default=1000,
                                help="Maximum optimization iterations")
    
    # Show next tasks
    next_parser = schedule_subparsers.add_parser("next", help="Show next recommended tasks")
    next_parser.add_argument("--count", type=int, default=5, help="Number of tasks to show")
    
    # Resource management
    resource_parser = subparsers.add_parser("resource", help="Resource management")
    resource_subparsers = resource_parser.add_subparsers(dest="resource_action")
    
    # Add resource
    add_res_parser = resource_subparsers.add_parser("add", help="Add a resource")
    add_res_parser.add_argument("resource_id", help="Resource identifier")
    add_res_parser.add_argument("type", choices=["cpu", "memory", "gpu", "storage", "network"],
                               help="Resource type")
    add_res_parser.add_argument("capacity", type=float, help="Total resource capacity")
    
    # Show resource utilization
    util_parser = resource_subparsers.add_parser("utilization", help="Show resource utilization")
    util_parser.add_argument("--format", choices=["table", "json"], default="table",
                            help="Output format")
    
    # State management
    state_parser = subparsers.add_parser("state", help="State management")
    state_subparsers = state_parser.add_subparsers(dest="state_action")
    
    # Export state
    export_parser = state_subparsers.add_parser("export", help="Export scheduler state")
    export_parser.add_argument("--output", help="Output file path")
    
    # Import state
    import_parser = state_subparsers.add_parser("import", help="Import scheduler state")
    import_parser.add_argument("input_file", help="Input file path")
    
    return parser


def format_task_table(tasks: List[QuantumTask]) -> str:
    """Format tasks as a table."""
    if not tasks:
        return "No tasks found."
    
    # Calculate column widths
    max_name_len = max(len(task.name) for task in tasks)
    max_id_len = max(len(task.id[:8]) for task in tasks)
    
    header = f"{'ID':<{max_id_len}} {'Name':<{max_name_len}} {'Priority':<8} {'Status':<10} {'Dependencies':<12}"
    separator = "-" * len(header)
    
    lines = [header, separator]
    
    for task in tasks:
        dep_count = len(task.dependencies)
        dep_str = f"{dep_count} deps" if dep_count > 0 else "None"
        
        line = (f"{task.id[:8]:<{max_id_len}} "
                f"{task.name:<{max_name_len}} "
                f"{task.priority.name:<8} "
                f"{task.status.value:<10} "
                f"{dep_str:<12}")
        lines.append(line)
    
    return "\n".join(lines)


def format_resource_table(utilization: Dict[str, Dict]) -> str:
    """Format resource utilization as a table."""
    if not utilization:
        return "No resources found."
    
    header = f"{'Resource ID':<15} {'Type':<8} {'Capacity':<10} {'Available':<10} {'Utilization':<12} {'Tasks':<6}"
    separator = "-" * len(header)
    
    lines = [header, separator]
    
    for resource_id, data in utilization.items():
        util_pct = f"{data['utilization_percent']:.1f}%"
        task_count = data['allocated_tasks_count']
        
        line = (f"{resource_id:<15} "
                f"{data['type']:<8} "
                f"{data['total_capacity']:<10.1f} "
                f"{data['available_capacity']:<10.1f} "
                f"{util_pct:<12} "
                f"{task_count:<6}")
        lines.append(line)
    
    return "\n".join(lines)


def main():
    """Main CLI entry point."""
    parser = create_scheduler_cli()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize scheduler and resource allocator
    scheduler = QuantumScheduler()
    resource_allocator = QuantumResourceAllocator()
    
    # Add default resources
    resource_allocator.add_resource("cpu_pool", ResourceType.CPU, 100.0)
    resource_allocator.add_resource("memory_pool", ResourceType.MEMORY, 1000.0)
    
    try:
        if args.command == "task":
            handle_task_command(args, scheduler)
        elif args.command == "schedule":
            handle_schedule_command(args, scheduler)
        elif args.command == "resource":
            handle_resource_command(args, resource_allocator)
        elif args.command == "state":
            handle_state_command(args, scheduler)
        else:
            parser.print_help()
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def handle_task_command(args, scheduler: QuantumScheduler):
    """Handle task-related commands."""
    if args.task_action == "add":
        priority_map = {
            "low": TaskPriority.LOW,
            "medium": TaskPriority.MEDIUM,
            "high": TaskPriority.HIGH,
            "critical": TaskPriority.CRITICAL
        }
        
        task_id = scheduler.create_task(
            name=args.name,
            description=args.description,
            priority=priority_map[args.priority],
            dependencies=args.dependencies,
            estimated_duration=timedelta(minutes=args.duration)
        )
        print(f"Created task: {task_id}")
    
    elif args.task_action == "list":
        tasks = list(scheduler.tasks.values())
        
        if args.status:
            status_filter = TaskStatus(args.status)
            tasks = [task for task in tasks if task.status == status_filter]
        
        if args.format == "json":
            task_data = [
                {
                    "id": task.id,
                    "name": task.name,
                    "priority": task.priority.name,
                    "status": task.status.value,
                    "dependencies": list(task.dependencies)
                }
                for task in tasks
            ]
            print(json.dumps(task_data, indent=2))
        else:
            print(format_task_table(tasks))
    
    elif args.task_action == "start":
        success = scheduler.start_task(args.task_id)
        if success:
            print(f"Started task: {args.task_id}")
        else:
            print(f"Failed to start task: {args.task_id}")
    
    elif args.task_action == "complete":
        success = scheduler.complete_task(args.task_id)
        if success:
            print(f"Completed task: {args.task_id}")
        else:
            print(f"Failed to complete task: {args.task_id}")


def handle_schedule_command(args, scheduler: QuantumScheduler):
    """Handle schedule-related commands."""
    if args.schedule_action == "optimize":
        if args.algorithm == "annealing":
            annealer = QuantumAnnealer()
            
            # Create cost function for current tasks
            def cost_function(schedule_order):
                total_cost = 0.0
                for i, task_id in enumerate(schedule_order):
                    task = scheduler.get_task(task_id)
                    if task:
                        # Cost increases with delay for high-priority tasks
                        priority_penalty = task.priority.value * i
                        total_cost += priority_penalty
                return total_cost
            
            pending_task_ids = [task.id for task in scheduler.tasks.values() 
                               if task.status == TaskStatus.PENDING]
            
            if pending_task_ids:
                result = annealer.anneal(cost_function, pending_task_ids, args.max_iterations)
                print(f"Optimization completed in {result.execution_time:.2f}s")
                print(f"Optimal energy: {result.energy:.2f}")
                print(f"Convergence: {'Yes' if result.convergence_achieved else 'No'}")
                print(f"Optimal schedule: {result.optimal_schedule}")
            else:
                print("No pending tasks to optimize.")
    
    elif args.schedule_action == "next":
        next_tasks = scheduler.get_next_tasks()
        if next_tasks:
            print(f"Next {min(args.count, len(next_tasks))} recommended tasks:")
            for i, task in enumerate(next_tasks[:args.count]):
                score = scheduler._calculate_priority_score(task)
                print(f"{i+1}. {task.name} (ID: {task.id[:8]}, Score: {score:.2f})")
        else:
            print("No tasks available for execution.")


def handle_resource_command(args, allocator: QuantumResourceAllocator):
    """Handle resource-related commands."""
    if args.resource_action == "add":
        resource_type = ResourceType(args.type)
        allocator.add_resource(args.resource_id, resource_type, args.capacity)
        print(f"Added {args.type} resource '{args.resource_id}' with capacity {args.capacity}")
    
    elif args.resource_action == "utilization":
        utilization = allocator.get_resource_utilization()
        
        if args.format == "json":
            print(json.dumps(utilization, indent=2))
        else:
            print(format_resource_table(utilization))


def handle_state_command(args, scheduler: QuantumScheduler):
    """Handle state management commands."""
    if args.state_action == "export":
        state_json = scheduler.export_state()
        
        if args.output:
            Path(args.output).write_text(state_json)
            print(f"State exported to: {args.output}")
        else:
            print(state_json)
    
    elif args.state_action == "import":
        state_json = Path(args.input_file).read_text()
        scheduler.import_state(state_json)
        print(f"State imported from: {args.input_file}")


if __name__ == "__main__":
    main()