"""API endpoints for the quantum-inspired task planner."""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from .schemas import (
    TaskCreateRequest, TaskResponse, ScheduleOptimizeRequest, 
    ScheduleResponse, ResourceAddRequest, ResourceUtilizationResponse, 
    HealthResponse
)
from .middleware import setup_logging, SecurityHeadersMiddleware
from ..quantum_inspired_task_planner import QuantumScheduler, QuantumTask, TaskPriority, TaskStatus
from ..quantum_inspired_task_planner.resource_allocator import QuantumResourceAllocator, ResourceType
from ..quantum_inspired_task_planner.quantum_optimization import QuantumAnnealer

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Quantum-Inspired Task Planner API",
    description="REST API for quantum-inspired task scheduling and resource allocation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add security headers middleware
app.add_middleware(SecurityHeadersMiddleware)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize quantum scheduler and resource allocator
quantum_scheduler = QuantumScheduler()
resource_allocator = QuantumResourceAllocator()

# Add default resources
resource_allocator.add_resource("cpu_pool", ResourceType.CPU, 100.0)
resource_allocator.add_resource("memory_pool", ResourceType.MEMORY, 1000.0)
resource_allocator.add_resource("gpu_pool", ResourceType.GPU, 8.0)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0"
    )


@app.post("/tasks", response_model=TaskResponse)
async def create_task(request: TaskCreateRequest):
    """Create a new task in the quantum scheduler."""
    try:
        priority_map = {
            "low": TaskPriority.LOW,
            "medium": TaskPriority.MEDIUM,
            "high": TaskPriority.HIGH,
            "critical": TaskPriority.CRITICAL
        }
        
        task_id = quantum_scheduler.create_task(
            name=request.name,
            description=request.description or "",
            priority=priority_map.get(request.priority, TaskPriority.MEDIUM),
            dependencies=request.dependencies or [],
            estimated_duration=timedelta(minutes=request.estimated_duration_minutes or 60),
            resource_requirements=request.resource_requirements or {}
        )
        
        task = quantum_scheduler.get_task(task_id)
        logger.info(f"Created task: {task.name} (ID: {task_id})")
        
        return TaskResponse(
            id=task.id,
            name=task.name,
            description=task.description,
            priority=task.priority.name,
            status=task.status.value,
            dependencies=list(task.dependencies),
            estimated_duration_minutes=int(task.estimated_duration.total_seconds() / 60),
            created_at=task.created_at
        )
        
    except Exception as e:
        logger.error(f"Task creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Task creation failed: {str(e)}")


@app.get("/tasks", response_model=List[TaskResponse])
async def list_tasks(status: Optional[str] = None):
    """List all tasks with optional status filter."""
    try:
        tasks = list(quantum_scheduler.tasks.values())
        
        if status:
            status_filter = TaskStatus(status)
            tasks = [task for task in tasks if task.status == status_filter]
        
        return [
            TaskResponse(
                id=task.id,
                name=task.name,
                description=task.description,
                priority=task.priority.name,
                status=task.status.value,
                dependencies=list(task.dependencies),
                estimated_duration_minutes=int(task.estimated_duration.total_seconds() / 60),
                created_at=task.created_at
            )
            for task in tasks
        ]
        
    except Exception as e:
        logger.error(f"Task listing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Task listing failed: {str(e)}")


@app.post("/schedule/optimize", response_model=ScheduleResponse)
async def optimize_schedule(request: ScheduleOptimizeRequest):
    """Optimize task schedule using quantum algorithms."""
    try:
        annealer = QuantumAnnealer(
            initial_temperature=request.temperature or 100.0,
            cooling_rate=request.cooling_rate or 0.95
        )
        
        # Create cost function
        def cost_function(schedule_order):
            total_cost = 0.0
            for i, task_id in enumerate(schedule_order):
                task = quantum_scheduler.get_task(task_id)
                if task:
                    priority_penalty = task.priority.value * i
                    total_cost += priority_penalty
            return total_cost
        
        pending_task_ids = [task.id for task in quantum_scheduler.tasks.values() 
                           if task.status == TaskStatus.PENDING]
        
        if not pending_task_ids:
            raise HTTPException(status_code=400, detail="No pending tasks to optimize")
        
        result = annealer.anneal(
            cost_function, 
            pending_task_ids, 
            request.max_iterations or 1000
        )
        
        logger.info(f"Schedule optimization completed with energy: {result.energy}")
        
        return ScheduleResponse(
            optimal_schedule=result.optimal_schedule,
            energy=result.energy,
            iterations=result.iterations,
            convergence_achieved=result.convergence_achieved,
            execution_time=result.execution_time
        )
        
    except Exception as e:
        logger.error(f"Schedule optimization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main_quantum:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )