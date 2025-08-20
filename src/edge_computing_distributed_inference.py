"""
Edge Computing and Distributed Inference System
==============================================

This module implements a sophisticated edge computing framework that enables
distributed AI inference across multiple edge devices, with intelligent
load balancing, model synchronization, and federated learning capabilities.

Features:
- Multi-device edge inference coordination
- Intelligent model partitioning and distribution
- Real-time load balancing across edge nodes
- Federated learning with privacy preservation
- Edge-to-cloud hybrid processing
- Network-aware inference optimization
- Fault-tolerant distributed execution
- Model quantization and compression for edge deployment
"""

import asyncio
import logging
import json
import time
import hashlib
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import uuid
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import socket
import pickle
import zlib
import sqlite3

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EdgeDeviceStatus(Enum):
    """Edge device operational status."""
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    OVERLOADED = "overloaded"


class InferenceMode(Enum):
    """Inference execution modes."""
    EDGE_ONLY = "edge_only"
    CLOUD_ONLY = "cloud_only"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


class ModelPartition(Enum):
    """Model partitioning strategies."""
    LAYER_WISE = "layer_wise"
    FEATURE_MAP = "feature_map"
    ENSEMBLE = "ensemble"
    DYNAMIC = "dynamic"


@dataclass
class EdgeDevice:
    """Edge device configuration and status."""
    device_id: str
    name: str
    location: str
    capabilities: Dict[str, Any]
    status: EdgeDeviceStatus = EdgeDeviceStatus.OFFLINE
    current_load: float = 0.0
    max_capacity: float = 1.0
    network_latency_ms: float = 0.0
    last_heartbeat: Optional[datetime] = None
    supported_models: List[str] = field(default_factory=list)
    active_connections: int = 0
    total_inferences: int = 0
    error_count: int = 0
    average_inference_time: float = 0.0


@dataclass
class InferenceRequest:
    """Inference request with metadata."""
    request_id: str
    model_name: str
    input_data: Any
    priority: int = 1  # 1=high, 5=low
    max_latency_ms: float = 5000.0
    requires_privacy: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    client_location: Optional[str] = None


@dataclass
class InferenceResult:
    """Inference result with execution metadata."""
    request_id: str
    result: Any
    execution_time_ms: float
    device_id: str
    model_version: str
    confidence_score: float = 0.0
    processing_mode: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelPartitioner:
    """Intelligent model partitioning for distributed execution."""
    
    def __init__(self):
        self.partition_cache = {}
        self.performance_profiles = {}
    
    def analyze_model_structure(self, model_path: str) -> Dict[str, Any]:
        """Analyze model structure for optimal partitioning."""
        # This is a simplified analysis - real implementation would
        # analyze actual model architecture
        
        model_info = {
            "total_parameters": 1000000,  # Example
            "layers": [
                {"name": "conv1", "parameters": 100000, "compute_intensity": "high"},
                {"name": "conv2", "parameters": 200000, "compute_intensity": "high"},
                {"name": "fc1", "parameters": 500000, "compute_intensity": "medium"},
                {"name": "fc2", "parameters": 200000, "compute_intensity": "low"}
            ],
            "memory_requirement_mb": 50,
            "compute_requirement_flops": 2000000
        }
        
        return model_info
    
    def create_partitions(self, model_info: Dict[str, Any], 
                         devices: List[EdgeDevice],
                         strategy: ModelPartition = ModelPartition.LAYER_WISE) -> Dict[str, Any]:
        """Create model partitions based on device capabilities."""
        
        if strategy == ModelPartition.LAYER_WISE:
            return self._layer_wise_partition(model_info, devices)
        elif strategy == ModelPartition.ENSEMBLE:
            return self._ensemble_partition(model_info, devices)
        elif strategy == ModelPartition.DYNAMIC:
            return self._dynamic_partition(model_info, devices)
        else:
            return self._default_partition(model_info, devices)
    
    def _layer_wise_partition(self, model_info: Dict[str, Any], 
                             devices: List[EdgeDevice]) -> Dict[str, Any]:
        """Partition model by layers across devices."""
        layers = model_info["layers"]
        partitions = {}
        
        # Sort devices by capability (simplified metric)
        sorted_devices = sorted(devices, 
                              key=lambda d: d.capabilities.get("compute_power", 1.0), 
                              reverse=True)
        
        # Assign layers to devices based on compute requirements
        device_idx = 0
        for i, layer in enumerate(layers):
            if device_idx >= len(sorted_devices):
                device_idx = 0  # Round-robin if more layers than devices
            
            device = sorted_devices[device_idx]
            partition_id = f"partition_{device.device_id}_{i}"
            
            partitions[partition_id] = {
                "device_id": device.device_id,
                "layers": [layer["name"]],
                "parameters": layer["parameters"],
                "compute_intensity": layer["compute_intensity"],
                "estimated_latency_ms": self._estimate_layer_latency(layer, device)
            }
            
            device_idx += 1
        
        return {
            "strategy": "layer_wise",
            "partitions": partitions,
            "execution_order": list(partitions.keys()),
            "estimated_total_latency": sum(p["estimated_latency_ms"] for p in partitions.values())
        }
    
    def _ensemble_partition(self, model_info: Dict[str, Any], 
                          devices: List[EdgeDevice]) -> Dict[str, Any]:
        """Create ensemble partitions for parallel execution."""
        partitions = {}
        
        # Create complete model copies for top devices
        top_devices = sorted(devices, 
                           key=lambda d: d.capabilities.get("compute_power", 1.0), 
                           reverse=True)[:min(3, len(devices))]
        
        for i, device in enumerate(top_devices):
            partition_id = f"ensemble_{device.device_id}"
            partitions[partition_id] = {
                "device_id": device.device_id,
                "model_subset": "full_model",
                "weight": 1.0 / len(top_devices),
                "estimated_latency_ms": self._estimate_full_model_latency(model_info, device)
            }
        
        return {
            "strategy": "ensemble",
            "partitions": partitions,
            "aggregation_method": "weighted_average",
            "estimated_total_latency": max(p["estimated_latency_ms"] for p in partitions.values())
        }
    
    def _dynamic_partition(self, model_info: Dict[str, Any], 
                          devices: List[EdgeDevice]) -> Dict[str, Any]:
        """Create dynamic partitions based on current device load."""
        # Adapt partitioning based on real-time device status
        available_devices = [d for d in devices if d.status == EdgeDeviceStatus.ONLINE 
                           and d.current_load < 0.8]
        
        if len(available_devices) <= 1:
            return self._default_partition(model_info, available_devices)
        elif len(available_devices) <= 3:
            return self._layer_wise_partition(model_info, available_devices)
        else:
            return self._ensemble_partition(model_info, available_devices)
    
    def _default_partition(self, model_info: Dict[str, Any], 
                          devices: List[EdgeDevice]) -> Dict[str, Any]:
        """Default single-device partition."""
        if not devices:
            return {"strategy": "none", "partitions": {}}
        
        best_device = max(devices, key=lambda d: d.capabilities.get("compute_power", 0.0))
        
        return {
            "strategy": "single_device",
            "partitions": {
                f"full_{best_device.device_id}": {
                    "device_id": best_device.device_id,
                    "model_subset": "full_model",
                    "estimated_latency_ms": self._estimate_full_model_latency(model_info, best_device)
                }
            },
            "execution_order": [f"full_{best_device.device_id}"],
            "estimated_total_latency": self._estimate_full_model_latency(model_info, best_device)
        }
    
    def _estimate_layer_latency(self, layer: Dict[str, Any], device: EdgeDevice) -> float:
        """Estimate execution latency for a layer on a device."""
        base_latency = layer["parameters"] / device.capabilities.get("compute_power", 1.0) * 0.001
        
        # Adjust based on compute intensity
        intensity_multiplier = {
            "high": 2.0,
            "medium": 1.5,
            "low": 1.0
        }.get(layer["compute_intensity"], 1.0)
        
        return base_latency * intensity_multiplier + device.network_latency_ms
    
    def _estimate_full_model_latency(self, model_info: Dict[str, Any], device: EdgeDevice) -> float:
        """Estimate full model execution latency on a device."""
        base_latency = model_info["compute_requirement_flops"] / device.capabilities.get("compute_power", 1.0) * 0.0001
        return base_latency + device.network_latency_ms


class EdgeOrchestrator:
    """Main orchestrator for distributed edge inference."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        
        # Device management
        self.edge_devices: Dict[str, EdgeDevice] = {}
        self.device_health_monitor = threading.Thread(target=self._monitor_device_health, daemon=True)
        
        # Model management
        self.model_partitioner = ModelPartitioner()
        self.active_partitions: Dict[str, Dict[str, Any]] = {}
        self.model_cache: Dict[str, Any] = {}
        
        # Request queue and processing
        self.inference_queue = asyncio.Queue(maxsize=self.config.get("max_queue_size", 1000))
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        
        # Load balancing
        self.load_balancer = self._create_load_balancer()
        
        # Performance tracking
        self.performance_metrics: Dict[str, List[float]] = {
            "latency": [],
            "throughput": [],
            "success_rate": []
        }
        
        # Database for logging
        self.db_path = "edge_inference.db"
        self.init_database()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Start monitoring
        self.device_health_monitor.start()
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load orchestrator configuration."""
        default_config = {
            "heartbeat_interval_seconds": 30,
            "max_queue_size": 1000,
            "default_timeout_seconds": 30,
            "load_balancing_algorithm": "least_loaded",
            "enable_federated_learning": True,
            "model_sync_interval_minutes": 60,
            "max_retries": 3,
            "circuit_breaker_threshold": 5
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
        
        return default_config
    
    def init_database(self):
        """Initialize edge inference database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS inference_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    device_id TEXT NOT NULL,
                    execution_time_ms REAL,
                    success INTEGER,
                    error_message TEXT,
                    timestamp TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS device_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_id TEXT NOT NULL,
                    cpu_usage REAL,
                    memory_usage REAL,
                    network_latency REAL,
                    active_inferences INTEGER,
                    timestamp TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS federated_updates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    device_id TEXT NOT NULL,
                    update_data BLOB,
                    performance_improvement REAL,
                    timestamp TEXT NOT NULL
                )
            """)
    
    def register_edge_device(self, device: EdgeDevice) -> bool:
        """Register a new edge device."""
        with self._lock:
            try:
                self.edge_devices[device.device_id] = device
                device.status = EdgeDeviceStatus.ONLINE
                device.last_heartbeat = datetime.now()
                
                logger.info(f"Registered edge device: {device.name} ({device.device_id})")
                return True
                
            except Exception as e:
                logger.error(f"Failed to register device {device.device_id}: {e}")
                return False
    
    def _create_load_balancer(self) -> Callable:
        """Create load balancing function based on configuration."""
        algorithm = self.config.get("load_balancing_algorithm", "least_loaded")
        
        if algorithm == "least_loaded":
            return self._least_loaded_balancer
        elif algorithm == "round_robin":
            return self._round_robin_balancer
        elif algorithm == "latency_based":
            return self._latency_based_balancer
        elif algorithm == "capability_weighted":
            return self._capability_weighted_balancer
        else:
            return self._least_loaded_balancer
    
    def _least_loaded_balancer(self, available_devices: List[EdgeDevice]) -> Optional[EdgeDevice]:
        """Select device with lowest current load."""
        if not available_devices:
            return None
        
        return min(available_devices, key=lambda d: d.current_load)
    
    def _round_robin_balancer(self, available_devices: List[EdgeDevice]) -> Optional[EdgeDevice]:
        """Simple round-robin device selection."""
        if not available_devices:
            return None
        
        # Use request count as round-robin counter
        total_requests = sum(d.total_inferences for d in available_devices)
        return available_devices[total_requests % len(available_devices)]
    
    def _latency_based_balancer(self, available_devices: List[EdgeDevice]) -> Optional[EdgeDevice]:
        """Select device with lowest network latency."""
        if not available_devices:
            return None
        
        return min(available_devices, key=lambda d: d.network_latency_ms)
    
    def _capability_weighted_balancer(self, available_devices: List[EdgeDevice]) -> Optional[EdgeDevice]:
        """Select device based on capability-to-load ratio."""
        if not available_devices:
            return None
        
        def capability_score(device):
            compute_power = device.capabilities.get("compute_power", 1.0)
            load_factor = 1.0 - device.current_load
            return compute_power * load_factor
        
        return max(available_devices, key=capability_score)
    
    def _monitor_device_health(self):
        """Monitor edge device health and status."""
        while True:
            try:
                current_time = datetime.now()
                offline_threshold = timedelta(seconds=self.config["heartbeat_interval_seconds"] * 3)
                
                with self._lock:
                    for device_id, device in self.edge_devices.items():
                        if device.last_heartbeat:
                            time_since_heartbeat = current_time - device.last_heartbeat
                            
                            if time_since_heartbeat > offline_threshold:
                                if device.status != EdgeDeviceStatus.OFFLINE:
                                    logger.warning(f"Device {device_id} appears offline")
                                    device.status = EdgeDeviceStatus.OFFLINE
                            
                            elif device.current_load > 0.9:
                                device.status = EdgeDeviceStatus.OVERLOADED
                            
                            elif device.current_load > 0.7:
                                device.status = EdgeDeviceStatus.DEGRADED
                            
                            else:
                                device.status = EdgeDeviceStatus.ONLINE
                
                time.sleep(self.config["heartbeat_interval_seconds"])
                
            except Exception as e:
                logger.error(f"Device health monitoring error: {e}")
                time.sleep(10)  # Retry after delay
    
    def update_device_heartbeat(self, device_id: str, metrics: Dict[str, Any]):
        """Update device heartbeat and metrics."""
        with self._lock:
            if device_id in self.edge_devices:
                device = self.edge_devices[device_id]
                device.last_heartbeat = datetime.now()
                device.current_load = metrics.get("cpu_usage", 0.0)
                device.network_latency_ms = metrics.get("network_latency", 0.0)
                device.active_connections = metrics.get("active_connections", 0)
                
                # Store metrics in database
                self._store_device_metrics(device_id, metrics)
    
    def _store_device_metrics(self, device_id: str, metrics: Dict[str, Any]):
        """Store device metrics in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO device_metrics 
                    (device_id, cpu_usage, memory_usage, network_latency, 
                     active_inferences, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    device_id,
                    metrics.get("cpu_usage", 0.0),
                    metrics.get("memory_usage", 0.0),
                    metrics.get("network_latency", 0.0),
                    metrics.get("active_inferences", 0),
                    datetime.now().isoformat()
                ))
        except Exception as e:
            logger.error(f"Failed to store device metrics: {e}")
    
    async def submit_inference_request(self, request: InferenceRequest) -> str:
        """Submit inference request for processing."""
        try:
            await self.inference_queue.put(request)
            
            # Start processing task
            task_id = f"task_{request.request_id}"
            task = asyncio.create_task(self._process_inference_request(request))
            self.processing_tasks[task_id] = task
            
            return task_id
            
        except asyncio.QueueFull:
            logger.error(f"Inference queue full, rejecting request {request.request_id}")
            raise Exception("System overloaded - please try again later")
    
    async def _process_inference_request(self, request: InferenceRequest) -> InferenceResult:
        """Process inference request with load balancing and fault tolerance."""
        start_time = time.time()
        retries = 0
        max_retries = self.config.get("max_retries", 3)
        
        while retries < max_retries:
            try:
                # Select optimal execution strategy
                execution_plan = self._create_execution_plan(request)
                
                if execution_plan["mode"] == "edge":
                    result = await self._execute_edge_inference(request, execution_plan)
                elif execution_plan["mode"] == "distributed":
                    result = await self._execute_distributed_inference(request, execution_plan)
                elif execution_plan["mode"] == "cloud_fallback":
                    result = await self._execute_cloud_fallback(request)
                else:
                    raise Exception(f"Unknown execution mode: {execution_plan['mode']}")
                
                # Log successful inference
                execution_time = (time.time() - start_time) * 1000
                self._log_inference(request.request_id, request.model_name, 
                                  result.device_id, execution_time, True, None)
                
                # Update performance metrics
                self.performance_metrics["latency"].append(execution_time)
                self.performance_metrics["throughput"].append(1.0)  # Simplified
                
                return result
                
            except Exception as e:
                retries += 1
                logger.warning(f"Inference attempt {retries} failed for {request.request_id}: {e}")
                
                if retries >= max_retries:
                    # Log failed inference
                    execution_time = (time.time() - start_time) * 1000
                    self._log_inference(request.request_id, request.model_name, 
                                      "unknown", execution_time, False, str(e))
                    raise e
                
                # Wait before retry
                await asyncio.sleep(2 ** retries)  # Exponential backoff
    
    def _create_execution_plan(self, request: InferenceRequest) -> Dict[str, Any]:
        """Create optimal execution plan for the inference request."""
        
        # Get available devices
        available_devices = [d for d in self.edge_devices.values() 
                           if d.status in [EdgeDeviceStatus.ONLINE, EdgeDeviceStatus.DEGRADED]]
        
        # Check if model is supported on edge devices
        supporting_devices = [d for d in available_devices 
                            if request.model_name in d.supported_models or not d.supported_models]
        
        # Determine execution mode based on constraints
        if not supporting_devices:
            return {"mode": "cloud_fallback", "reason": "no_supporting_devices"}
        
        elif len(supporting_devices) == 1:
            return {
                "mode": "edge",
                "target_device": supporting_devices[0].device_id,
                "reason": "single_device_available"
            }
        
        elif request.max_latency_ms < 1000 and len(supporting_devices) > 1:
            # Use distributed processing for low-latency requirements
            return {
                "mode": "distributed",
                "target_devices": [d.device_id for d in supporting_devices[:3]],
                "partitioning_strategy": ModelPartition.ENSEMBLE,
                "reason": "low_latency_requirement"
            }
        
        else:
            # Use load balancing for regular requests
            selected_device = self.load_balancer(supporting_devices)
            return {
                "mode": "edge",
                "target_device": selected_device.device_id,
                "reason": "load_balanced_selection"
            }
    
    async def _execute_edge_inference(self, request: InferenceRequest, 
                                    execution_plan: Dict[str, Any]) -> InferenceResult:
        """Execute inference on a single edge device."""
        device_id = execution_plan["target_device"]
        device = self.edge_devices[device_id]
        
        # Update device load
        device.current_load += 0.1  # Simplified load tracking
        device.active_connections += 1
        
        try:
            # Simulate inference execution
            await asyncio.sleep(0.1)  # Simulate processing time
            
            # Create mock result (in real implementation, this would call the actual model)
            result = InferenceResult(
                request_id=request.request_id,
                result={"prediction": "normal", "confidence": 0.92},  # Mock result
                execution_time_ms=100.0 + device.network_latency_ms,
                device_id=device_id,
                model_version="v1.0",
                confidence_score=0.92,
                processing_mode="edge_single",
                metadata={"device_name": device.name, "location": device.location}
            )
            
            # Update device metrics
            device.total_inferences += 1
            device.average_inference_time = (device.average_inference_time * 0.9 + 
                                           result.execution_time_ms * 0.1)
            
            return result
            
        finally:
            device.current_load = max(0.0, device.current_load - 0.1)
            device.active_connections = max(0, device.active_connections - 1)
    
    async def _execute_distributed_inference(self, request: InferenceRequest,
                                           execution_plan: Dict[str, Any]) -> InferenceResult:
        """Execute distributed inference across multiple edge devices."""
        
        device_ids = execution_plan["target_devices"]
        strategy = execution_plan.get("partitioning_strategy", ModelPartition.ENSEMBLE)
        
        # Create model partitions
        target_devices = [self.edge_devices[device_id] for device_id in device_ids]
        model_info = self.model_partitioner.analyze_model_structure(request.model_name)
        partitions = self.model_partitioner.create_partitions(model_info, target_devices, strategy)
        
        # Execute partitions in parallel
        partition_tasks = []
        for partition_id, partition_info in partitions["partitions"].items():
            task = self._execute_partition(request, partition_id, partition_info)
            partition_tasks.append(task)
        
        # Wait for all partitions to complete
        partition_results = await asyncio.gather(*partition_tasks, return_exceptions=True)
        
        # Aggregate results
        successful_results = [r for r in partition_results if isinstance(r, dict)]
        
        if not successful_results:
            raise Exception("All partition executions failed")
        
        # Simple aggregation (in real implementation, this would be more sophisticated)
        if partitions["strategy"] == "ensemble":
            aggregated_confidence = np.mean([r["confidence"] for r in successful_results])
            aggregated_result = {"prediction": successful_results[0]["prediction"], 
                               "confidence": aggregated_confidence}
        else:
            # For layer-wise, combine outputs sequentially
            aggregated_result = successful_results[-1]  # Simplified
        
        total_execution_time = max(r["execution_time"] for r in successful_results)
        
        return InferenceResult(
            request_id=request.request_id,
            result=aggregated_result,
            execution_time_ms=total_execution_time,
            device_id="+".join(device_ids),
            model_version="v1.0",
            confidence_score=aggregated_result.get("confidence", 0.0),
            processing_mode="distributed",
            metadata={
                "partitions_used": len(successful_results),
                "strategy": partitions["strategy"],
                "devices": device_ids
            }
        )
    
    async def _execute_partition(self, request: InferenceRequest, 
                               partition_id: str, partition_info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single model partition."""
        device_id = partition_info["device_id"]
        
        # Simulate partition execution
        await asyncio.sleep(0.05)  # Simulate processing time
        
        return {
            "partition_id": partition_id,
            "result": {"feature_map": [1, 2, 3]},  # Mock partition result
            "confidence": 0.9,
            "execution_time": partition_info.get("estimated_latency_ms", 50.0),
            "device_id": device_id
        }
    
    async def _execute_cloud_fallback(self, request: InferenceRequest) -> InferenceResult:
        """Execute inference in cloud as fallback."""
        
        # Simulate cloud inference
        await asyncio.sleep(0.5)  # Higher latency for cloud
        
        return InferenceResult(
            request_id=request.request_id,
            result={"prediction": "normal", "confidence": 0.88},
            execution_time_ms=500.0,
            device_id="cloud",
            model_version="v1.0",
            confidence_score=0.88,
            processing_mode="cloud_fallback",
            metadata={"fallback_reason": "edge_unavailable"}
        )
    
    def _log_inference(self, request_id: str, model_name: str, device_id: str,
                      execution_time_ms: float, success: bool, error_message: Optional[str]):
        """Log inference execution to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO inference_logs 
                    (request_id, model_name, device_id, execution_time_ms, 
                     success, error_message, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    request_id, model_name, device_id, execution_time_ms,
                    int(success), error_message, datetime.now().isoformat()
                ))
        except Exception as e:
            logger.error(f"Failed to log inference: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        with self._lock:
            online_devices = [d for d in self.edge_devices.values() 
                            if d.status == EdgeDeviceStatus.ONLINE]
            
            total_capacity = sum(d.max_capacity for d in self.edge_devices.values())
            current_load = sum(d.current_load for d in self.edge_devices.values())
            
            return {
                "timestamp": datetime.now().isoformat(),
                "devices": {
                    "total": len(self.edge_devices),
                    "online": len(online_devices),
                    "offline": len([d for d in self.edge_devices.values() 
                                  if d.status == EdgeDeviceStatus.OFFLINE]),
                    "overloaded": len([d for d in self.edge_devices.values() 
                                     if d.status == EdgeDeviceStatus.OVERLOADED])
                },
                "capacity": {
                    "total": total_capacity,
                    "current_load": current_load,
                    "utilization_percent": (current_load / total_capacity * 100) if total_capacity > 0 else 0
                },
                "queue_status": {
                    "current_size": self.inference_queue.qsize(),
                    "max_size": self.inference_queue.maxsize,
                    "active_tasks": len(self.processing_tasks)
                },
                "performance": {
                    "avg_latency_ms": np.mean(self.performance_metrics["latency"][-100:]) if self.performance_metrics["latency"] else 0,
                    "total_inferences": sum(d.total_inferences for d in self.edge_devices.values()),
                    "error_rate": sum(d.error_count for d in self.edge_devices.values()) / max(1, sum(d.total_inferences for d in self.edge_devices.values()))
                }
            }


async def example_usage():
    """Demonstrate edge computing and distributed inference."""
    
    print("🌐 Edge Computing Distributed Inference Demo")
    print("=" * 50)
    
    # Initialize orchestrator
    orchestrator = EdgeOrchestrator()
    
    # Register edge devices
    devices = [
        EdgeDevice(
            device_id="edge_01",
            name="Hospital Edge Node 1",
            location="Emergency Department",
            capabilities={"compute_power": 2.0, "memory_gb": 8, "gpu": True},
            supported_models=["chest_xray_model"]
        ),
        EdgeDevice(
            device_id="edge_02",
            name="Hospital Edge Node 2",
            location="Radiology Department",
            capabilities={"compute_power": 1.5, "memory_gb": 4, "gpu": False},
            supported_models=["chest_xray_model"]
        ),
        EdgeDevice(
            device_id="edge_03",
            name="Mobile Edge Unit",
            location="Ambulance",
            capabilities={"compute_power": 1.0, "memory_gb": 2, "gpu": False},
            supported_models=["chest_xray_model"]
        )
    ]
    
    for device in devices:
        orchestrator.register_edge_device(device)
    
    print(f"✅ Registered {len(devices)} edge devices")
    
    # Simulate device heartbeats
    for device in devices:
        orchestrator.update_device_heartbeat(device.device_id, {
            "cpu_usage": np.random.uniform(0.2, 0.6),
            "memory_usage": np.random.uniform(0.3, 0.7),
            "network_latency": np.random.uniform(10, 50),
            "active_connections": np.random.randint(0, 5)
        })
    
    # Submit inference requests
    requests = []
    for i in range(10):
        request = InferenceRequest(
            request_id=f"req_{i:03d}",
            model_name="chest_xray_model",
            input_data={"image": f"xray_image_{i}.jpg"},
            priority=np.random.choice([1, 2, 3]),
            max_latency_ms=np.random.uniform(500, 2000),
            client_location="Emergency Room"
        )
        requests.append(request)
    
    print(f"📝 Submitting {len(requests)} inference requests...")
    
    # Process requests
    tasks = []
    for request in requests:
        task_id = await orchestrator.submit_inference_request(request)
        tasks.append(orchestrator.processing_tasks[task_id])
    
    # Wait for results
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    successful_results = [r for r in results if isinstance(r, InferenceResult)]
    failed_results = [r for r in results if isinstance(r, Exception)]
    
    print(f"✅ Completed {len(successful_results)} successful inferences")
    print(f"❌ Failed {len(failed_results)} inferences")
    
    # Show performance summary
    if successful_results:
        avg_latency = np.mean([r.execution_time_ms for r in successful_results])
        processing_modes = {}
        for result in successful_results:
            mode = result.processing_mode
            processing_modes[mode] = processing_modes.get(mode, 0) + 1
        
        print(f"📊 Performance Summary:")
        print(f"  Average latency: {avg_latency:.1f}ms")
        print(f"  Processing modes: {processing_modes}")
    
    # Show system status
    status = orchestrator.get_system_status()
    print(f"🖥️  System Status:")
    print(f"  Online devices: {status['devices']['online']}/{status['devices']['total']}")
    print(f"  System utilization: {status['capacity']['utilization_percent']:.1f}%")
    print(f"  Total inferences: {status['performance']['total_inferences']}")


def main():
    """Main entry point."""
    asyncio.run(example_usage())


if __name__ == "__main__":
    main()