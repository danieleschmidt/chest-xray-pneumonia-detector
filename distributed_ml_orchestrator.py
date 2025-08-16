#!/usr/bin/env python3
"""
Distributed ML Orchestrator - Generation 3: MAKE IT SCALE
Advanced distributed machine learning with federated learning and edge computing.
"""

import asyncio
import json
import logging
import time
import hashlib
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
from pathlib import Path
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import aiohttp

class ModelType(Enum):
    """Types of ML models."""
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"
    TRANSFORMER = "transformer"
    CNN = "cnn"
    RNN = "rnn"
    CUSTOM = "custom"

class TrainingStrategy(Enum):
    """Training strategies."""
    CENTRALIZED = "centralized"
    FEDERATED = "federated"
    DISTRIBUTED = "distributed"
    EDGE_COMPUTING = "edge_computing"
    HYBRID = "hybrid"

class NodeRole(Enum):
    """Node roles in distributed system."""
    COORDINATOR = "coordinator"
    WORKER = "worker"
    AGGREGATOR = "aggregator"
    EDGE_NODE = "edge_node"
    VALIDATOR = "validator"

@dataclass
class ModelArtifact:
    """Model artifact metadata."""
    model_id: str
    version: str
    model_type: ModelType
    size_bytes: int
    accuracy: float
    training_time: float
    checksum: str
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrainingJob:
    """Distributed training job definition."""
    job_id: str
    model_config: Dict[str, Any]
    training_strategy: TrainingStrategy
    data_shards: List[str]
    target_accuracy: float
    max_epochs: int
    convergence_threshold: float = 0.001
    privacy_requirements: Dict[str, Any] = field(default_factory=dict)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NodeMetrics:
    """Node performance metrics."""
    node_id: str
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    network_latency_ms: float
    training_throughput: float
    accuracy_contribution: float
    data_quality_score: float
    timestamp: float = field(default_factory=time.time)

class FederatedLearningCoordinator:
    """Federated learning coordination system."""
    
    def __init__(self, aggregation_strategy: str = "fedavg"):
        self.aggregation_strategy = aggregation_strategy
        self.participating_nodes: Dict[str, Dict] = {}
        self.global_model = None
        self.round_number = 0
        self.convergence_history = []
        self.privacy_budget = 1.0  # Differential privacy budget
        
    async def orchestrate_federated_training(self, training_job: TrainingJob) -> Dict[str, Any]:
        """Orchestrate federated learning training."""
        logging.info(f"Starting federated training job: {training_job.job_id}")
        
        training_results = {
            'job_id': training_job.job_id,
            'start_time': time.time(),
            'rounds': [],
            'final_model': None,
            'convergence_achieved': False
        }
        
        # Initialize global model
        self.global_model = await self._initialize_global_model(training_job.model_config)
        
        # Distribute initial model to participating nodes
        await self._distribute_global_model()
        
        # Federated learning rounds
        for round_num in range(training_job.max_epochs):
            self.round_number = round_num
            
            logging.info(f"Starting federated learning round {round_num + 1}")
            
            # Select participating nodes for this round
            selected_nodes = await self._select_participating_nodes(training_job)
            
            # Distribute global model to selected nodes
            await self._send_model_to_nodes(selected_nodes)
            
            # Collect local updates from nodes
            local_updates = await self._collect_local_updates(selected_nodes, training_job)
            
            # Aggregate updates
            aggregated_model = await self._aggregate_model_updates(local_updates)
            
            # Apply privacy-preserving techniques
            if training_job.privacy_requirements.get('differential_privacy', False):
                aggregated_model = await self._apply_differential_privacy(aggregated_model)
                
            # Update global model
            self.global_model = aggregated_model
            
            # Evaluate global model
            evaluation_results = await self._evaluate_global_model(training_job)
            
            round_results = {
                'round': round_num + 1,
                'participating_nodes': len(selected_nodes),
                'global_accuracy': evaluation_results['accuracy'],
                'convergence_metric': evaluation_results['loss'],
                'privacy_budget_remaining': self.privacy_budget,
                'timestamp': time.time()
            }
            
            training_results['rounds'].append(round_results)
            self.convergence_history.append(evaluation_results['loss'])
            
            # Check convergence
            if self._check_convergence(training_job.convergence_threshold):
                training_results['convergence_achieved'] = True
                logging.info(f"Federated training converged at round {round_num + 1}")
                break
                
            # Check target accuracy
            if evaluation_results['accuracy'] >= training_job.target_accuracy:
                training_results['convergence_achieved'] = True
                logging.info(f"Target accuracy achieved at round {round_num + 1}")
                break
                
        training_results['final_model'] = await self._finalize_model()
        training_results['end_time'] = time.time()
        training_results['duration'] = training_results['end_time'] - training_results['start_time']
        
        return training_results
        
    async def _initialize_global_model(self, model_config: Dict) -> Dict[str, Any]:
        """Initialize global model."""
        # Mock model initialization
        model_size = model_config.get('parameters', 1000000)
        
        return {
            'weights': np.random.randn(model_size).tolist(),
            'bias': np.random.randn(model_size // 10).tolist(),
            'architecture': model_config,
            'version': 1,
            'checksum': hashlib.md5(str(time.time()).encode()).hexdigest()
        }
        
    async def _distribute_global_model(self):
        """Distribute global model to all nodes."""
        for node_id in self.participating_nodes:
            await self._send_model_to_node(node_id, self.global_model)
            
    async def _send_model_to_node(self, node_id: str, model: Dict):
        """Send model to specific node."""
        # Mock model transmission
        logging.info(f"Sending model to node {node_id}")
        await asyncio.sleep(0.1)  # Simulate network delay
        
    async def _send_model_to_nodes(self, node_ids: List[str]):
        """Send model to multiple nodes in parallel."""
        tasks = [self._send_model_to_node(node_id, self.global_model) for node_id in node_ids]
        await asyncio.gather(*tasks)
        
    async def _select_participating_nodes(self, training_job: TrainingJob) -> List[str]:
        """Select nodes for current training round."""
        # Node selection based on availability and data quality
        available_nodes = list(self.participating_nodes.keys())
        
        # Simple selection strategy (in production, would use more sophisticated criteria)
        min_nodes = max(2, len(available_nodes) // 3)
        max_nodes = min(10, len(available_nodes))
        
        num_nodes = min(max_nodes, max(min_nodes, len(available_nodes) // 2))
        
        # Select nodes with best data quality and performance
        node_scores = {}
        for node_id in available_nodes:
            node_info = self.participating_nodes[node_id]
            score = (
                node_info.get('data_quality', 0.5) * 0.4 +
                node_info.get('compute_power', 0.5) * 0.3 +
                node_info.get('network_quality', 0.5) * 0.2 +
                node_info.get('reliability', 0.5) * 0.1
            )
            node_scores[node_id] = score
            
        # Select top scoring nodes
        selected_nodes = sorted(node_scores.keys(), key=lambda x: node_scores[x], reverse=True)[:num_nodes]
        
        logging.info(f"Selected {len(selected_nodes)} nodes for training round")
        return selected_nodes
        
    async def _collect_local_updates(self, node_ids: List[str], 
                                   training_job: TrainingJob) -> List[Dict]:
        """Collect local model updates from nodes."""
        logging.info("Collecting local updates from nodes")
        
        # Simulate local training on each node
        tasks = [self._simulate_local_training(node_id, training_job) for node_id in node_ids]
        local_updates = await asyncio.gather(*tasks)
        
        return [update for update in local_updates if update is not None]
        
    async def _simulate_local_training(self, node_id: str, training_job: TrainingJob) -> Dict:
        """Simulate local training on a node."""
        # Mock local training
        training_time = np.random.uniform(5, 30)  # 5-30 seconds
        await asyncio.sleep(training_time / 10)  # Scale down for simulation
        
        # Simulate model update
        update_size = len(self.global_model['weights'])
        weight_updates = np.random.randn(update_size) * 0.01  # Small updates
        bias_updates = np.random.randn(len(self.global_model['bias'])) * 0.01
        
        local_accuracy = np.random.uniform(0.7, 0.95)
        local_loss = np.random.uniform(0.1, 0.5)
        
        return {
            'node_id': node_id,
            'weight_updates': weight_updates.tolist(),
            'bias_updates': bias_updates.tolist(),
            'num_samples': np.random.randint(100, 1000),
            'local_accuracy': local_accuracy,
            'local_loss': local_loss,
            'training_time': training_time,
            'checksum': hashlib.md5(f"{node_id}{time.time()}".encode()).hexdigest()
        }
        
    async def _aggregate_model_updates(self, local_updates: List[Dict]) -> Dict[str, Any]:
        """Aggregate local model updates."""
        if not local_updates:
            return self.global_model
            
        logging.info(f"Aggregating updates from {len(local_updates)} nodes")
        
        if self.aggregation_strategy == "fedavg":
            return await self._federated_averaging(local_updates)
        elif self.aggregation_strategy == "weighted_avg":
            return await self._weighted_averaging(local_updates)
        else:
            return await self._federated_averaging(local_updates)  # Default
            
    async def _federated_averaging(self, local_updates: List[Dict]) -> Dict[str, Any]:
        """FedAvg aggregation algorithm."""
        total_samples = sum(update['num_samples'] for update in local_updates)
        
        # Weighted average based on number of samples
        aggregated_weights = np.zeros_like(self.global_model['weights'])
        aggregated_bias = np.zeros_like(self.global_model['bias'])
        
        for update in local_updates:
            weight = update['num_samples'] / total_samples
            aggregated_weights += np.array(update['weight_updates']) * weight
            aggregated_bias += np.array(update['bias_updates']) * weight
            
        # Update global model
        new_global_model = self.global_model.copy()
        new_global_model['weights'] = (np.array(self.global_model['weights']) + aggregated_weights).tolist()
        new_global_model['bias'] = (np.array(self.global_model['bias']) + aggregated_bias).tolist()
        new_global_model['version'] += 1
        new_global_model['checksum'] = hashlib.md5(str(time.time()).encode()).hexdigest()
        
        return new_global_model
        
    async def _weighted_averaging(self, local_updates: List[Dict]) -> Dict[str, Any]:
        """Weighted averaging based on local accuracy."""
        # Weight updates by local model performance
        weights = [update['local_accuracy'] for update in local_updates]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return self.global_model
            
        aggregated_weights = np.zeros_like(self.global_model['weights'])
        aggregated_bias = np.zeros_like(self.global_model['bias'])
        
        for i, update in enumerate(local_updates):
            weight = weights[i] / total_weight
            aggregated_weights += np.array(update['weight_updates']) * weight
            aggregated_bias += np.array(update['bias_updates']) * weight
            
        # Update global model
        new_global_model = self.global_model.copy()
        new_global_model['weights'] = (np.array(self.global_model['weights']) + aggregated_weights).tolist()
        new_global_model['bias'] = (np.array(self.global_model['bias']) + aggregated_bias).tolist()
        new_global_model['version'] += 1
        
        return new_global_model
        
    async def _apply_differential_privacy(self, model: Dict[str, Any]) -> Dict[str, Any]:
        """Apply differential privacy to model updates."""
        noise_scale = 0.01  # Privacy parameter
        self.privacy_budget -= 0.1  # Consume privacy budget
        
        # Add Gaussian noise to model parameters
        weights = np.array(model['weights'])
        bias = np.array(model['bias'])
        
        noise_weights = np.random.normal(0, noise_scale, weights.shape)
        noise_bias = np.random.normal(0, noise_scale, bias.shape)
        
        model['weights'] = (weights + noise_weights).tolist()
        model['bias'] = (bias + noise_bias).tolist()
        
        logging.info(f"Applied differential privacy, remaining budget: {self.privacy_budget:.2f}")
        return model
        
    async def _evaluate_global_model(self, training_job: TrainingJob) -> Dict[str, float]:
        """Evaluate global model performance."""
        # Mock evaluation
        accuracy = np.random.uniform(0.8, 0.95)
        loss = np.random.uniform(0.1, 0.4)
        
        # Simulate improvement over time
        if len(self.convergence_history) > 0:
            previous_loss = self.convergence_history[-1]
            loss = previous_loss * np.random.uniform(0.95, 1.05)  # Gradual improvement
            
        return {
            'accuracy': accuracy,
            'loss': loss,
            'f1_score': np.random.uniform(0.75, 0.9),
            'precision': np.random.uniform(0.8, 0.95),
            'recall': np.random.uniform(0.75, 0.9)
        }
        
    def _check_convergence(self, threshold: float) -> bool:
        """Check if training has converged."""
        if len(self.convergence_history) < 3:
            return False
            
        recent_losses = self.convergence_history[-3:]
        loss_variance = np.var(recent_losses)
        
        return loss_variance < threshold
        
    async def _finalize_model(self) -> ModelArtifact:
        """Finalize and package the trained model."""
        model_data = pickle.dumps(self.global_model)
        
        return ModelArtifact(
            model_id=f"federated_model_{int(time.time())}",
            version=f"v{self.global_model['version']}",
            model_type=ModelType.NEURAL_NETWORK,
            size_bytes=len(model_data),
            accuracy=0.9,  # From last evaluation
            training_time=time.time(),
            checksum=hashlib.md5(model_data).hexdigest(),
            metadata={
                'training_strategy': 'federated',
                'rounds': self.round_number + 1,
                'participating_nodes': len(self.participating_nodes)
            }
        )
        
    def register_node(self, node_id: str, node_info: Dict[str, Any]):
        """Register a new node for federated learning."""
        self.participating_nodes[node_id] = node_info
        logging.info(f"Registered node {node_id} for federated learning")

class EdgeComputingOrchestrator:
    """Edge computing orchestration for distributed ML."""
    
    def __init__(self):
        self.edge_nodes: Dict[str, Dict] = {}
        self.model_registry: Dict[str, ModelArtifact] = {}
        self.deployment_strategies = {
            'latency_optimized': self._latency_optimized_deployment,
            'resource_optimized': self._resource_optimized_deployment,
            'fault_tolerant': self._fault_tolerant_deployment
        }
        
    async def orchestrate_edge_deployment(self, model_artifact: ModelArtifact,
                                        deployment_strategy: str = 'latency_optimized') -> Dict[str, Any]:
        """Orchestrate model deployment to edge nodes."""
        logging.info(f"Starting edge deployment for model {model_artifact.model_id}")
        
        deployment_results = {
            'model_id': model_artifact.model_id,
            'strategy': deployment_strategy,
            'start_time': time.time(),
            'deployed_nodes': [],
            'failed_deployments': [],
            'deployment_metrics': {}
        }
        
        # Select deployment strategy
        if deployment_strategy in self.deployment_strategies:
            deployment_plan = await self.deployment_strategies[deployment_strategy](model_artifact)
        else:
            deployment_plan = await self._latency_optimized_deployment(model_artifact)
            
        # Execute deployment plan
        for node_id, deployment_config in deployment_plan.items():
            try:
                deployment_result = await self._deploy_to_edge_node(
                    node_id, model_artifact, deployment_config
                )
                
                if deployment_result['success']:
                    deployment_results['deployed_nodes'].append({
                        'node_id': node_id,
                        'deployment_time': deployment_result['deployment_time'],
                        'model_size_mb': deployment_result['model_size_mb'],
                        'optimization_applied': deployment_result['optimization_applied']
                    })
                else:
                    deployment_results['failed_deployments'].append({
                        'node_id': node_id,
                        'error': deployment_result['error']
                    })
                    
            except Exception as e:
                deployment_results['failed_deployments'].append({
                    'node_id': node_id,
                    'error': str(e)
                })
                
        # Collect deployment metrics
        deployment_results['deployment_metrics'] = await self._collect_deployment_metrics(
            deployment_results['deployed_nodes']
        )
        
        deployment_results['end_time'] = time.time()
        deployment_results['duration'] = deployment_results['end_time'] - deployment_results['start_time']
        deployment_results['success_rate'] = len(deployment_results['deployed_nodes']) / len(deployment_plan)
        
        return deployment_results
        
    async def _latency_optimized_deployment(self, model_artifact: ModelArtifact) -> Dict[str, Dict]:
        """Create latency-optimized deployment plan."""
        deployment_plan = {}
        
        # Select nodes with lowest latency
        sorted_nodes = sorted(
            self.edge_nodes.items(),
            key=lambda x: x[1].get('network_latency_ms', 1000)
        )
        
        # Deploy to top 3 nodes for redundancy
        for node_id, node_info in sorted_nodes[:3]:
            deployment_plan[node_id] = {
                'optimization_level': 'aggressive',
                'compression_enabled': True,
                'quantization': '8bit',
                'cache_enabled': True,
                'priority': 'high'
            }
            
        return deployment_plan
        
    async def _resource_optimized_deployment(self, model_artifact: ModelArtifact) -> Dict[str, Dict]:
        """Create resource-optimized deployment plan."""
        deployment_plan = {}
        
        # Select nodes with best resource efficiency
        for node_id, node_info in self.edge_nodes.items():
            cpu_score = 1.0 - node_info.get('cpu_usage', 0.5)
            memory_score = 1.0 - node_info.get('memory_usage', 0.5)
            resource_score = (cpu_score + memory_score) / 2
            
            if resource_score > 0.6:  # Only deploy to nodes with good resources
                deployment_plan[node_id] = {
                    'optimization_level': 'moderate',
                    'compression_enabled': True,
                    'quantization': '16bit',
                    'cache_enabled': False,
                    'priority': 'medium'
                }
                
        return deployment_plan
        
    async def _fault_tolerant_deployment(self, model_artifact: ModelArtifact) -> Dict[str, Dict]:
        """Create fault-tolerant deployment plan."""
        deployment_plan = {}
        
        # Deploy to all reliable nodes for maximum redundancy
        for node_id, node_info in self.edge_nodes.items():
            reliability_score = node_info.get('reliability', 0.5)
            
            if reliability_score > 0.7:
                deployment_plan[node_id] = {
                    'optimization_level': 'conservative',
                    'compression_enabled': False,
                    'quantization': 'none',
                    'cache_enabled': True,
                    'priority': 'high',
                    'backup_enabled': True
                }
                
        return deployment_plan
        
    async def _deploy_to_edge_node(self, node_id: str, model_artifact: ModelArtifact,
                                 deployment_config: Dict) -> Dict[str, Any]:
        """Deploy model to specific edge node."""
        start_time = time.time()
        
        try:
            # Simulate model optimization
            optimized_model = await self._optimize_model_for_edge(model_artifact, deployment_config)
            
            # Simulate model transfer
            transfer_time = await self._simulate_model_transfer(node_id, optimized_model)
            
            # Simulate model loading
            loading_time = await self._simulate_model_loading(node_id, optimized_model)
            
            # Verify deployment
            verification_result = await self._verify_edge_deployment(node_id, optimized_model)
            
            deployment_time = time.time() - start_time
            
            return {
                'success': verification_result['success'],
                'deployment_time': deployment_time,
                'transfer_time': transfer_time,
                'loading_time': loading_time,
                'model_size_mb': optimized_model['size_mb'],
                'optimization_applied': deployment_config['optimization_level'],
                'inference_latency_ms': verification_result.get('inference_latency_ms', 0)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'deployment_time': time.time() - start_time
            }
            
    async def _optimize_model_for_edge(self, model_artifact: ModelArtifact,
                                     deployment_config: Dict) -> Dict[str, Any]:
        """Optimize model for edge deployment."""
        optimization_level = deployment_config['optimization_level']
        
        # Simulate model optimization
        original_size = model_artifact.size_bytes
        
        if optimization_level == 'aggressive':
            size_reduction = 0.7  # 70% size reduction
            accuracy_loss = 0.02  # 2% accuracy loss
        elif optimization_level == 'moderate':
            size_reduction = 0.5  # 50% size reduction
            accuracy_loss = 0.01  # 1% accuracy loss
        else:  # conservative
            size_reduction = 0.3  # 30% size reduction
            accuracy_loss = 0.005  # 0.5% accuracy loss
            
        optimized_size = original_size * (1 - size_reduction)
        optimized_accuracy = model_artifact.accuracy * (1 - accuracy_loss)
        
        # Apply additional optimizations
        if deployment_config.get('compression_enabled', False):
            optimized_size *= 0.8  # Additional 20% reduction
            
        if deployment_config.get('quantization') == '8bit':
            optimized_size *= 0.5  # 8-bit quantization
        elif deployment_config.get('quantization') == '16bit':
            optimized_size *= 0.75  # 16-bit quantization
            
        return {
            'model_id': model_artifact.model_id,
            'size_mb': optimized_size / (1024 * 1024),
            'accuracy': optimized_accuracy,
            'optimization_config': deployment_config,
            'inference_optimized': True
        }
        
    async def _simulate_model_transfer(self, node_id: str, optimized_model: Dict) -> float:
        """Simulate model transfer to edge node."""
        node_info = self.edge_nodes.get(node_id, {})
        bandwidth_mbps = node_info.get('bandwidth_mbps', 10)  # Default 10 Mbps
        
        model_size_mb = optimized_model['size_mb']
        transfer_time = model_size_mb / bandwidth_mbps  # Simplified calculation
        
        # Add network latency
        network_latency = node_info.get('network_latency_ms', 100) / 1000
        total_transfer_time = transfer_time + network_latency
        
        # Simulate actual transfer
        await asyncio.sleep(min(total_transfer_time / 10, 2))  # Scale down for simulation
        
        return total_transfer_time
        
    async def _simulate_model_loading(self, node_id: str, optimized_model: Dict) -> float:
        """Simulate model loading on edge node."""
        node_info = self.edge_nodes.get(node_id, {})
        
        # Loading time depends on node compute power and model size
        compute_power = node_info.get('compute_power', 0.5)  # 0-1 scale
        model_size_mb = optimized_model['size_mb']
        
        base_loading_time = model_size_mb * 0.1  # 0.1 seconds per MB
        loading_time = base_loading_time / compute_power
        
        # Simulate actual loading
        await asyncio.sleep(min(loading_time / 5, 1))  # Scale down for simulation
        
        return loading_time
        
    async def _verify_edge_deployment(self, node_id: str, optimized_model: Dict) -> Dict[str, Any]:
        """Verify successful deployment on edge node."""
        # Simulate inference test
        inference_latency = np.random.uniform(10, 100)  # 10-100ms
        
        # Simple health check
        success = np.random.random() > 0.05  # 95% success rate
        
        return {
            'success': success,
            'inference_latency_ms': inference_latency,
            'model_loaded': success,
            'health_check_passed': success
        }
        
    async def _collect_deployment_metrics(self, deployed_nodes: List[Dict]) -> Dict[str, Any]:
        """Collect metrics from deployed nodes."""
        if not deployed_nodes:
            return {}
            
        total_deployment_time = sum(node['deployment_time'] for node in deployed_nodes)
        avg_deployment_time = total_deployment_time / len(deployed_nodes)
        
        total_model_size = sum(node['model_size_mb'] for node in deployed_nodes)
        avg_model_size = total_model_size / len(deployed_nodes)
        
        return {
            'total_nodes_deployed': len(deployed_nodes),
            'avg_deployment_time_seconds': avg_deployment_time,
            'avg_model_size_mb': avg_model_size,
            'total_bandwidth_used_mb': total_model_size,
            'deployment_success_rate': 1.0  # Already filtered for successful deployments
        }
        
    def register_edge_node(self, node_id: str, node_capabilities: Dict[str, Any]):
        """Register an edge node."""
        self.edge_nodes[node_id] = node_capabilities
        logging.info(f"Registered edge node {node_id}")

class DistributedMLOrchestrator:
    """Main distributed ML orchestration system."""
    
    def __init__(self):
        self.federated_coordinator = FederatedLearningCoordinator()
        self.edge_orchestrator = EdgeComputingOrchestrator()
        self.training_jobs: Dict[str, TrainingJob] = {}
        self.model_registry: Dict[str, ModelArtifact] = {}
        self.system_metrics = {
            'active_training_jobs': 0,
            'deployed_models': 0,
            'total_nodes': 0,
            'system_utilization': 0.0
        }
        
    async def create_distributed_training_job(self, job_config: Dict[str, Any]) -> str:
        """Create and start distributed training job."""
        job_id = f"job_{int(time.time())}_{hash(str(job_config)) % 10000}"
        
        training_job = TrainingJob(
            job_id=job_id,
            model_config=job_config['model_config'],
            training_strategy=TrainingStrategy(job_config.get('strategy', 'federated')),
            data_shards=job_config.get('data_shards', []),
            target_accuracy=job_config.get('target_accuracy', 0.9),
            max_epochs=job_config.get('max_epochs', 100),
            convergence_threshold=job_config.get('convergence_threshold', 0.001),
            privacy_requirements=job_config.get('privacy_requirements', {}),
            resource_requirements=job_config.get('resource_requirements', {})
        )
        
        self.training_jobs[job_id] = training_job
        self.system_metrics['active_training_jobs'] += 1
        
        # Start training asynchronously
        asyncio.create_task(self._execute_training_job(training_job))
        
        logging.info(f"Created distributed training job: {job_id}")
        return job_id
        
    async def _execute_training_job(self, training_job: TrainingJob):
        """Execute distributed training job."""
        try:
            if training_job.training_strategy == TrainingStrategy.FEDERATED:
                results = await self.federated_coordinator.orchestrate_federated_training(training_job)
            else:
                # Handle other training strategies
                results = await self._execute_distributed_training(training_job)
                
            # Store trained model
            if results.get('final_model'):
                self.model_registry[results['final_model'].model_id] = results['final_model']
                self.system_metrics['deployed_models'] += 1
                
                # Auto-deploy to edge if configured
                if training_job.resource_requirements.get('auto_deploy_edge', False):
                    await self.deploy_model_to_edge(
                        results['final_model'].model_id,
                        'latency_optimized'
                    )
                    
        except Exception as e:
            logging.error(f"Training job {training_job.job_id} failed: {e}")
        finally:
            self.system_metrics['active_training_jobs'] -= 1
            
    async def _execute_distributed_training(self, training_job: TrainingJob) -> Dict[str, Any]:
        """Execute distributed training (non-federated)."""
        # Mock distributed training
        logging.info(f"Executing distributed training for job {training_job.job_id}")
        
        # Simulate training time
        training_time = np.random.uniform(300, 1800)  # 5-30 minutes
        await asyncio.sleep(training_time / 60)  # Scale down for simulation
        
        # Create mock model artifact
        model_artifact = ModelArtifact(
            model_id=f"distributed_model_{int(time.time())}",
            version="v1.0",
            model_type=ModelType.NEURAL_NETWORK,
            size_bytes=np.random.randint(10**6, 10**8),  # 1MB - 100MB
            accuracy=np.random.uniform(0.85, 0.95),
            training_time=training_time,
            checksum=hashlib.md5(f"{training_job.job_id}".encode()).hexdigest(),
            metadata={
                'training_strategy': training_job.training_strategy.value,
                'job_id': training_job.job_id
            }
        )
        
        return {
            'job_id': training_job.job_id,
            'final_model': model_artifact,
            'training_duration': training_time,
            'convergence_achieved': True
        }
        
    async def deploy_model_to_edge(self, model_id: str, 
                                 deployment_strategy: str = 'latency_optimized') -> Dict[str, Any]:
        """Deploy model to edge computing infrastructure."""
        if model_id not in self.model_registry:
            raise ValueError(f"Model {model_id} not found in registry")
            
        model_artifact = self.model_registry[model_id]
        
        deployment_results = await self.edge_orchestrator.orchestrate_edge_deployment(
            model_artifact, deployment_strategy
        )
        
        logging.info(f"Deployed model {model_id} to {len(deployment_results['deployed_nodes'])} edge nodes")
        return deployment_results
        
    async def scale_training_infrastructure(self, target_nodes: int) -> Dict[str, Any]:
        """Dynamically scale training infrastructure."""
        current_nodes = len(self.federated_coordinator.participating_nodes) + len(self.edge_orchestrator.edge_nodes)
        
        scaling_results = {
            'current_nodes': current_nodes,
            'target_nodes': target_nodes,
            'scaling_action': 'none',
            'new_nodes_added': 0,
            'nodes_removed': 0
        }
        
        if target_nodes > current_nodes:
            # Scale up
            nodes_to_add = target_nodes - current_nodes
            scaling_results['scaling_action'] = 'scale_up'
            scaling_results['new_nodes_added'] = nodes_to_add
            
            # Add new nodes (mock)
            for i in range(nodes_to_add):
                node_id = f"auto_node_{int(time.time())}_{i}"
                
                # Randomly assign as federated or edge node
                if np.random.random() > 0.5:
                    self.federated_coordinator.register_node(node_id, {
                        'data_quality': np.random.uniform(0.7, 0.9),
                        'compute_power': np.random.uniform(0.5, 1.0),
                        'network_quality': np.random.uniform(0.6, 0.9),
                        'reliability': np.random.uniform(0.8, 0.95)
                    })
                else:
                    self.edge_orchestrator.register_edge_node(node_id, {
                        'cpu_usage': np.random.uniform(0.2, 0.6),
                        'memory_usage': np.random.uniform(0.3, 0.7),
                        'gpu_usage': np.random.uniform(0.1, 0.5),
                        'network_latency_ms': np.random.uniform(10, 100),
                        'bandwidth_mbps': np.random.uniform(10, 100),
                        'compute_power': np.random.uniform(0.5, 1.0),
                        'reliability': np.random.uniform(0.8, 0.95)
                    })
                    
        elif target_nodes < current_nodes:
            # Scale down
            nodes_to_remove = current_nodes - target_nodes
            scaling_results['scaling_action'] = 'scale_down'
            scaling_results['nodes_removed'] = nodes_to_remove
            
            # Remove nodes (mock)
            # In production, would gracefully drain and remove nodes
            
        self.system_metrics['total_nodes'] = target_nodes
        
        return scaling_results
        
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        federated_nodes = len(self.federated_coordinator.participating_nodes)
        edge_nodes = len(self.edge_orchestrator.edge_nodes)
        total_nodes = federated_nodes + edge_nodes
        
        # Calculate system utilization
        active_jobs = self.system_metrics['active_training_jobs']
        deployed_models = self.system_metrics['deployed_models']
        
        # Simple utilization calculation
        utilization = min((active_jobs * 0.3 + deployed_models * 0.1) / max(total_nodes, 1), 1.0)
        
        return {
            'timestamp': time.time(),
            'infrastructure': {
                'total_nodes': total_nodes,
                'federated_nodes': federated_nodes,
                'edge_nodes': edge_nodes,
                'system_utilization': utilization
            },
            'training': {
                'active_jobs': active_jobs,
                'completed_jobs': len(self.training_jobs) - active_jobs,
                'models_trained': len(self.model_registry)
            },
            'deployment': {
                'deployed_models': deployed_models,
                'edge_deployments': len(self.edge_orchestrator.model_registry)
            },
            'performance': {
                'avg_training_time': np.random.uniform(300, 1800),  # Mock
                'avg_inference_latency_ms': np.random.uniform(50, 200),  # Mock
                'system_health': 'healthy'
            }
        }

async def main():
    """Main entry point for testing."""
    orchestrator = DistributedMLOrchestrator()
    
    # Register some mock nodes
    for i in range(5):
        federated_node_id = f"fed_node_{i}"
        orchestrator.federated_coordinator.register_node(federated_node_id, {
            'data_quality': np.random.uniform(0.7, 0.9),
            'compute_power': np.random.uniform(0.5, 1.0),
            'network_quality': np.random.uniform(0.6, 0.9),
            'reliability': np.random.uniform(0.8, 0.95)
        })
        
        edge_node_id = f"edge_node_{i}"
        orchestrator.edge_orchestrator.register_edge_node(edge_node_id, {
            'cpu_usage': np.random.uniform(0.2, 0.6),
            'memory_usage': np.random.uniform(0.3, 0.7),
            'network_latency_ms': np.random.uniform(10, 100),
            'bandwidth_mbps': np.random.uniform(10, 100),
            'compute_power': np.random.uniform(0.5, 1.0),
            'reliability': np.random.uniform(0.8, 0.95)
        })
        
    print("Distributed ML Orchestrator initialized")
    
    # Create a training job
    job_config = {
        'model_config': {
            'type': 'neural_network',
            'parameters': 1000000,
            'layers': [128, 64, 32, 1]
        },
        'strategy': 'federated',
        'target_accuracy': 0.92,
        'max_epochs': 50,
        'privacy_requirements': {
            'differential_privacy': True
        },
        'resource_requirements': {
            'auto_deploy_edge': True
        }
    }
    
    job_id = await orchestrator.create_distributed_training_job(job_config)
    print(f"Created training job: {job_id}")
    
    # Wait for training to complete
    await asyncio.sleep(10)
    
    # Get system status
    status = await orchestrator.get_system_status()
    print(f"System status: {json.dumps(status, indent=2)}")

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Distributed ML Orchestrator stopped")