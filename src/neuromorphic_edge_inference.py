"""Neuromorphic Edge Inference Engine for Ultra-Low Power Medical AI.

This module implements neuromorphic computing principles for edge-based
medical AI inference, optimizing for ultra-low power consumption while
maintaining high accuracy for critical medical applications.
"""

import logging
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
import json
import numpy as np
from collections import deque
import asyncio
import heapq


class SpikeType(Enum):
    """Types of neural spikes in neuromorphic computing."""
    EXCITATORY = "excitatory"
    INHIBITORY = "inhibitory"
    MODULATORY = "modulatory"


class InferenceMode(Enum):
    """Inference operation modes for different power/accuracy tradeoffs."""
    ULTRA_LOW_POWER = "ultra_low_power"      # <10mW, basic accuracy
    LOW_POWER = "low_power"                  # <50mW, good accuracy
    BALANCED = "balanced"                    # <200mW, high accuracy
    HIGH_ACCURACY = "high_accuracy"          # <500mW, maximum accuracy
    EMERGENCY = "emergency"                  # No power limit, instant response


@dataclass
class Spike:
    """Represents a neural spike in the neuromorphic system."""
    neuron_id: int
    timestamp: float
    spike_type: SpikeType = SpikeType.EXCITATORY
    amplitude: float = 1.0
    duration: float = 1.0  # milliseconds
    
    def __lt__(self, other):
        """For priority queue ordering."""
        return self.timestamp < other.timestamp


@dataclass
class NeuromorphicNeuron:
    """Neuromorphic neuron with adaptive behavior."""
    neuron_id: int
    threshold: float = 1.0
    potential: float = 0.0
    leak_rate: float = 0.1
    refractory_period: float = 2.0  # milliseconds
    last_spike_time: float = 0.0
    adaptation_rate: float = 0.01
    synaptic_weights: Dict[int, float] = field(default_factory=dict)
    spike_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def update_potential(self, input_spikes: List[Spike], current_time: float) -> bool:
        """Update neuron potential and check for spike generation."""
        
        # Apply membrane leak
        time_delta = current_time - self.last_spike_time
        self.potential *= np.exp(-time_delta * self.leak_rate)
        
        # Check refractory period
        if current_time - self.last_spike_time < self.refractory_period:
            return False
            
        # Process input spikes
        for spike in input_spikes:
            if spike.neuron_id in self.synaptic_weights:
                weight = self.synaptic_weights[spike.neuron_id]
                
                if spike.spike_type == SpikeType.EXCITATORY:
                    self.potential += weight * spike.amplitude
                elif spike.spike_type == SpikeType.INHIBITORY:
                    self.potential -= weight * spike.amplitude
                elif spike.spike_type == SpikeType.MODULATORY:
                    self.potential *= (1 + weight * spike.amplitude * 0.1)
                    
        # Check for spike generation
        if self.potential >= self.threshold:
            self.last_spike_time = current_time
            self.spike_history.append(current_time)
            self.potential = 0.0  # Reset after spike
            
            # Adaptive threshold adjustment
            self._adapt_threshold()
            
            return True
            
        return False
        
    def _adapt_threshold(self) -> None:
        """Adapt threshold based on recent activity."""
        if len(self.spike_history) >= 2:
            recent_interval = self.spike_history[-1] - self.spike_history[-2]
            
            # Increase threshold if spiking too frequently
            if recent_interval < 5.0:  # Less than 5ms
                self.threshold += self.adaptation_rate
            # Decrease threshold if spiking infrequently
            elif recent_interval > 50.0:  # More than 50ms
                self.threshold -= self.adaptation_rate
                
            # Keep threshold in reasonable bounds
            self.threshold = max(0.1, min(10.0, self.threshold))


class SpikingNeuralNetwork:
    """Spiking neural network for neuromorphic inference."""
    
    def __init__(self, architecture: Dict[str, Any]):
        self.neurons: Dict[int, NeuromorphicNeuron] = {}
        self.spike_queue = []  # Priority queue for temporal spike processing
        self.current_time = 0.0
        self.time_step = 0.1  # milliseconds
        
        # Network architecture
        self.input_size = architecture.get("input_size", 784)
        self.hidden_layers = architecture.get("hidden_layers", [256, 128])
        self.output_size = architecture.get("output_size", 2)
        
        self._build_network()
        self._initialize_weights()
        
    def _build_network(self) -> None:
        """Build the spiking neural network structure."""
        neuron_id = 0
        
        # Input layer
        self.input_neurons = list(range(neuron_id, neuron_id + self.input_size))
        for i in self.input_neurons:
            self.neurons[i] = NeuromorphicNeuron(neuron_id=i, threshold=0.5)
        neuron_id += self.input_size
        
        # Hidden layers
        self.hidden_neurons = []
        for layer_size in self.hidden_layers:
            layer_neurons = list(range(neuron_id, neuron_id + layer_size))
            self.hidden_neurons.append(layer_neurons)
            
            for i in layer_neurons:
                self.neurons[i] = NeuromorphicNeuron(neuron_id=i, threshold=1.0)
            neuron_id += layer_size
            
        # Output layer
        self.output_neurons = list(range(neuron_id, neuron_id + self.output_size))
        for i in self.output_neurons:
            self.neurons[i] = NeuromorphicNeuron(neuron_id=i, threshold=1.5)
            
    def _initialize_weights(self) -> None:
        """Initialize synaptic weights between layers."""
        
        # Connect input to first hidden layer
        if self.hidden_neurons:
            first_hidden = self.hidden_neurons[0]
            for input_neuron in self.input_neurons:
                for hidden_neuron in first_hidden:
                    weight = np.random.normal(0, 0.1)
                    self.neurons[hidden_neuron].synaptic_weights[input_neuron] = weight
                    
            # Connect hidden layers
            for i in range(len(self.hidden_neurons) - 1):
                current_layer = self.hidden_neurons[i]
                next_layer = self.hidden_neurons[i + 1]
                
                for current_neuron in current_layer:
                    for next_neuron in next_layer:
                        weight = np.random.normal(0, 0.1)
                        self.neurons[next_neuron].synaptic_weights[current_neuron] = weight
                        
            # Connect last hidden layer to output
            last_hidden = self.hidden_neurons[-1]
            for hidden_neuron in last_hidden:
                for output_neuron in self.output_neurons:
                    weight = np.random.normal(0, 0.1)
                    self.neurons[output_neuron].synaptic_weights[hidden_neuron] = weight
        else:
            # Direct input to output connection
            for input_neuron in self.input_neurons:
                for output_neuron in self.output_neurons:
                    weight = np.random.normal(0, 0.1)
                    self.neurons[output_neuron].synaptic_weights[input_neuron] = weight
                    
    def encode_input(self, data: np.ndarray) -> List[Spike]:
        """Encode input data into spike trains."""
        spikes = []
        
        # Rate coding: higher values -> higher spike rates
        normalized_data = (data - data.min()) / (data.max() - data.min() + 1e-10)
        
        for i, value in enumerate(normalized_data.flatten()[:self.input_size]):
            # Generate spikes based on input intensity
            spike_rate = value * 100  # Max 100 Hz
            num_spikes = int(spike_rate * 0.01)  # For 10ms window
            
            for spike_idx in range(num_spikes):
                spike_time = self.current_time + np.random.uniform(0, 10)  # Within 10ms
                spike = Spike(
                    neuron_id=self.input_neurons[i],
                    timestamp=spike_time,
                    spike_type=SpikeType.EXCITATORY,
                    amplitude=1.0
                )
                spikes.append(spike)
                
        return spikes
        
    def forward_pass(self, input_spikes: List[Spike], simulation_time: float = 50.0) -> Dict[int, List[float]]:
        """Perform forward pass through the spiking network."""
        
        # Add input spikes to queue
        for spike in input_spikes:
            heapq.heappush(self.spike_queue, spike)
            
        output_spikes = {neuron_id: [] for neuron_id in self.output_neurons}
        end_time = self.current_time + simulation_time
        
        while self.current_time < end_time:
            # Process spikes at current time
            current_spikes = []
            while self.spike_queue and self.spike_queue[0].timestamp <= self.current_time:
                current_spikes.append(heapq.heappop(self.spike_queue))
                
            # Group spikes by target neurons
            spikes_by_neuron = {}
            for spike in current_spikes:
                # Find neurons that receive this spike
                for neuron_id, neuron in self.neurons.items():
                    if spike.neuron_id in neuron.synaptic_weights:
                        if neuron_id not in spikes_by_neuron:
                            spikes_by_neuron[neuron_id] = []
                        spikes_by_neuron[neuron_id].append(spike)
                        
            # Update neurons and generate new spikes
            for neuron_id, received_spikes in spikes_by_neuron.items():
                neuron = self.neurons[neuron_id]
                if neuron.update_potential(received_spikes, self.current_time):
                    # Neuron spiked, create output spike
                    output_spike = Spike(
                        neuron_id=neuron_id,
                        timestamp=self.current_time + 1.0,  # 1ms delay
                        spike_type=SpikeType.EXCITATORY
                    )
                    heapq.heappush(self.spike_queue, output_spike)
                    
                    # Record output spikes
                    if neuron_id in self.output_neurons:
                        output_spikes[neuron_id].append(self.current_time)
                        
            self.current_time += self.time_step
            
        return output_spikes
        
    def decode_output(self, output_spikes: Dict[int, List[float]]) -> np.ndarray:
        """Decode output spikes into classification probabilities."""
        
        # Count spikes for each output neuron
        spike_counts = np.zeros(self.output_size)
        
        for i, neuron_id in enumerate(self.output_neurons):
            spike_counts[i] = len(output_spikes[neuron_id])
            
        # Convert to probabilities (softmax-like)
        if spike_counts.sum() > 0:
            probabilities = spike_counts / spike_counts.sum()
        else:
            probabilities = np.ones(self.output_size) / self.output_size
            
        return probabilities


class PowerOptimizedProcessor:
    """Ultra-low power processor for neuromorphic inference."""
    
    def __init__(self):
        self.power_budget = 50.0  # mW
        self.current_power_usage = 0.0
        self.adaptive_scaling = True
        self.processing_frequency = 1000.0  # Hz
        
        # Power consumption models (mW)
        self.power_models = {
            "spike_processing": 0.001,     # Per spike
            "memory_access": 0.01,         # Per access
            "computation": 0.1,            # Per operation
            "communication": 1.0,          # Per transmission
            "idle": 5.0                    # Base idle power
        }
        
    def estimate_power_consumption(
        self, 
        num_spikes: int, 
        num_operations: int, 
        num_memory_accesses: int,
        duration: float
    ) -> float:
        """Estimate power consumption for given workload."""
        
        spike_power = num_spikes * self.power_models["spike_processing"]
        memory_power = num_memory_accesses * self.power_models["memory_access"]
        compute_power = num_operations * self.power_models["computation"]
        idle_power = self.power_models["idle"] * duration / 1000.0  # Convert ms to seconds
        
        total_power = spike_power + memory_power + compute_power + idle_power
        
        return total_power
        
    def optimize_for_power_budget(self, estimated_power: float) -> Dict[str, float]:
        """Optimize processing parameters to meet power budget."""
        
        optimization_params = {
            "frequency_scaling": 1.0,
            "precision_scaling": 1.0,
            "spike_rate_limit": float('inf'),
            "network_pruning": 0.0
        }
        
        if estimated_power > self.power_budget:
            # Apply power optimization strategies
            power_ratio = self.power_budget / estimated_power
            
            # Frequency scaling (most effective)
            optimization_params["frequency_scaling"] = min(1.0, power_ratio * 1.2)
            
            # Precision reduction
            if power_ratio < 0.8:
                optimization_params["precision_scaling"] = 0.8
                
            # Spike rate limiting
            if power_ratio < 0.6:
                optimization_params["spike_rate_limit"] = 50.0  # Hz
                
            # Network pruning (aggressive)
            if power_ratio < 0.4:
                optimization_params["network_pruning"] = 0.3  # Remove 30% of connections
                
        return optimization_params


class EdgeInferenceEngine:
    """Main edge inference engine with neuromorphic computing."""
    
    def __init__(self, device_config: Dict[str, Any]):
        self.device_id = device_config.get("device_id", "edge_device_001")
        self.max_power_budget = device_config.get("max_power_mw", 100.0)
        self.target_latency = device_config.get("target_latency_ms", 50.0)
        
        # Initialize components
        self.snn = SpikingNeuralNetwork(device_config.get("network_architecture", {}))
        self.power_optimizer = PowerOptimizedProcessor()
        self.power_optimizer.power_budget = self.max_power_budget
        
        # Inference modes and their configurations
        self.inference_modes = {
            InferenceMode.ULTRA_LOW_POWER: {
                "power_limit": 10.0,
                "accuracy_target": 0.80,
                "frequency_scale": 0.5,
                "precision_bits": 4
            },
            InferenceMode.LOW_POWER: {
                "power_limit": 50.0,
                "accuracy_target": 0.90,
                "frequency_scale": 0.8,
                "precision_bits": 8
            },
            InferenceMode.BALANCED: {
                "power_limit": 200.0,
                "accuracy_target": 0.95,
                "frequency_scale": 1.0,
                "precision_bits": 16
            },
            InferenceMode.HIGH_ACCURACY: {
                "power_limit": 500.0,
                "accuracy_target": 0.98,
                "frequency_scale": 1.2,
                "precision_bits": 32
            },
            InferenceMode.EMERGENCY: {
                "power_limit": float('inf'),
                "accuracy_target": 0.99,
                "frequency_scale": 2.0,
                "precision_bits": 32
            }
        }
        
        self.current_mode = InferenceMode.BALANCED
        self.performance_history = deque(maxlen=100)
        
    def set_inference_mode(self, mode: InferenceMode) -> None:
        """Set the inference mode based on power/accuracy requirements."""
        self.current_mode = mode
        mode_config = self.inference_modes[mode]
        
        # Update power budget
        self.power_optimizer.power_budget = mode_config["power_limit"]
        
        # Update processing frequency
        base_frequency = 1000.0  # Hz
        self.power_optimizer.processing_frequency = base_frequency * mode_config["frequency_scale"]
        
        logging.info(f"Switched to {mode.value} mode: {mode_config['power_limit']}mW, target accuracy {mode_config['accuracy_target']}")
        
    def predict(self, input_data: np.ndarray, priority: str = "routine") -> Dict[str, Any]:
        """Perform neuromorphic inference on input data."""
        
        start_time = time.time()
        
        # Adjust mode based on priority
        if priority == "emergency":
            original_mode = self.current_mode
            self.set_inference_mode(InferenceMode.EMERGENCY)
        else:
            original_mode = None
            
        try:
            # Encode input to spikes
            input_spikes = self.snn.encode_input(input_data)
            
            # Estimate computational requirements
            estimated_power = self._estimate_inference_power(input_spikes)
            
            # Optimize for power budget
            optimization_params = self.power_optimizer.optimize_for_power_budget(estimated_power)
            
            # Apply optimizations
            optimized_spikes = self._apply_optimizations(input_spikes, optimization_params)
            
            # Perform spiking neural network inference
            output_spikes = self.snn.forward_pass(optimized_spikes)
            
            # Decode output
            probabilities = self.snn.decode_output(output_spikes)
            
            # Calculate metrics
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            actual_power = self._measure_actual_power_consumption(len(optimized_spikes), inference_time)
            
            # Store performance history
            self.performance_history.append({
                "timestamp": time.time(),
                "inference_time_ms": inference_time,
                "power_consumption_mw": actual_power,
                "num_spikes": len(optimized_spikes),
                "mode": self.current_mode.value
            })
            
            # Determine prediction
            prediction_class = int(np.argmax(probabilities))
            confidence = float(np.max(probabilities))
            
            result = {
                "prediction": "pneumonia" if prediction_class == 1 else "normal",
                "confidence": confidence,
                "probabilities": probabilities.tolist(),
                "inference_time_ms": inference_time,
                "power_consumption_mw": actual_power,
                "num_spikes_processed": len(optimized_spikes),
                "inference_mode": self.current_mode.value,
                "device_id": self.device_id,
                "optimization_applied": optimization_params
            }
            
            return result
            
        finally:
            # Restore original mode if changed for emergency
            if original_mode:
                self.set_inference_mode(original_mode)
                
    def _estimate_inference_power(self, spikes: List[Spike]) -> float:
        """Estimate power consumption for inference."""
        
        num_spikes = len(spikes)
        num_neurons = len(self.snn.neurons)
        
        # Estimate operations (simplified)
        num_operations = num_spikes * 10  # Approximate operations per spike
        num_memory_accesses = num_spikes * 5  # Memory accesses per spike
        
        # Estimate duration based on simulation time
        duration = 50.0  # ms (default simulation time)
        
        return self.power_optimizer.estimate_power_consumption(
            num_spikes, num_operations, num_memory_accesses, duration
        )
        
    def _apply_optimizations(
        self, 
        spikes: List[Spike], 
        optimization_params: Dict[str, float]
    ) -> List[Spike]:
        """Apply power optimizations to spike trains."""
        
        optimized_spikes = spikes.copy()
        
        # Frequency scaling (reduce spike rates)
        frequency_scale = optimization_params["frequency_scaling"]
        if frequency_scale < 1.0:
            # Remove spikes randomly to reduce frequency
            num_to_keep = int(len(optimized_spikes) * frequency_scale)
            indices = np.random.choice(len(optimized_spikes), num_to_keep, replace=False)
            optimized_spikes = [optimized_spikes[i] for i in sorted(indices)]
            
        # Spike rate limiting
        spike_rate_limit = optimization_params["spike_rate_limit"]
        if spike_rate_limit != float('inf'):
            # Group spikes by neuron and limit rate
            spikes_by_neuron = {}
            for spike in optimized_spikes:
                if spike.neuron_id not in spikes_by_neuron:
                    spikes_by_neuron[spike.neuron_id] = []
                spikes_by_neuron[spike.neuron_id].append(spike)
                
            limited_spikes = []
            for neuron_spikes in spikes_by_neuron.values():
                # Sort by timestamp
                neuron_spikes.sort(key=lambda s: s.timestamp)
                
                # Apply rate limiting
                last_spike_time = 0
                min_interval = 1000.0 / spike_rate_limit  # ms
                
                for spike in neuron_spikes:
                    if spike.timestamp - last_spike_time >= min_interval:
                        limited_spikes.append(spike)
                        last_spike_time = spike.timestamp
                        
            optimized_spikes = limited_spikes
            
        return optimized_spikes
        
    def _measure_actual_power_consumption(self, num_spikes: int, duration_ms: float) -> float:
        """Measure actual power consumption (simulated)."""
        
        # Simulate power measurement based on activity
        base_power = self.power_optimizer.power_models["idle"]
        spike_power = num_spikes * self.power_optimizer.power_models["spike_processing"]
        compute_power = (duration_ms / 10.0) * self.power_optimizer.power_models["computation"]
        
        total_power = base_power + spike_power + compute_power
        
        # Add some measurement noise
        noise = np.random.normal(0, total_power * 0.05)
        return max(0, total_power + noise)
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        
        if not self.performance_history:
            return {}
            
        recent_history = list(self.performance_history)
        
        avg_inference_time = np.mean([h["inference_time_ms"] for h in recent_history])
        avg_power = np.mean([h["power_consumption_mw"] for h in recent_history])
        avg_spikes = np.mean([h["num_spikes"] for h in recent_history])
        
        return {
            "average_inference_time_ms": avg_inference_time,
            "average_power_consumption_mw": avg_power,
            "average_spikes_processed": avg_spikes,
            "total_inferences": len(recent_history),
            "current_mode": self.current_mode.value,
            "power_efficiency": avg_spikes / max(avg_power, 1),  # Spikes per mW
            "energy_per_inference_mj": (avg_power * avg_inference_time / 1000.0),  # millijoules
            "history": recent_history[-10:]  # Last 10 inferences
        }
        
    def adaptive_mode_selection(self, context: Dict[str, Any]) -> InferenceMode:
        """Automatically select optimal inference mode based on context."""
        
        battery_level = context.get("battery_level", 1.0)  # 0.0 to 1.0
        urgency = context.get("urgency", "routine")  # "emergency", "urgent", "routine"
        accuracy_requirement = context.get("accuracy_requirement", 0.90)
        
        # Emergency cases always use maximum accuracy
        if urgency == "emergency":
            return InferenceMode.EMERGENCY
            
        # Low battery - use power-efficient modes
        if battery_level < 0.2:  # <20% battery
            return InferenceMode.ULTRA_LOW_POWER
        elif battery_level < 0.5:  # <50% battery
            return InferenceMode.LOW_POWER
            
        # High accuracy requirements
        if accuracy_requirement > 0.95:
            return InferenceMode.HIGH_ACCURACY
        elif accuracy_requirement > 0.90:
            return InferenceMode.BALANCED
        else:
            return InferenceMode.LOW_POWER


class DistributedEdgeCluster:
    """Cluster of edge devices for distributed neuromorphic inference."""
    
    def __init__(self):
        self.edge_devices: Dict[str, EdgeInferenceEngine] = {}
        self.load_balancer = EdgeLoadBalancer()
        self.consensus_engine = ConsensusEngine()
        
    def add_edge_device(self, device_config: Dict[str, Any]) -> str:
        """Add an edge device to the cluster."""
        device = EdgeInferenceEngine(device_config)
        device_id = device.device_id
        self.edge_devices[device_id] = device
        self.load_balancer.register_device(device)
        return device_id
        
    def distributed_inference(
        self, 
        input_data: np.ndarray, 
        consensus_required: bool = True,
        min_devices: int = 3
    ) -> Dict[str, Any]:
        """Perform distributed inference across multiple edge devices."""
        
        # Select devices for inference
        selected_devices = self.load_balancer.select_devices(min_devices)
        
        if len(selected_devices) < min_devices:
            raise RuntimeError(f"Insufficient devices available: {len(selected_devices)} < {min_devices}")
            
        # Perform inference on each device
        individual_results = {}
        for device in selected_devices:
            result = device.predict(input_data)
            individual_results[device.device_id] = result
            
        # Apply consensus if required
        if consensus_required:
            consensus_result = self.consensus_engine.reach_consensus(individual_results)
        else:
            # Use result from device with highest confidence
            best_device = max(selected_devices, key=lambda d: individual_results[d.device_id]["confidence"])
            consensus_result = individual_results[best_device.device_id]
            
        # Add metadata about distributed inference
        consensus_result["distributed_inference"] = {
            "num_devices": len(selected_devices),
            "device_ids": [d.device_id for d in selected_devices],
            "consensus_applied": consensus_required,
            "individual_results": individual_results
        }
        
        return consensus_result


class EdgeLoadBalancer:
    """Load balancer for edge device cluster."""
    
    def __init__(self):
        self.devices: List[EdgeInferenceEngine] = []
        
    def register_device(self, device: EdgeInferenceEngine) -> None:
        """Register an edge device."""
        self.devices.append(device)
        
    def select_devices(self, num_devices: int) -> List[EdgeInferenceEngine]:
        """Select best devices for inference based on current state."""
        
        # Score devices based on performance and availability
        device_scores = []
        for device in self.devices:
            metrics = device.get_performance_metrics()
            
            # Calculate composite score
            power_efficiency = metrics.get("power_efficiency", 1.0)
            avg_inference_time = metrics.get("average_inference_time_ms", 100.0)
            
            # Lower inference time and higher power efficiency is better
            score = power_efficiency / max(avg_inference_time, 1.0)
            device_scores.append((score, device))
            
        # Sort by score and select top devices
        device_scores.sort(reverse=True, key=lambda x: x[0])
        selected = [device for _, device in device_scores[:num_devices]]
        
        return selected


class ConsensusEngine:
    """Consensus engine for aggregating results from multiple edge devices."""
    
    def reach_consensus(self, individual_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Reach consensus from multiple device results."""
        
        if not individual_results:
            raise ValueError("No results to reach consensus on")
            
        # Extract predictions and confidences
        predictions = []
        confidences = []
        probabilities = []
        
        for device_id, result in individual_results.items():
            predictions.append(result["prediction"])
            confidences.append(result["confidence"])
            probabilities.append(result["probabilities"])
            
        # Majority voting for prediction
        from collections import Counter
        prediction_counts = Counter(predictions)
        consensus_prediction = prediction_counts.most_common(1)[0][0]
        
        # Weighted average of probabilities (weighted by confidence)
        probabilities_array = np.array(probabilities)
        confidences_array = np.array(confidences)
        
        # Normalize weights
        normalized_weights = confidences_array / confidences_array.sum()
        
        # Weighted average
        consensus_probabilities = np.average(probabilities_array, axis=0, weights=normalized_weights)
        consensus_confidence = float(np.max(consensus_probabilities))
        
        # Consensus metadata
        consensus_strength = prediction_counts[consensus_prediction] / len(predictions)
        confidence_variance = np.var(confidences)
        
        consensus_result = {
            "prediction": consensus_prediction,
            "confidence": consensus_confidence,
            "probabilities": consensus_probabilities.tolist(),
            "consensus_metadata": {
                "consensus_strength": consensus_strength,
                "confidence_variance": confidence_variance,
                "num_agreeing_devices": prediction_counts[consensus_prediction],
                "total_devices": len(individual_results)
            }
        }
        
        # Add timing and power information (averages)
        avg_inference_time = np.mean([r["inference_time_ms"] for r in individual_results.values()])
        avg_power = np.mean([r["power_consumption_mw"] for r in individual_results.values()])
        
        consensus_result.update({
            "inference_time_ms": avg_inference_time,
            "power_consumption_mw": avg_power,
            "consensus_applied": True
        })
        
        return consensus_result


if __name__ == "__main__":
    # Example usage and demonstration
    
    # Create edge inference engine
    device_config = {
        "device_id": "medical_edge_001",
        "max_power_mw": 100.0,
        "target_latency_ms": 50.0,
        "network_architecture": {
            "input_size": 784,  # 28x28 image flattened
            "hidden_layers": [128, 64],
            "output_size": 2  # normal, pneumonia
        }
    }
    
    edge_engine = EdgeInferenceEngine(device_config)
    
    # Test different inference modes
    test_modes = [
        InferenceMode.ULTRA_LOW_POWER,
        InferenceMode.LOW_POWER,
        InferenceMode.BALANCED,
        InferenceMode.HIGH_ACCURACY
    ]
    
    # Generate synthetic medical image data
    test_image = np.random.rand(28, 28).astype(np.float32)
    
    print("Neuromorphic Edge Inference Testing\\n")
    print("=" * 50)
    
    for mode in test_modes:
        edge_engine.set_inference_mode(mode)
        
        # Perform inference
        result = edge_engine.predict(test_image, priority="routine")
        
        print(f"\\nMode: {mode.value}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Inference Time: {result['inference_time_ms']:.1f} ms")
        print(f"Power Consumption: {result['power_consumption_mw']:.1f} mW")
        print(f"Spikes Processed: {result['num_spikes_processed']}")
        
    # Test adaptive mode selection
    print("\\n" + "=" * 50)
    print("Adaptive Mode Selection Testing\\n")
    
    test_contexts = [
        {"battery_level": 0.1, "urgency": "routine", "accuracy_requirement": 0.85},
        {"battery_level": 0.8, "urgency": "urgent", "accuracy_requirement": 0.95},
        {"battery_level": 0.5, "urgency": "emergency", "accuracy_requirement": 0.90},
    ]
    
    for i, context in enumerate(test_contexts):
        optimal_mode = edge_engine.adaptive_mode_selection(context)
        edge_engine.set_inference_mode(optimal_mode)
        
        result = edge_engine.predict(test_image, priority=context["urgency"])
        
        print(f"Context {i+1}: Battery={context['battery_level']:.1f}, "
              f"Urgency={context['urgency']}, Accuracy={context['accuracy_requirement']:.2f}")
        print(f"Selected Mode: {optimal_mode.value}")
        print(f"Power: {result['power_consumption_mw']:.1f} mW, "
              f"Time: {result['inference_time_ms']:.1f} ms\\n")
    
    # Test distributed edge cluster
    print("=" * 50)
    print("Distributed Edge Cluster Testing\\n")
    
    cluster = DistributedEdgeCluster()
    
    # Add multiple edge devices
    for i in range(5):
        device_config_cluster = {
            "device_id": f"edge_device_{i:03d}",
            "max_power_mw": 50.0 + i * 20,  # Varying power budgets
            "target_latency_ms": 30.0 + i * 10,
            "network_architecture": {
                "input_size": 784,
                "hidden_layers": [64, 32] if i < 3 else [128, 64],  # Varying architectures
                "output_size": 2
            }
        }
        cluster.add_edge_device(device_config_cluster)
    
    # Perform distributed inference
    distributed_result = cluster.distributed_inference(
        test_image, 
        consensus_required=True, 
        min_devices=3
    )
    
    print(f"Distributed Prediction: {distributed_result['prediction']}")
    print(f"Consensus Confidence: {distributed_result['confidence']:.3f}")
    print(f"Devices Used: {distributed_result['distributed_inference']['num_devices']}")
    print(f"Consensus Strength: {distributed_result['consensus_metadata']['consensus_strength']:.3f}")
    
    # Performance metrics
    metrics = edge_engine.get_performance_metrics()
    print(f"\\nPerformance Summary:")
    print(f"Average Inference Time: {metrics.get('average_inference_time_ms', 0):.1f} ms")
    print(f"Average Power: {metrics.get('average_power_consumption_mw', 0):.1f} mW")
    print(f"Power Efficiency: {metrics.get('power_efficiency', 0):.2f} spikes/mW")
    print(f"Energy per Inference: {metrics.get('energy_per_inference_mj', 0):.2f} mJ")
    
    print("\\nNeuromorphic edge inference demonstration complete")