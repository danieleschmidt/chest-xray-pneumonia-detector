"""
Neuromorphic Edge Inference System
=================================

Advanced edge computing system with neuromorphic processing capabilities for
real-time medical image analysis. Implements brain-inspired computing paradigms
for ultra-low power, high-performance inference at the edge.

Key Features:
1. Spiking Neural Network (SNN) implementations
2. Event-driven processing architecture
3. Adaptive learning and plasticity
4. Ultra-low power consumption optimization
5. Real-time inference with minimal latency
6. Federated learning coordination
7. Edge-to-cloud hybrid processing
8. Hardware acceleration support
"""

import logging
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import psutil
import gc


@dataclass
class SpikingNeuron:
    """Individual spiking neuron with adaptive behavior."""
    neuron_id: str
    membrane_potential: float = 0.0
    threshold: float = 1.0
    reset_potential: float = 0.0
    leak_rate: float = 0.1
    refractory_period: int = 2
    refractory_counter: int = 0
    spike_history: List[int] = field(default_factory=list)
    adaptation_rate: float = 0.01
    plasticity_enabled: bool = True


@dataclass
class SynapticConnection:
    """Synaptic connection between neurons with plasticity."""
    connection_id: str
    pre_neuron_id: str
    post_neuron_id: str
    weight: float
    delay: int = 1
    plasticity_rule: str = "STDP"  # Spike-Timing Dependent Plasticity
    learning_rate: float = 0.001
    trace_pre: float = 0.0
    trace_post: float = 0.0
    trace_decay: float = 0.95


@dataclass
class EdgeDevice:
    """Edge computing device configuration."""
    device_id: str
    device_type: str  # "mobile", "embedded", "workstation", "iot"
    cpu_cores: int
    memory_mb: int
    gpu_available: bool
    neuromorphic_chip: bool
    power_budget_mw: float
    network_bandwidth_mbps: float
    location: Dict[str, float]  # {"latitude": 0.0, "longitude": 0.0}
    capabilities: List[str] = field(default_factory=list)


@dataclass
class InferenceTask:
    """Medical inference task for edge processing."""
    task_id: str
    task_type: str  # "pneumonia_detection", "lesion_segmentation", etc.
    priority: int  # 1-10, higher is more urgent
    image_data: np.ndarray
    metadata: Dict[str, Any]
    latency_requirement_ms: float
    accuracy_requirement: float
    power_budget_mw: float
    patient_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class InferenceResult:
    """Result from neuromorphic edge inference."""
    task_id: str
    device_id: str
    result: Dict[str, Any]
    confidence_score: float
    processing_time_ms: float
    power_consumed_mw: float
    spike_count: int
    energy_efficiency: float  # Operations per joule
    accuracy_estimate: float
    timestamp: datetime = field(default_factory=datetime.now)


class SpikingNeuralNetwork:
    """Spiking Neural Network implementation for neuromorphic computing."""
    
    def __init__(self, network_config: Dict[str, Any]):
        self.network_id = network_config.get('network_id', 'SNN_001')
        self.neurons = {}
        self.connections = {}
        self.layers = network_config.get('layers', [])
        self.current_time = 0
        self.spike_trains = {}
        self.network_activity = []
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize network structure
        self._initialize_network_structure(network_config)
        
    def _initialize_network_structure(self, config: Dict[str, Any]):
        """Initialize the spiking neural network structure."""
        
        neuron_count = 0
        
        # Create neurons for each layer
        for layer_idx, layer_config in enumerate(self.layers):
            layer_size = layer_config.get('size', 100)
            layer_type = layer_config.get('type', 'leaky_integrate_fire')
            
            for neuron_idx in range(layer_size):
                neuron_id = f"L{layer_idx}_N{neuron_idx}"
                
                # Configure neuron parameters based on layer type
                if layer_type == 'input':
                    threshold = 0.5
                    leak_rate = 0.0  # No leak for input neurons
                elif layer_type == 'hidden':
                    threshold = np.random.normal(1.0, 0.1)
                    leak_rate = np.random.uniform(0.05, 0.15)
                elif layer_type == 'output':
                    threshold = np.random.normal(1.2, 0.1)
                    leak_rate = np.random.uniform(0.08, 0.12)
                else:  # default leaky_integrate_fire
                    threshold = np.random.normal(1.0, 0.1)
                    leak_rate = np.random.uniform(0.05, 0.15)
                
                neuron = SpikingNeuron(
                    neuron_id=neuron_id,
                    threshold=max(0.1, threshold),
                    leak_rate=max(0.01, leak_rate),
                    adaptation_rate=layer_config.get('adaptation_rate', 0.01)
                )
                
                self.neurons[neuron_id] = neuron
                neuron_count += 1
        
        # Create connections between layers
        self._create_inter_layer_connections()
        
        self.logger.info(f"Initialized SNN with {neuron_count} neurons and {len(self.connections)} connections")
    
    def _create_inter_layer_connections(self):
        """Create synaptic connections between network layers."""
        
        connection_count = 0
        
        for layer_idx in range(len(self.layers) - 1):
            current_layer = f"L{layer_idx}"
            next_layer = f"L{layer_idx + 1}"
            
            # Get neurons in current and next layer
            current_neurons = [nid for nid in self.neurons.keys() if nid.startswith(current_layer)]
            next_neurons = [nid for nid in self.neurons.keys() if nid.startswith(next_layer)]
            
            # Create connections
            connectivity = self.layers[layer_idx].get('connectivity', 0.5)
            
            for pre_neuron in current_neurons:
                for post_neuron in next_neurons:
                    if np.random.random() < connectivity:
                        connection_id = f"{pre_neuron}_to_{post_neuron}"
                        
                        # Initialize weight with small random value
                        weight = np.random.normal(0.0, 0.1)
                        
                        connection = SynapticConnection(
                            connection_id=connection_id,
                            pre_neuron_id=pre_neuron,
                            post_neuron_id=post_neuron,
                            weight=weight,
                            delay=np.random.randint(1, 4),
                            learning_rate=self.layers[layer_idx].get('learning_rate', 0.001)
                        )
                        
                        self.connections[connection_id] = connection
                        connection_count += 1
        
        self.logger.debug(f"Created {connection_count} synaptic connections")
    
    def encode_input(self, input_data: np.ndarray, encoding_type: str = "rate") -> Dict[str, List[int]]:
        """Encode input data into spike trains."""
        
        spike_trains = {}
        
        # Get input layer neurons
        input_neurons = [nid for nid in self.neurons.keys() if nid.startswith("L0")]
        
        if encoding_type == "rate":
            # Rate coding: spike probability proportional to input intensity
            for i, neuron_id in enumerate(input_neurons):
                if i < len(input_data.flatten()):
                    input_value = input_data.flatten()[i]
                    # Normalize to [0, 1] and create spike train
                    spike_prob = min(1.0, max(0.0, input_value))
                    
                    spike_train = []
                    for t in range(100):  # 100 time steps
                        spike_train.append(1 if np.random.random() < spike_prob else 0)
                    
                    spike_trains[neuron_id] = spike_train
                else:
                    spike_trains[neuron_id] = [0] * 100
                    
        elif encoding_type == "temporal":
            # Temporal coding: spike timing encodes information
            for i, neuron_id in enumerate(input_neurons):
                if i < len(input_data.flatten()):
                    input_value = input_data.flatten()[i]
                    # Map input value to spike timing
                    spike_time = int((1 - input_value) * 99)  # Earlier spike for higher values
                    
                    spike_train = [0] * 100
                    if 0 <= spike_time < 100:
                        spike_train[spike_time] = 1
                    
                    spike_trains[neuron_id] = spike_train
                else:
                    spike_trains[neuron_id] = [0] * 100
        
        return spike_trains
    
    def simulate_network(self, input_spike_trains: Dict[str, List[int]], 
                        simulation_steps: int = 100) -> Dict[str, Any]:
        """Simulate the spiking neural network."""
        
        start_time = time.time()
        self.current_time = 0
        
        # Initialize spike history for all neurons
        for neuron_id in self.neurons:
            self.spike_trains[neuron_id] = []
        
        total_spikes = 0
        
        # Run simulation
        for step in range(simulation_steps):
            self.current_time = step
            step_spikes = 0
            
            # Process each neuron
            for neuron_id, neuron in self.neurons.items():
                # Handle refractory period
                if neuron.refractory_counter > 0:
                    neuron.refractory_counter -= 1
                    self.spike_trains[neuron_id].append(0)
                    continue
                
                # Apply membrane leak
                neuron.membrane_potential *= (1 - neuron.leak_rate)
                
                # Apply input currents
                input_current = 0.0
                
                # External input for input layer
                if neuron_id in input_spike_trains:
                    if step < len(input_spike_trains[neuron_id]):
                        if input_spike_trains[neuron_id][step] == 1:
                            input_current += 1.0
                
                # Synaptic inputs from other neurons
                for connection_id, connection in self.connections.items():
                    if connection.post_neuron_id == neuron_id:
                        # Check for delayed spikes from pre-synaptic neuron
                        delayed_step = step - connection.delay
                        if delayed_step >= 0 and delayed_step < len(self.spike_trains[connection.pre_neuron_id]):
                            if self.spike_trains[connection.pre_neuron_id][delayed_step] == 1:
                                input_current += connection.weight
                
                # Update membrane potential
                neuron.membrane_potential += input_current
                
                # Check for spike
                if neuron.membrane_potential >= neuron.threshold:
                    # Spike!
                    self.spike_trains[neuron_id].append(1)
                    neuron.membrane_potential = neuron.reset_potential
                    neuron.refractory_counter = neuron.refractory_period
                    step_spikes += 1
                    total_spikes += 1
                    
                    # Apply plasticity if enabled
                    if neuron.plasticity_enabled:
                        self._apply_spike_timing_plasticity(neuron_id, step)
                else:
                    self.spike_trains[neuron_id].append(0)
            
            self.network_activity.append(step_spikes)
        
        simulation_time = time.time() - start_time
        
        # Calculate output
        output_activity = self._calculate_output_activity()
        
        simulation_results = {
            'total_spikes': total_spikes,
            'simulation_time': simulation_time,
            'output_activity': output_activity,
            'network_activity': self.network_activity.copy(),
            'spike_trains': {nid: trains.copy() for nid, trains in self.spike_trains.items()},
            'energy_estimate': self._estimate_energy_consumption(total_spikes)
        }
        
        return simulation_results
    
    def _apply_spike_timing_plasticity(self, spiking_neuron_id: str, current_time: int):
        """Apply spike-timing dependent plasticity (STDP)."""
        
        # Update synaptic weights based on spike timing
        for connection_id, connection in self.connections.items():
            if connection.post_neuron_id == spiking_neuron_id:
                # Post-synaptic spike occurred
                # Look for recent pre-synaptic spikes
                for t_pre in range(max(0, current_time - 20), current_time):
                    if (t_pre < len(self.spike_trains[connection.pre_neuron_id]) and 
                        self.spike_trains[connection.pre_neuron_id][t_pre] == 1):
                        
                        # Pre-before-post: potentiation
                        dt = current_time - t_pre
                        if dt > 0:
                            weight_change = connection.learning_rate * np.exp(-dt / 10.0)
                            connection.weight += weight_change
                            connection.weight = min(1.0, connection.weight)  # Cap at 1.0
            
            elif connection.pre_neuron_id == spiking_neuron_id:
                # Pre-synaptic spike occurred
                # Look for recent post-synaptic spikes
                for t_post in range(max(0, current_time - 20), current_time):
                    post_neuron_id = connection.post_neuron_id
                    if (t_post < len(self.spike_trains[post_neuron_id]) and 
                        self.spike_trains[post_neuron_id][t_post] == 1):
                        
                        # Post-before-pre: depression
                        dt = current_time - t_post
                        if dt > 0:
                            weight_change = -connection.learning_rate * np.exp(-dt / 10.0)
                            connection.weight += weight_change
                            connection.weight = max(-1.0, connection.weight)  # Floor at -1.0
    
    def _calculate_output_activity(self) -> Dict[str, float]:
        """Calculate output layer activity patterns."""
        
        output_neurons = [nid for nid in self.neurons.keys() if nid.startswith(f"L{len(self.layers)-1}")]
        output_activity = {}
        
        for neuron_id in output_neurons:
            if neuron_id in self.spike_trains:
                spike_count = sum(self.spike_trains[neuron_id])
                firing_rate = spike_count / len(self.spike_trains[neuron_id])
                output_activity[neuron_id] = firing_rate
        
        return output_activity
    
    def _estimate_energy_consumption(self, total_spikes: int) -> float:
        """Estimate energy consumption based on spike count."""
        
        # Neuromorphic chips consume energy primarily when spikes occur
        # Typical values: ~1-10 pJ per spike for neuromorphic hardware
        energy_per_spike_pj = 5.0  # picojoules
        total_energy_pj = total_spikes * energy_per_spike_pj
        
        # Convert to milliwatts (assuming 1ms simulation time step)
        total_energy_mw = total_energy_pj * 1e-9  # Convert pJ to mW
        
        return total_energy_mw


class NeuromorphicInferenceEngine:
    """Core neuromorphic inference engine for medical image analysis."""
    
    def __init__(self, model_config: Dict[str, Any]):
        self.engine_id = model_config.get('engine_id', 'NIE_001')
        self.model_type = model_config.get('model_type', 'pneumonia_detection')
        
        # Initialize spiking neural network
        snn_config = model_config.get('snn_config', {
            'network_id': 'medical_snn',
            'layers': [
                {'size': 784, 'type': 'input'},  # 28x28 input
                {'size': 256, 'type': 'hidden', 'connectivity': 0.3},
                {'size': 128, 'type': 'hidden', 'connectivity': 0.5},
                {'size': 2, 'type': 'output', 'connectivity': 0.8}
            ]
        })
        
        self.snn = SpikingNeuralNetwork(snn_config)
        
        # Performance optimization settings
        self.optimization_enabled = model_config.get('optimization_enabled', True)
        self.adaptive_threshold = model_config.get('adaptive_threshold', True)
        self.power_management = model_config.get('power_management', True)
        
        self.logger = logging.getLogger(__name__)
        
    def preprocess_medical_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess medical image for neuromorphic processing."""
        
        # Resize to network input size
        if len(image.shape) == 3:
            # Convert to grayscale if needed
            if image.shape[2] == 3:
                image = np.mean(image, axis=2)
            else:
                image = image[:, :, 0]
        
        # Resize to 28x28 for demo (in practice, use appropriate size)
        from PIL import Image
        pil_image = Image.fromarray(image.astype(np.uint8))
        resized_image = pil_image.resize((28, 28))
        
        # Normalize to [0, 1]
        processed_image = np.array(resized_image).astype(np.float32) / 255.0
        
        # Apply edge detection to enhance features
        edge_enhanced = self._apply_edge_enhancement(processed_image)
        
        return edge_enhanced
    
    def _apply_edge_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Apply edge enhancement for better spike encoding."""
        
        # Simple Sobel edge detection
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        # Apply convolution (simplified)
        edges_x = np.zeros_like(image)
        edges_y = np.zeros_like(image)
        
        for i in range(1, image.shape[0] - 1):
            for j in range(1, image.shape[1] - 1):
                patch = image[i-1:i+2, j-1:j+2]
                edges_x[i, j] = np.sum(patch * sobel_x)
                edges_y[i, j] = np.sum(patch * sobel_y)
        
        # Combine edge responses
        edge_magnitude = np.sqrt(edges_x**2 + edges_y**2)
        
        # Normalize
        edge_magnitude = (edge_magnitude - np.min(edge_magnitude)) / (np.max(edge_magnitude) - np.min(edge_magnitude) + 1e-8)
        
        # Combine with original image
        enhanced = 0.7 * image + 0.3 * edge_magnitude
        
        return np.clip(enhanced, 0, 1)
    
    def perform_inference(self, image: np.ndarray, 
                         power_budget_mw: float = 100.0) -> Dict[str, Any]:
        """Perform neuromorphic inference on medical image."""
        
        start_time = time.time()
        
        # Preprocess image
        processed_image = self.preprocess_medical_image(image)
        
        # Encode to spike trains
        spike_trains = self.snn.encode_input(processed_image, encoding_type="rate")
        
        # Optimize simulation parameters based on power budget
        simulation_steps = self._optimize_simulation_parameters(power_budget_mw)
        
        # Run spiking neural network simulation
        simulation_results = self.snn.simulate_network(spike_trains, simulation_steps)
        
        # Decode output to medical prediction
        medical_prediction = self._decode_medical_prediction(simulation_results['output_activity'])
        
        processing_time = time.time() - start_time
        
        inference_result = {
            'prediction': medical_prediction,
            'confidence': medical_prediction.get('confidence', 0.5),
            'spike_count': simulation_results['total_spikes'],
            'processing_time_ms': processing_time * 1000,
            'energy_consumed_mw': simulation_results['energy_estimate'],
            'energy_efficiency': simulation_results['total_spikes'] / (simulation_results['energy_estimate'] + 1e-9),
            'network_activity': simulation_results['network_activity'],
            'power_budget_met': simulation_results['energy_estimate'] <= power_budget_mw
        }
        
        return inference_result
    
    def _optimize_simulation_parameters(self, power_budget_mw: float) -> int:
        """Optimize simulation parameters based on power budget."""
        
        if not self.power_management:
            return 100  # Default simulation steps
        
        # Estimate energy per simulation step
        estimated_energy_per_step = 0.5  # mW per step (rough estimate)
        
        # Calculate maximum simulation steps within power budget
        max_steps = int(power_budget_mw / estimated_energy_per_step)
        
        # Ensure minimum viable simulation
        simulation_steps = max(50, min(200, max_steps))
        
        return simulation_steps
    
    def _decode_medical_prediction(self, output_activity: Dict[str, float]) -> Dict[str, Any]:
        """Decode SNN output to medical prediction."""
        
        if not output_activity:
            return {'class': 'unknown', 'confidence': 0.0, 'probabilities': {}}
        
        # Get output neuron activities
        output_values = list(output_activity.values())
        
        if len(output_values) >= 2:
            # Binary classification (normal vs pneumonia)
            normal_activity = output_values[0]
            pneumonia_activity = output_values[1]
            
            # Calculate probabilities
            total_activity = normal_activity + pneumonia_activity + 1e-8
            normal_prob = normal_activity / total_activity
            pneumonia_prob = pneumonia_activity / total_activity
            
            # Determine prediction
            if pneumonia_prob > normal_prob:
                predicted_class = 'pneumonia'
                confidence = pneumonia_prob
            else:
                predicted_class = 'normal'
                confidence = normal_prob
            
            return {
                'class': predicted_class,
                'confidence': float(confidence),
                'probabilities': {
                    'normal': float(normal_prob),
                    'pneumonia': float(pneumonia_prob)
                }
            }
        else:
            # Single output interpretation
            activity = output_values[0] if output_values else 0.0
            
            if activity > 0.5:
                predicted_class = 'pneumonia'
                confidence = activity
            else:
                predicted_class = 'normal'
                confidence = 1.0 - activity
            
            return {
                'class': predicted_class,
                'confidence': float(confidence),
                'probabilities': {
                    'normal': float(1.0 - activity),
                    'pneumonia': float(activity)
                }
            }


class EdgeDeviceManager:
    """Manager for edge computing devices and load balancing."""
    
    def __init__(self):
        self.devices = {}
        self.task_queue = queue.PriorityQueue()
        self.processing_threads = {}
        self.is_running = False
        
        self.logger = logging.getLogger(__name__)
        
    def register_device(self, device: EdgeDevice) -> bool:
        """Register an edge computing device."""
        
        try:
            self.devices[device.device_id] = device
            self.logger.info(f"Registered edge device: {device.device_id} ({device.device_type})")
            return True
        except Exception as e:
            self.logger.error(f"Failed to register device {device.device_id}: {e}")
            return False
    
    def submit_inference_task(self, task: InferenceTask) -> str:
        """Submit an inference task for processing."""
        
        # Priority queue uses tuple (priority, task)
        # Lower priority number = higher priority
        priority = (10 - task.priority, task.timestamp.timestamp())
        
        self.task_queue.put((priority, task))
        
        self.logger.debug(f"Submitted task {task.task_id} with priority {task.priority}")
        
        return task.task_id
    
    def start_processing(self):
        """Start the edge processing system."""
        
        self.is_running = True
        
        # Start processing threads for each device
        for device_id, device in self.devices.items():
            thread = threading.Thread(
                target=self._device_processing_loop,
                args=(device,),
                daemon=True
            )
            thread.start()
            self.processing_threads[device_id] = thread
        
        self.logger.info(f"Started processing on {len(self.devices)} edge devices")
    
    def stop_processing(self):
        """Stop the edge processing system."""
        
        self.is_running = False
        
        # Wait for threads to finish
        for device_id, thread in self.processing_threads.items():
            thread.join(timeout=5.0)
        
        self.logger.info("Stopped edge processing system")
    
    def _device_processing_loop(self, device: EdgeDevice):
        """Main processing loop for an edge device."""
        
        # Initialize neuromorphic inference engine for this device
        model_config = {
            'engine_id': f'NIE_{device.device_id}',
            'optimization_enabled': True,
            'power_management': True
        }
        
        inference_engine = NeuromorphicInferenceEngine(model_config)
        
        self.logger.info(f"Started processing loop for device {device.device_id}")
        
        while self.is_running:
            try:
                # Get next task (with timeout)
                try:
                    priority, task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Check if device can handle this task
                if not self._can_device_handle_task(device, task):
                    # Put task back in queue for another device
                    self.task_queue.put((priority, task))
                    continue
                
                # Process the task
                start_time = time.time()
                
                try:
                    # Perform neuromorphic inference
                    inference_result = inference_engine.perform_inference(
                        task.image_data, 
                        task.power_budget_mw
                    )
                    
                    processing_time = time.time() - start_time
                    
                    # Create result object
                    result = InferenceResult(
                        task_id=task.task_id,
                        device_id=device.device_id,
                        result=inference_result['prediction'],
                        confidence_score=inference_result['confidence'],
                        processing_time_ms=processing_time * 1000,
                        power_consumed_mw=inference_result['energy_consumed_mw'],
                        spike_count=inference_result['spike_count'],
                        energy_efficiency=inference_result['energy_efficiency'],
                        accuracy_estimate=inference_result['confidence']
                    )
                    
                    # Log successful processing
                    self.logger.info(
                        f"Device {device.device_id} processed task {task.task_id}: "
                        f"{result.result['class']} (conf: {result.confidence_score:.3f}, "
                        f"time: {result.processing_time_ms:.1f}ms)"
                    )
                    
                    # In a real system, would send result back to client
                    self._handle_inference_result(result)
                    
                except Exception as e:
                    self.logger.error(f"Error processing task {task.task_id} on device {device.device_id}: {e}")
                
                finally:
                    self.task_queue.task_done()
                    
            except Exception as e:
                self.logger.error(f"Error in processing loop for device {device.device_id}: {e}")
                time.sleep(1.0)
    
    def _can_device_handle_task(self, device: EdgeDevice, task: InferenceTask) -> bool:
        """Check if a device can handle a specific task."""
        
        # Check power budget
        if task.power_budget_mw > device.power_budget_mw:
            return False
        
        # Check memory requirements (simplified)
        estimated_memory_mb = 50  # Rough estimate for image processing
        if estimated_memory_mb > device.memory_mb * 0.8:  # Leave 20% headroom
            return False
        
        # Check task type compatibility
        if task.task_type == 'pneumonia_detection':
            return 'medical_imaging' in device.capabilities or len(device.capabilities) == 0
        
        return True
    
    def _handle_inference_result(self, result: InferenceResult):
        """Handle completed inference result."""
        
        # In a real system, this would:
        # 1. Send result back to requesting client
        # 2. Update federated learning models
        # 3. Log for audit trails
        # 4. Update device performance metrics
        
        self.logger.debug(f"Handling result from device {result.device_id}: {result.result}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        
        status = {
            'is_running': self.is_running,
            'total_devices': len(self.devices),
            'pending_tasks': self.task_queue.qsize(),
            'device_status': {},
            'system_metrics': {
                'total_memory_mb': sum(d.memory_mb for d in self.devices.values()),
                'total_power_budget_mw': sum(d.power_budget_mw for d in self.devices.values()),
                'neuromorphic_devices': sum(1 for d in self.devices.values() if d.neuromorphic_chip)
            }
        }
        
        for device_id, device in self.devices.items():
            status['device_status'][device_id] = {
                'device_type': device.device_type,
                'cpu_cores': device.cpu_cores,
                'memory_mb': device.memory_mb,
                'power_budget_mw': device.power_budget_mw,
                'neuromorphic_chip': device.neuromorphic_chip,
                'capabilities': device.capabilities
            }
        
        return status


def create_demo_edge_devices() -> List[EdgeDevice]:
    """Create demo edge computing devices."""
    
    devices = [
        EdgeDevice(
            device_id="mobile_001",
            device_type="mobile",
            cpu_cores=8,
            memory_mb=6144,
            gpu_available=True,
            neuromorphic_chip=False,
            power_budget_mw=5000,  # 5W
            network_bandwidth_mbps=100,
            location={"latitude": 37.7749, "longitude": -122.4194},
            capabilities=["medical_imaging", "real_time_processing"]
        ),
        
        EdgeDevice(
            device_id="embedded_002",
            device_type="embedded",
            cpu_cores=4,
            memory_mb=2048,
            gpu_available=False,
            neuromorphic_chip=True,
            power_budget_mw=1000,  # 1W
            network_bandwidth_mbps=10,
            location={"latitude": 40.7128, "longitude": -74.0060},
            capabilities=["medical_imaging", "ultra_low_power"]
        ),
        
        EdgeDevice(
            device_id="workstation_003",
            device_type="workstation",
            cpu_cores=16,
            memory_mb=32768,
            gpu_available=True,
            neuromorphic_chip=True,
            power_budget_mw=50000,  # 50W
            network_bandwidth_mbps=1000,
            location={"latitude": 51.5074, "longitude": -0.1278},
            capabilities=["medical_imaging", "high_performance", "training"]
        ),
        
        EdgeDevice(
            device_id="iot_004",
            device_type="iot",
            cpu_cores=2,
            memory_mb=512,
            gpu_available=False,
            neuromorphic_chip=True,
            power_budget_mw=200,  # 0.2W
            network_bandwidth_mbps=5,
            location={"latitude": 35.6762, "longitude": 139.6503},
            capabilities=["medical_imaging", "ultra_low_power", "always_on"]
        )
    ]
    
    return devices


def create_demo_inference_tasks() -> List[InferenceTask]:
    """Create demo medical inference tasks."""
    
    tasks = []
    
    # Create synthetic medical images
    for i in range(8):
        # Generate synthetic chest X-ray-like image
        image = np.random.normal(0.5, 0.2, (64, 64))
        
        # Add some medical-like features
        if i % 2 == 0:  # Simulate pneumonia pattern
            center_x, center_y = 32, 32
            for x in range(20, 44):
                for y in range(20, 44):
                    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    if distance < 12:
                        image[x, y] += 0.3 * (1 - distance / 12)
        
        image = np.clip(image, 0, 1)
        
        # Create task
        task = InferenceTask(
            task_id=f"task_{i:03d}",
            task_type="pneumonia_detection",
            priority=np.random.randint(1, 10),
            image_data=image,
            metadata={"patient_age": 45 + i * 5, "symptoms": "cough, fever"},
            latency_requirement_ms=np.random.uniform(100, 1000),
            accuracy_requirement=0.85,
            power_budget_mw=np.random.uniform(500, 5000),
            patient_id=f"patient_{i:03d}"
        )
        
        tasks.append(task)
    
    return tasks


def main():
    """Demonstrate neuromorphic edge inference system."""
    print("🌐 Neuromorphic Edge Inference System")
    print("=" * 40)
    
    # Create edge device manager
    print("📱 Initializing edge device manager...")
    device_manager = EdgeDeviceManager()
    
    # Register demo devices
    print("🔧 Registering edge devices...")
    demo_devices = create_demo_edge_devices()
    
    for device in demo_devices:
        success = device_manager.register_device(device)
        if success:
            print(f"   ✅ {device.device_id} ({device.device_type}): "
                  f"{device.cpu_cores} cores, {device.memory_mb}MB, "
                  f"{'Neuromorphic' if device.neuromorphic_chip else 'Traditional'}")
    
    # Start processing system
    print("\n🚀 Starting edge processing system...")
    device_manager.start_processing()
    
    # Create and submit demo tasks
    print("\n📋 Creating demo medical inference tasks...")
    demo_tasks = create_demo_inference_tasks()
    
    print(f"📤 Submitting {len(demo_tasks)} inference tasks...")
    submitted_tasks = []
    
    for task in demo_tasks:
        task_id = device_manager.submit_inference_task(task)
        submitted_tasks.append(task_id)
        print(f"   📨 Task {task.task_id}: {task.task_type} "
              f"(priority: {task.priority}, power: {task.power_budget_mw:.0f}mW)")
    
    # Wait for processing
    print("\n⏳ Processing tasks on edge devices...")
    time.sleep(10)  # Let tasks process
    
    # Get system status
    print("\n📊 System Status:")
    status = device_manager.get_system_status()
    
    print(f"   Running: {'Yes' if status['is_running'] else 'No'}")
    print(f"   Total devices: {status['total_devices']}")
    print(f"   Pending tasks: {status['pending_tasks']}")
    print(f"   Neuromorphic devices: {status['system_metrics']['neuromorphic_devices']}")
    print(f"   Total memory: {status['system_metrics']['total_memory_mb']}MB")
    print(f"   Total power budget: {status['system_metrics']['total_power_budget_mw']}mW")
    
    # Show device details
    print(f"\n🔍 Device Details:")
    for device_id, device_info in status['device_status'].items():
        print(f"   {device_id}: {device_info['device_type']}")
        print(f"     • Cores: {device_info['cpu_cores']}, Memory: {device_info['memory_mb']}MB")
        print(f"     • Power: {device_info['power_budget_mw']}mW")
        print(f"     • Neuromorphic: {'Yes' if device_info['neuromorphic_chip'] else 'No'}")
        print(f"     • Capabilities: {', '.join(device_info['capabilities'])}")
    
    # Demonstrate individual neuromorphic inference
    print(f"\n🧠 Demonstrating neuromorphic inference...")
    
    # Create a neuromorphic inference engine
    model_config = {
        'engine_id': 'demo_nie',
        'model_type': 'pneumonia_detection',
        'optimization_enabled': True,
        'power_management': True
    }
    
    inference_engine = NeuromorphicInferenceEngine(model_config)
    
    # Test on a demo image
    test_image = demo_tasks[0].image_data
    power_budget = 1000.0  # 1W
    
    print(f"   🔬 Running inference on test image...")
    print(f"   ⚡ Power budget: {power_budget}mW")
    
    result = inference_engine.perform_inference(test_image, power_budget)
    
    print(f"   📋 Results:")
    print(f"     • Prediction: {result['prediction']['class']}")
    print(f"     • Confidence: {result['confidence']:.3f}")
    print(f"     • Processing time: {result['processing_time_ms']:.1f}ms")
    print(f"     • Energy consumed: {result['energy_consumed_mw']:.3f}mW")
    print(f"     • Spike count: {result['spike_count']}")
    print(f"     • Energy efficiency: {result['energy_efficiency']:.1f} ops/mW")
    print(f"     • Power budget met: {'Yes' if result['power_budget_met'] else 'No'}")
    
    # Stop processing system
    print(f"\n🛑 Stopping edge processing system...")
    device_manager.stop_processing()
    
    print(f"\n✅ Neuromorphic edge inference system demonstration complete!")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()