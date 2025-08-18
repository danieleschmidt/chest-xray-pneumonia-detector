"""Quantum-inspired optimization algorithms for task planning.

Implements quantum annealing, adiabatic evolution, and variational algorithms
for solving complex scheduling optimization problems.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
import logging
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of quantum-inspired optimization."""
    optimal_schedule: List[str]
    energy: float
    iterations: int
    convergence_achieved: bool
    execution_time: float


class QuantumAnnealer:
    """Advanced quantum annealing simulator with adaptive temperature control."""
    
    def __init__(self, initial_temperature: float = 100.0, 
                 cooling_rate: float = 0.95, min_temperature: float = 0.01,
                 adaptive_cooling: bool = True, quantum_coherence: float = 0.8):
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.adaptive_cooling = adaptive_cooling
        self.quantum_coherence = quantum_coherence
        self.energy_history: List[float] = []
        self.temperature_history: List[float] = []
        self.acceptance_rate_history: List[float] = []
    
    def anneal(self, cost_function: Callable[[List[str]], float],
               initial_schedule: List[str], max_iterations: int = 1000,
               target_energy: Optional[float] = None) -> OptimizationResult:
        """Perform quantum annealing optimization."""
        import time
        start_time = time.time()
        
        current_schedule = initial_schedule.copy()
        current_energy = cost_function(current_schedule)
        best_schedule = current_schedule.copy()
        best_energy = current_energy
        
        temperature = self.initial_temperature
        iteration = 0
        
        accepted_transitions = 0
        while temperature > self.min_temperature and iteration < max_iterations:
            # Early termination if target energy reached
            if target_energy is not None and best_energy <= target_energy:
                break
                
            # Generate multiple neighbor solutions for better exploration
            best_neighbor = None
            best_neighbor_energy = float('inf')
            
            for _ in range(min(5, max(1, int(temperature / 10)))):
                neighbor_schedule = self._quantum_tunnel(current_schedule, temperature)
                neighbor_energy = cost_function(neighbor_schedule)
                
                if neighbor_energy < best_neighbor_energy:
                    best_neighbor = neighbor_schedule
                    best_neighbor_energy = neighbor_energy
            
            # Accept or reject based on quantum probability
            if self._accept_transition(current_energy, best_neighbor_energy, temperature):
                current_schedule = best_neighbor
                current_energy = best_neighbor_energy
                accepted_transitions += 1
                
                if current_energy < best_energy:
                    best_schedule = current_schedule.copy()
                    best_energy = current_energy
            
            self.energy_history.append(current_energy)
            self.temperature_history.append(temperature)
            
            # Adaptive cooling based on acceptance rate
            if self.adaptive_cooling and iteration > 0 and iteration % 50 == 0:
                acceptance_rate = accepted_transitions / 50
                self.acceptance_rate_history.append(acceptance_rate)
                if acceptance_rate < 0.1:  # Too low acceptance
                    self.cooling_rate = max(0.9, self.cooling_rate - 0.01)
                elif acceptance_rate > 0.5:  # Too high acceptance
                    self.cooling_rate = min(0.99, self.cooling_rate + 0.01)
                accepted_transitions = 0
            
            temperature *= self.cooling_rate
            iteration += 1
        
        execution_time = time.time() - start_time
        convergence_achieved = temperature <= self.min_temperature
        
        return OptimizationResult(
            optimal_schedule=best_schedule,
            energy=best_energy,
            iterations=iteration,
            convergence_achieved=convergence_achieved,
            execution_time=execution_time
        )
    
    def _quantum_tunnel(self, schedule: List[str], temperature: float) -> List[str]:
        """Generate neighbor solution using temperature-dependent quantum tunneling."""
        if len(schedule) < 2:
            return schedule.copy()
        
        new_schedule = schedule.copy()
        
        # Temperature-dependent tunneling strategies
        tunnel_probability = min(1.0, temperature / self.initial_temperature)
        
        if np.random.random() < tunnel_probability * self.quantum_coherence:
            # High-energy quantum tunneling: large rearrangements
            if len(schedule) >= 4:
                # Reverse a random subsequence
                start = np.random.randint(0, len(schedule) - 2)
                end = np.random.randint(start + 2, len(schedule) + 1)
                new_schedule[start:end] = new_schedule[start:end][::-1]
            else:
                # Simple swap for short schedules
                i, j = np.random.choice(len(schedule), 2, replace=False)
                new_schedule[i], new_schedule[j] = new_schedule[j], new_schedule[i]
        else:
            # Low-energy local moves: adjacent swaps
            if len(schedule) > 1:
                i = np.random.randint(0, len(schedule) - 1)
                new_schedule[i], new_schedule[i + 1] = new_schedule[i + 1], new_schedule[i]
        
        return new_schedule
    
    def _accept_transition(self, current_energy: float, neighbor_energy: float, 
                          temperature: float) -> bool:
        """Determine if transition should be accepted using quantum probability."""
        if neighbor_energy < current_energy:
            return True
        
        # Quantum probability of tunneling to higher energy state
        energy_diff = neighbor_energy - current_energy
        probability = np.exp(-energy_diff / temperature)
        return np.random.random() < probability


class QuantumVariationalOptimizer:
    """Variational quantum algorithm for continuous optimization problems."""
    
    def __init__(self, num_qubits: int = 8):
        self.num_qubits = num_qubits
        self.circuit_depth = 3
        self.parameter_count = num_qubits * self.circuit_depth * 2  # 2 params per gate
        
    def optimize_continuous(self, objective_function: Callable[[np.ndarray], float],
                          parameter_bounds: List[Tuple[float, float]],
                          max_iterations: int = 100) -> OptimizationResult:
        """Optimize continuous parameters using variational quantum algorithm."""
        import time
        start_time = time.time()
        
        # Initialize random parameters for quantum circuit
        initial_params = np.random.uniform(-np.pi, np.pi, self.parameter_count)
        
        # Classical optimization of quantum circuit parameters
        result = minimize(
            self._quantum_cost_wrapper(objective_function),
            initial_params,
            bounds=[(-np.pi, np.pi)] * self.parameter_count,
            method='L-BFGS-B',
            options={'maxiter': max_iterations}
        )
        
        execution_time = time.time() - start_time
        
        # Extract optimal schedule from quantum state
        optimal_state = self._quantum_circuit(result.x)
        optimal_schedule = self._extract_schedule_from_state(optimal_state)
        
        return OptimizationResult(
            optimal_schedule=optimal_schedule,
            energy=result.fun,
            iterations=result.nit,
            convergence_achieved=result.success,
            execution_time=execution_time
        )
    
    def _quantum_cost_wrapper(self, objective_function: Callable) -> Callable:
        """Wrap objective function for quantum circuit optimization."""
        def wrapped_cost(params: np.ndarray) -> float:
            quantum_state = self._quantum_circuit(params)
            schedule = self._extract_schedule_from_state(quantum_state)
            return objective_function(schedule)
        return wrapped_cost
    
    def _quantum_circuit(self, parameters: np.ndarray) -> np.ndarray:
        """Simulate quantum circuit with given parameters."""
        # Initialize quantum state in superposition
        state = np.ones(2**self.num_qubits, dtype=complex) / np.sqrt(2**self.num_qubits)
        
        # Apply quantum gates with parameters
        param_idx = 0
        for layer in range(self.circuit_depth):
            for qubit in range(self.num_qubits):
                # Apply rotation gates
                theta = parameters[param_idx]
                phi = parameters[param_idx + 1]
                param_idx += 2
                
                # Simulate rotation gate effect on quantum state
                rotation_matrix = self._rotation_gate(theta, phi)
                state = self._apply_single_qubit_gate(state, rotation_matrix, qubit)
        
        return state
    
    def _rotation_gate(self, theta: float, phi: float) -> np.ndarray:
        """Create quantum rotation gate matrix."""
        cos_half = np.cos(theta / 2)
        sin_half = np.sin(theta / 2)
        exp_phi = np.exp(1j * phi)
        
        return np.array([
            [cos_half, -sin_half * exp_phi],
            [sin_half, cos_half * exp_phi]
        ], dtype=complex)
    
    def _apply_single_qubit_gate(self, state: np.ndarray, gate: np.ndarray, 
                                qubit: int) -> np.ndarray:
        """Apply single qubit gate to quantum state."""
        # Simplified simulation - in practice would use proper tensor products
        # Apply phase rotation to simulate gate effect
        n_states = len(state)
        new_state = state.copy()
        
        for i in range(n_states):
            if (i >> qubit) & 1:  # If qubit is in |1⟩ state
                new_state[i] *= gate[1, 1]
            else:  # If qubit is in |0⟩ state
                new_state[i] *= gate[0, 0]
        
        # Normalize
        norm = np.linalg.norm(new_state)
        if norm > 0:
            new_state /= norm
        
        return new_state
    
    def _extract_schedule_from_state(self, quantum_state: np.ndarray) -> List[str]:
        """Extract task schedule from quantum state measurement."""
        # Measure quantum state to get classical schedule
        probabilities = np.abs(quantum_state) ** 2
        
        # Sample from probability distribution
        measurement = np.random.choice(len(probabilities), p=probabilities)
        
        # Convert measurement to task ordering
        # This is a simplified mapping - real implementation would be more sophisticated
        task_order = []
        for i in range(self.num_qubits):
            if (measurement >> i) & 1:
                task_order.append(f"task_{i}")
        
        return task_order


class AdiabaticEvolutionOptimizer:
    """Adiabatic quantum evolution for global optimization."""
    
    def __init__(self, evolution_time: float = 10.0, time_steps: int = 100):
        self.evolution_time = evolution_time
        self.time_steps = time_steps
        
    def evolve(self, initial_hamiltonian: np.ndarray, final_hamiltonian: np.ndarray,
               initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        """Perform adiabatic evolution from initial to final Hamiltonian."""
        dt = self.evolution_time / self.time_steps
        
        if initial_state is None:
            # Start in ground state of initial Hamiltonian
            eigenvals, eigenvecs = np.linalg.eigh(initial_hamiltonian)
            initial_state = eigenvecs[:, 0]  # Ground state
        
        current_state = initial_state.copy()
        
        for step in range(self.time_steps):
            # Linear interpolation between Hamiltonians
            s = step / self.time_steps
            hamiltonian = (1 - s) * initial_hamiltonian + s * final_hamiltonian
            
            # Time evolution operator
            evolution_operator = self._matrix_exponential(-1j * hamiltonian * dt)
            
            # Evolve state
            current_state = evolution_operator @ current_state
            
            # Normalize
            current_state /= np.linalg.norm(current_state)
        
        return current_state
    
    def _matrix_exponential(self, matrix: np.ndarray) -> np.ndarray:
        """Compute matrix exponential for time evolution."""
        # Use eigendecomposition for Hermitian matrices
        eigenvals, eigenvecs = np.linalg.eigh(matrix)
        exp_eigenvals = np.exp(eigenvals)
        return eigenvecs @ np.diag(exp_eigenvals) @ eigenvecs.T.conj()


def create_scheduling_hamiltonian(task_priorities: Dict[str, float],
                                dependencies: Dict[str, List[str]]) -> np.ndarray:
    """Create Hamiltonian matrix for task scheduling problem."""
    n_tasks = len(task_priorities)
    task_ids = list(task_priorities.keys())
    
    # Initialize Hamiltonian
    hamiltonian = np.zeros((n_tasks, n_tasks), dtype=complex)
    
    # Diagonal terms: task priorities (negative for ground state minimization)
    for i, task_id in enumerate(task_ids):
        hamiltonian[i, i] = -task_priorities[task_id]
    
    # Off-diagonal terms: dependency constraints
    for i, task_id in enumerate(task_ids):
        for dep_id in dependencies.get(task_id, []):
            if dep_id in task_ids:
                j = task_ids.index(dep_id)
                # Coupling strength for dependency constraint
                hamiltonian[i, j] = 0.5
                hamiltonian[j, i] = 0.5
    
    return hamiltonian