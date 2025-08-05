"""Tests for quantum optimization algorithms."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import time

from src.quantum_inspired_task_planner.quantum_optimization import (
    QuantumAnnealer, QuantumVariationalOptimizer, AdiabaticEvolutionOptimizer,
    OptimizationResult, create_scheduling_hamiltonian
)


class TestQuantumAnnealer:
    """Test quantum annealing algorithm."""
    
    def setup_method(self):
        """Setup test environment."""
        self.annealer = QuantumAnnealer(
            initial_temperature=10.0,
            cooling_rate=0.9,
            min_temperature=0.1
        )
    
    def test_annealer_initialization(self):
        """Test annealer initialization."""
        assert self.annealer.initial_temperature == 10.0
        assert self.annealer.cooling_rate == 0.9
        assert self.annealer.min_temperature == 0.1
        assert len(self.annealer.energy_history) == 0
    
    def test_simple_optimization(self):
        """Test annealing with simple cost function."""
        # Simple cost function: minimize sum of indices
        def simple_cost(schedule):
            return sum(int(item.split('_')[1]) for item in schedule if '_' in item)
        
        initial_schedule = ["task_3", "task_1", "task_2"]
        
        result = self.annealer.anneal(simple_cost, initial_schedule, max_iterations=100)
        
        assert isinstance(result, OptimizationResult)
        assert result.iterations > 0
        assert result.execution_time > 0
        assert len(result.optimal_schedule) == len(initial_schedule)
        
        # Should find better or equal solution
        initial_cost = simple_cost(initial_schedule)
        optimal_cost = simple_cost(result.optimal_schedule)
        assert optimal_cost <= initial_cost
    
    def test_quantum_tunneling(self):
        """Test quantum tunneling mechanism."""
        schedule = ["task_1", "task_2", "task_3", "task_4"]
        
        # Test multiple tunneling operations
        tunneled_schedules = []
        for _ in range(10):
            tunneled = self.annealer._quantum_tunnel(schedule)
            tunneled_schedules.append(tunneled)
        
        # Should generate different schedules
        unique_schedules = set(tuple(s) for s in tunneled_schedules)
        assert len(unique_schedules) > 1  # Should have some variation
        
        # All should be permutations of original
        for tunneled in tunneled_schedules:
            assert sorted(tunneled) == sorted(schedule)
    
    def test_accept_transition_logic(self):
        """Test transition acceptance logic."""
        temperature = 5.0
        
        # Better energy should always be accepted
        assert self.annealer._accept_transition(10.0, 5.0, temperature) is True
        
        # Worse energy acceptance should be probabilistic
        acceptances = []
        for _ in range(100):
            accepted = self.annealer._accept_transition(5.0, 10.0, temperature)
            acceptances.append(accepted)
        
        # Should have some acceptances and some rejections
        accept_rate = sum(acceptances) / len(acceptances)
        assert 0.0 < accept_rate < 1.0
    
    def test_cooling_schedule(self):
        """Test annealing cooling schedule."""
        def constant_cost(schedule):
            return 1.0  # Constant cost to focus on cooling
        
        initial_schedule = ["task_1", "task_2"]
        
        result = self.annealer.anneal(constant_cost, initial_schedule, max_iterations=50)
        
        # Should have recorded energy history
        assert len(self.annealer.energy_history) > 0
        assert result.iterations > 0
    
    def test_empty_schedule_handling(self):
        """Test handling of empty schedules."""
        def dummy_cost(schedule):
            return len(schedule)
        
        empty_schedule = []
        
        result = self.annealer.anneal(dummy_cost, empty_schedule, max_iterations=10)
        
        assert result.optimal_schedule == []
        assert result.energy == 0.0
    
    def test_single_item_schedule(self):
        """Test handling of single-item schedules."""
        def dummy_cost(schedule):
            return len(schedule)
        
        single_schedule = ["task_1"]
        
        result = self.annealer.anneal(dummy_cost, single_schedule, max_iterations=10)
        
        assert result.optimal_schedule == ["task_1"]
        assert result.energy == 1.0


class TestQuantumVariationalOptimizer:
    """Test variational quantum optimizer."""
    
    def setup_method(self):
        """Setup test environment."""
        self.optimizer = QuantumVariationalOptimizer(num_qubits=4)
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        assert self.optimizer.num_qubits == 4
        assert self.optimizer.circuit_depth == 3
        assert self.optimizer.parameter_count == 4 * 3 * 2  # qubits * depth * params_per_gate
    
    def test_quantum_circuit_simulation(self):
        """Test quantum circuit simulation."""
        # Random parameters
        params = np.random.uniform(-np.pi, np.pi, self.optimizer.parameter_count)
        
        state = self.optimizer._quantum_circuit(params)
        
        # Should return complex quantum state
        assert isinstance(state, np.ndarray)
        assert state.dtype == complex
        assert len(state) == 2 ** self.optimizer.num_qubits
        
        # State should be normalized
        norm = np.linalg.norm(state)
        assert abs(norm - 1.0) < 1e-10
    
    def test_rotation_gate_creation(self):
        """Test quantum rotation gate creation."""
        theta, phi = np.pi / 4, np.pi / 6
        
        gate = self.optimizer._rotation_gate(theta, phi)
        
        assert gate.shape == (2, 2)
        assert gate.dtype == complex
        
        # Gate should be unitary (approximately)
        gate_dagger = gate.T.conj()
        identity_approx = gate @ gate_dagger
        identity = np.eye(2)
        
        assert np.allclose(identity_approx, identity, atol=1e-10)
    
    def test_schedule_extraction(self):
        """Test schedule extraction from quantum state."""
        # Create quantum state
        state = np.random.random(2 ** self.optimizer.num_qubits) + 1j * np.random.random(2 ** self.optimizer.num_qubits)
        state = state / np.linalg.norm(state)  # Normalize
        
        schedule = self.optimizer._extract_schedule_from_state(state)
        
        assert isinstance(schedule, list)
        assert len(schedule) <= self.optimizer.num_qubits
        
        # All items should be task identifiers
        for item in schedule:
            assert item.startswith("task_")
    
    def test_continuous_optimization(self):
        """Test continuous parameter optimization."""
        # Simple quadratic objective function
        def objective(params):
            if len(params) == 0:
                return 0.0
            return sum(x**2 for x in params[:4])  # Use first 4 params only
        
        bounds = [(-1.0, 1.0)] * 4
        
        result = self.optimizer.optimize_continuous(objective, bounds, max_iterations=50)
        
        assert isinstance(result, OptimizationResult)
        assert result.iterations > 0
        assert result.execution_time > 0
        assert isinstance(result.optimal_schedule, list)


class TestAdiabaticEvolutionOptimizer:
    """Test adiabatic evolution optimizer."""
    
    def setup_method(self):
        """Setup test environment."""
        self.optimizer = AdiabaticEvolutionOptimizer(evolution_time=1.0, time_steps=10)
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        assert self.optimizer.evolution_time == 1.0
        assert self.optimizer.time_steps == 10
    
    def test_matrix_exponential(self):
        """Test matrix exponential calculation."""
        # Simple 2x2 Hermitian matrix
        matrix = np.array([[1.0, 0.5], [0.5, -1.0]], dtype=complex)
        
        exp_matrix = self.optimizer._matrix_exponential(matrix)
        
        assert exp_matrix.shape == (2, 2)
        assert exp_matrix.dtype == complex
        
        # Should be unitary for Hermitian input
        exp_dagger = exp_matrix.T.conj()
        product = exp_matrix @ exp_dagger
        identity = np.eye(2)
        
        # Allow for numerical precision
        assert np.allclose(product, identity, atol=1e-10)
    
    def test_adiabatic_evolution(self):
        """Test adiabatic evolution process."""
        # Simple 2x2 Hamiltonians
        initial_h = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)  # X gate
        final_h = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)    # Z gate
        
        # Start in ground state of initial Hamiltonian
        initial_state = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        
        final_state = self.optimizer.evolve(initial_h, final_h, initial_state)
        
        assert final_state.shape == (2,)
        assert final_state.dtype == complex
        
        # State should remain normalized
        norm = np.linalg.norm(final_state)
        assert abs(norm - 1.0) < 1e-10
    
    def test_evolution_without_initial_state(self):
        """Test evolution starting from ground state."""
        # Simple Hamiltonians
        initial_h = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=complex)
        final_h = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=complex)
        
        final_state = self.optimizer.evolve(initial_h, final_h)
        
        assert final_state.shape == (2,)
        assert abs(np.linalg.norm(final_state) - 1.0) < 1e-10


class TestSchedulingHamiltonian:
    """Test Hamiltonian creation for scheduling problems."""
    
    def test_hamiltonian_creation(self):
        """Test creation of scheduling Hamiltonian."""
        task_priorities = {"task_1": 3.0, "task_2": 2.0, "task_3": 1.0}
        dependencies = {"task_2": ["task_1"], "task_3": ["task_1", "task_2"]}
        
        hamiltonian = create_scheduling_hamiltonian(task_priorities, dependencies)
        
        assert hamiltonian.shape == (3, 3)
        assert hamiltonian.dtype == complex
        
        # Diagonal elements should be negative priorities
        assert hamiltonian[0, 0] == -3.0  # task_1
        assert hamiltonian[1, 1] == -2.0  # task_2
        assert hamiltonian[2, 2] == -1.0  # task_3
        
        # Should be Hermitian
        assert np.allclose(hamiltonian, hamiltonian.T.conj())
    
    def test_hamiltonian_with_dependencies(self):
        """Test Hamiltonian with dependency constraints."""
        task_priorities = {"task_1": 1.0, "task_2": 1.0}
        dependencies = {"task_2": ["task_1"]}
        
        hamiltonian = create_scheduling_hamiltonian(task_priorities, dependencies)
        
        # Off-diagonal elements should represent dependencies
        assert hamiltonian[1, 0] == 0.5  # Dependency coupling
        assert hamiltonian[0, 1] == 0.5  # Hermitian constraint
    
    def test_empty_hamiltonian(self):
        """Test Hamiltonian creation with empty inputs."""
        hamiltonian = create_scheduling_hamiltonian({}, {})
        
        assert hamiltonian.shape == (0, 0)
    
    def test_hamiltonian_single_task(self):
        """Test Hamiltonian with single task."""
        task_priorities = {"task_1": 2.5}
        dependencies = {}
        
        hamiltonian = create_scheduling_hamiltonian(task_priorities, dependencies)
        
        assert hamiltonian.shape == (1, 1)
        assert hamiltonian[0, 0] == -2.5


@pytest.fixture
def optimization_test_data():
    """Fixture providing test data for optimization."""
    return {
        "task_priorities": {
            "task_1": 4.0,
            "task_2": 3.0,
            "task_3": 2.0,
            "task_4": 1.0
        },
        "dependencies": {
            "task_2": ["task_1"],
            "task_4": ["task_3"]
        },
        "simple_schedule": ["task_1", "task_2", "task_3", "task_4"]
    }


class TestOptimizationIntegration:
    """Integration tests for optimization algorithms."""
    
    def test_annealing_with_realistic_problem(self, optimization_test_data):
        """Test annealing with realistic scheduling problem."""
        priorities = optimization_test_data["task_priorities"]
        dependencies = optimization_test_data["dependencies"]
        
        def scheduling_cost(schedule):
            total_cost = 0.0
            completed = set()
            
            for i, task_id in enumerate(schedule):
                # Penalty for executing tasks out of priority order
                priority = priorities.get(task_id, 1.0)
                position_penalty = i * (5.0 - priority)  # Higher penalty for low priority tasks executed early
                
                # Penalty for dependency violations
                deps = dependencies.get(task_id, [])
                for dep in deps:
                    if dep not in completed:
                        position_penalty += 100.0  # Heavy penalty for dependency violations
                
                total_cost += position_penalty
                completed.add(task_id)
            
            return total_cost
        
        initial_schedule = list(priorities.keys())
        annealer = QuantumAnnealer(initial_temperature=100.0, cooling_rate=0.95)
        
        result = annealer.anneal(scheduling_cost, initial_schedule, max_iterations=200)
        
        # Should find valid solution
        assert result.optimal_schedule is not None
        assert len(result.optimal_schedule) == len(initial_schedule)
        
        # Should respect dependencies
        optimal_cost = scheduling_cost(result.optimal_schedule)
        assert optimal_cost < 1000  # Should avoid major dependency violations
    
    def test_variational_optimization_convergence(self):
        """Test variational optimizer convergence."""
        optimizer = QuantumVariationalOptimizer(num_qubits=3)
        
        # Simple convex objective (should converge easily)
        def convex_objective(schedule):
            return len(schedule) ** 2  # Quadratic in schedule length
        
        bounds = [(-np.pi, np.pi)] * 4
        
        result = optimizer.optimize_continuous(convex_objective, bounds, max_iterations=30)
        
        assert result.convergence_achieved or result.iterations == 30
        assert result.execution_time > 0
    
    def test_adiabatic_evolution_consistency(self):
        """Test adiabatic evolution consistency."""
        optimizer = AdiabaticEvolutionOptimizer(evolution_time=0.5, time_steps=20)
        
        # Create simple Hamiltonians
        h_initial = np.array([[0, 1], [1, 0]], dtype=complex)  # X operator
        h_final = np.array([[1, 0], [0, -1]], dtype=complex)   # Z operator
        
        # Multiple evolutions should be consistent
        initial_state = np.array([1, 0], dtype=complex)
        
        final_states = []
        for _ in range(5):
            final_state = optimizer.evolve(h_initial, h_final, initial_state.copy())
            final_states.append(final_state)
        
        # Results should be similar (allowing for numerical precision)
        for state in final_states[1:]:
            assert np.allclose(final_states[0], state, atol=1e-10)
    
    def test_optimization_performance(self, optimization_test_data):
        """Test optimization algorithm performance."""
        priorities = optimization_test_data["task_priorities"]
        
        def medium_complexity_cost(schedule):
            # Medium complexity cost function
            cost = 0.0
            for i, task_id in enumerate(schedule):
                priority = priorities.get(task_id, 1.0)
                cost += i * (5.0 - priority) + i ** 1.5
            return cost
        
        initial_schedule = list(priorities.keys())
        annealer = QuantumAnnealer()
        
        # Measure performance
        start_time = time.time()
        result = annealer.anneal(medium_complexity_cost, initial_schedule, max_iterations=100)
        execution_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert execution_time < 2.0
        assert result.execution_time < 2.0
        assert result.iterations <= 100


class TestOptimizationErrorHandling:
    """Test error handling in optimization algorithms."""
    
    def test_annealing_with_exception_in_cost_function(self):
        """Test annealing behavior when cost function raises exception."""
        def failing_cost(schedule):
            if len(schedule) > 2:
                raise ValueError("Simulated cost function failure")
            return 1.0
        
        annealer = QuantumAnnealer()
        
        with pytest.raises(ValueError):
            annealer.anneal(failing_cost, ["task_1", "task_2", "task_3"], max_iterations=10)
    
    def test_variational_with_invalid_bounds(self):
        """Test variational optimizer with invalid bounds."""
        optimizer = QuantumVariationalOptimizer(num_qubits=2)
        
        def simple_objective(schedule):
            return 1.0
        
        # Invalid bounds (min > max)
        invalid_bounds = [(1.0, -1.0), (2.0, -2.0)]
        
        # Should handle gracefully or raise appropriate exception
        with pytest.raises((ValueError, RuntimeError)):
            optimizer.optimize_continuous(simple_objective, invalid_bounds)
    
    def test_adiabatic_with_non_hermitian_hamiltonian(self):
        """Test adiabatic evolution with non-Hermitian Hamiltonian."""
        optimizer = AdiabaticEvolutionOptimizer(evolution_time=0.1, time_steps=5)
        
        # Non-Hermitian matrix
        non_hermitian = np.array([[1, 2], [3, 4]], dtype=complex)
        
        # Should still work (numerical evolution)
        try:
            result = optimizer.evolve(non_hermitian, non_hermitian)
            assert result is not None
        except Exception as e:
            # Acceptable to fail with non-physical Hamiltonian
            assert isinstance(e, (ValueError, np.linalg.LinAlgError))


class TestOptimizationUtilities:
    """Test optimization utility functions."""
    
    def test_hamiltonian_eigenvalue_properties(self):
        """Test that created Hamiltonians have expected properties."""
        task_priorities = {"task_1": 3.0, "task_2": 2.0}
        dependencies = {}
        
        hamiltonian = create_scheduling_hamiltonian(task_priorities, dependencies)
        
        # Calculate eigenvalues
        eigenvals, _ = np.linalg.eigh(hamiltonian)
        
        # For this simple case, eigenvalues should be the negative priorities
        expected_eigenvals = sorted([-3.0, -2.0])
        actual_eigenvals = sorted(eigenvals.real)
        
        assert np.allclose(actual_eigenvals, expected_eigenvals, atol=1e-10)
    
    def test_hamiltonian_with_complex_dependencies(self):
        """Test Hamiltonian creation with complex dependency structure."""
        task_priorities = {f"task_{i}": float(i) for i in range(1, 6)}
        dependencies = {
            "task_2": ["task_1"],
            "task_3": ["task_1"],
            "task_4": ["task_2", "task_3"],
            "task_5": ["task_4"]
        }
        
        hamiltonian = create_scheduling_hamiltonian(task_priorities, dependencies)
        
        assert hamiltonian.shape == (5, 5)
        
        # Check that all dependencies create couplings
        for task_id, deps in dependencies.items():
            task_idx = int(task_id.split('_')[1]) - 1
            for dep_id in deps:
                dep_idx = int(dep_id.split('_')[1]) - 1
                assert hamiltonian[task_idx, dep_idx] == 0.5
                assert hamiltonian[dep_idx, task_idx] == 0.5  # Hermitian