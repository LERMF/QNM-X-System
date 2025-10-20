"""
Quantum utilities for QNM-X-System.

Provides helper functions and utilities for quantum computing operations.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import logging

try:
    from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace
    from qiskit.quantum_info.operators import Operator, SparsePauliOp
    from qiskit.circuit import Parameter
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False


class QuantumUtils:
    """
    Utility functions for quantum computing operations.
    
    Provides common quantum operations, state manipulation,
    and measurement utilities.
    """
    
    @staticmethod
    def create_entangled_state(n_qubits: int, state_type: str = "ghz") -> np.ndarray:
        """
        Create entangled quantum states.
        
        Args:
            n_qubits: Number of qubits
            state_type: Type of entangled state ("ghz", "w", "cluster")
            
        Returns:
            State vector
        """
        if state_type == "ghz":
            state = np.zeros(2**n_qubits)
            state[0] = 1/np.sqrt(2)
            state[-1] = 1/np.sqrt(2)
            return state
        elif state_type == "w":
            state = np.zeros(2**n_qubits)
            for i in range(n_qubits):
                state[2**i] = 1/np.sqrt(n_qubits)
            return state
        elif state_type == "cluster":
            # Simplified cluster state
            state = np.ones(2**n_qubits) / np.sqrt(2**n_qubits)
            return state
        else:
            raise ValueError(f"Unknown state type: {state_type}")
            
    @staticmethod
    def calculate_fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
        """
        Calculate fidelity between two quantum states.
        
        Args:
            state1: First state vector
            state2: Second state vector
            
        Returns:
            Fidelity value
        """
        if len(state1) != len(state2):
            raise ValueError("State vectors must have the same dimension")
            
        overlap = np.abs(np.vdot(state1, state2))**2
        return overlap
        
    @staticmethod
    def calculate_entanglement_entropy(state: np.ndarray, partition: int) -> float:
        """
        Calculate entanglement entropy for a given partition.
        
        Args:
            state: Quantum state vector
            partition: Partition point (number of qubits in first subsystem)
            
        Returns:
            Entanglement entropy
        """
        if QISKIT_AVAILABLE:
            # Use Qiskit for proper density matrix calculation
            statevector = Statevector(state)
            density_matrix = DensityMatrix(statevector)
            
            # Partial trace
            subsystem_dims = [2**partition, 2**(len(state).bit_length() - 1 - partition)]
            reduced_dm = partial_trace(density_matrix, [0] * partition)
            
            # Calculate eigenvalues
            eigenvals = np.real(np.linalg.eigvals(reduced_dm.data))
            eigenvals = eigenvals[eigenvals > 1e-10]  # Remove numerical zeros
            
            # Calculate von Neumann entropy
            entropy = -np.sum(eigenvals * np.log2(eigenvals))
            return entropy
        else:
            # Simplified calculation
            return np.log(2) * min(partition, len(state).bit_length() - 1 - partition)
            
    @staticmethod
    def create_pauli_string(n_qubits: int, pauli_type: str = "z") -> str:
        """
        Create Pauli string for measurement.
        
        Args:
            n_qubits: Number of qubits
            pauli_type: Type of Pauli operator ("x", "y", "z")
            
        Returns:
            Pauli string
        """
        return pauli_type * n_qubits
        
    @staticmethod
    def calculate_expectation_value(
        state: np.ndarray, 
        observable: Union[str, np.ndarray]
    ) -> float:
        """
        Calculate expectation value of an observable.
        
        Args:
            state: Quantum state vector
            observable: Observable (Pauli string or matrix)
            
        Returns:
            Expectation value
        """
        if isinstance(observable, str):
            # Convert Pauli string to matrix
            observable_matrix = QuantumUtils._pauli_string_to_matrix(observable)
        else:
            observable_matrix = observable
            
        return np.real(np.vdot(state, observable_matrix @ state))
        
    @staticmethod
    def _pauli_string_to_matrix(pauli_string: str) -> np.ndarray:
        """Convert Pauli string to matrix representation."""
        pauli_matrices = {
            'I': np.array([[1, 0], [0, 1]]),
            'X': np.array([[0, 1], [1, 0]]),
            'Y': np.array([[0, -1j], [1j, 0]]),
            'Z': np.array([[1, 0], [0, -1]])
        }
        
        # Start with identity
        matrix = np.array([[1]])
        
        for pauli in pauli_string.upper():
            matrix = np.kron(matrix, pauli_matrices[pauli])
            
        return matrix
        
    @staticmethod
    def optimize_parameters(
        objective_function: callable,
        initial_params: np.ndarray,
        method: str = "gradient_descent",
        max_iterations: int = 1000,
        learning_rate: float = 0.01
    ) -> Tuple[np.ndarray, float]:
        """
        Optimize parameters using classical optimization.
        
        Args:
            objective_function: Function to minimize
            initial_params: Initial parameter values
            method: Optimization method
            max_iterations: Maximum number of iterations
            learning_rate: Learning rate for gradient methods
            
        Returns:
            Tuple of (optimized_parameters, optimal_value)
        """
        if method == "gradient_descent":
            return QuantumUtils._gradient_descent(
                objective_function, initial_params, max_iterations, learning_rate
            )
        elif method == "adam":
            return QuantumUtils._adam_optimizer(
                objective_function, initial_params, max_iterations, learning_rate
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")
            
    @staticmethod
    def _gradient_descent(
        objective_function: callable,
        initial_params: np.ndarray,
        max_iterations: int,
        learning_rate: float
    ) -> Tuple[np.ndarray, float]:
        """Gradient descent optimization."""
        params = initial_params.copy()
        
        for _ in range(max_iterations):
            # Calculate gradient (finite difference)
            gradient = QuantumUtils._finite_difference_gradient(objective_function, params)
            
            # Update parameters
            params -= learning_rate * gradient
            
        return params, objective_function(params)
        
    @staticmethod
    def _adam_optimizer(
        objective_function: callable,
        initial_params: np.ndarray,
        max_iterations: int,
        learning_rate: float
    ) -> Tuple[np.ndarray, float]:
        """Adam optimizer implementation."""
        params = initial_params.copy()
        m = np.zeros_like(params)  # First moment
        v = np.zeros_like(params)  # Second moment
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        
        for t in range(1, max_iterations + 1):
            gradient = QuantumUtils._finite_difference_gradient(objective_function, params)
            
            # Update biased first moment estimate
            m = beta1 * m + (1 - beta1) * gradient
            
            # Update biased second raw moment estimate
            v = beta2 * v + (1 - beta2) * (gradient ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = m / (1 - beta1 ** t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = v / (1 - beta2 ** t)
            
            # Update parameters
            params -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            
        return params, objective_function(params)
        
    @staticmethod
    def _finite_difference_gradient(
        objective_function: callable,
        params: np.ndarray,
        epsilon: float = 1e-5
    ) -> np.ndarray:
        """Calculate gradient using finite differences."""
        gradient = np.zeros_like(params)
        
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += epsilon
            params_minus = params.copy()
            params_minus[i] -= epsilon
            
            gradient[i] = (objective_function(params_plus) - objective_function(params_minus)) / (2 * epsilon)
            
        return gradient
        
    @staticmethod
    def create_noise_model(
        noise_type: str = "depolarizing",
        noise_level: float = 0.01
    ) -> Dict[str, float]:
        """
        Create noise model for quantum circuits.
        
        Args:
            noise_type: Type of noise ("depolarizing", "bit_flip", "phase_flip")
            noise_level: Noise level (0-1)
            
        Returns:
            Noise model parameters
        """
        noise_models = {
            "depolarizing": {
                "single_qubit_error": noise_level,
                "two_qubit_error": noise_level * 2
            },
            "bit_flip": {
                "bit_flip_probability": noise_level
            },
            "phase_flip": {
                "phase_flip_probability": noise_level
            }
        }
        
        return noise_models.get(noise_type, noise_models["depolarizing"])
        
    @staticmethod
    def calculate_quantum_volume(
        n_qubits: int,
        circuit_depth: int,
        gate_fidelity: float = 0.99
    ) -> float:
        """
        Calculate quantum volume metric.
        
        Args:
            n_qubits: Number of qubits
            circuit_depth: Circuit depth
            gate_fidelity: Average gate fidelity
            
        Returns:
            Quantum volume
        """
        # Simplified quantum volume calculation
        effective_volume = min(n_qubits, circuit_depth) ** 2
        fidelity_factor = gate_fidelity ** (n_qubits * circuit_depth)
        
        return effective_volume * fidelity_factor