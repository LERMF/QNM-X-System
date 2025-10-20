"""
Quantum circuit builder for QNM-X-System.

Provides utilities for constructing quantum circuits optimized for edge deployment.
"""

import numpy as np
from typing import List, Dict, Optional, Union, Tuple
import logging

try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.circuit import Parameter, Gate
    from qiskit.quantum_info import Operator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False


class QuantumCircuitBuilder:
    """
    Builder for quantum circuits optimized for edge deployment.
    
    Provides methods to construct various types of quantum circuits
    with focus on efficiency and resource optimization.
    """
    
    def __init__(self, n_qubits: int, backend: str = "qiskit"):
        """
        Initialize quantum circuit builder.
        
        Args:
            n_qubits: Number of qubits in the circuit
            backend: Quantum backend to use
        """
        self.n_qubits = n_qubits
        self.backend = backend
        
    def create_parameterized_circuit(
        self, 
        n_params: int,
        depth: int = 3,
        entanglement_pattern: str = "linear"
    ) -> Union[QuantumCircuit, Callable]:
        """
        Create a parameterized quantum circuit.
        
        Args:
            n_params: Number of parameters
            depth: Circuit depth (number of layers)
            entanglement_pattern: Entanglement pattern ("linear", "circular", "all-to-all")
            
        Returns:
            Parameterized quantum circuit
        """
        if self.backend == "qiskit" and QISKIT_AVAILABLE:
            return self._create_qiskit_parameterized_circuit(n_params, depth, entanglement_pattern)
        elif self.backend == "pennylane" and PENNYLANE_AVAILABLE:
            return self._create_pennylane_parameterized_circuit(n_params, depth, entanglement_pattern)
        else:
            raise ValueError(f"Backend {self.backend} not available")
            
    def _create_qiskit_parameterized_circuit(
        self, 
        n_params: int, 
        depth: int, 
        entanglement_pattern: str
    ) -> QuantumCircuit:
        """Create Qiskit parameterized circuit."""
        qc = QuantumCircuit(self.n_qubits)
        params = [Parameter(f'θ_{i}') for i in range(n_params)]
        
        param_idx = 0
        for layer in range(depth):
            # Single-qubit rotations
            for qubit in range(self.n_qubits):
                if param_idx < len(params):
                    qc.ry(params[param_idx], qubit)
                    param_idx += 1
                if param_idx < len(params):
                    qc.rz(params[param_idx], qubit)
                    param_idx += 1
                    
            # Entangling gates
            self._add_entanglement(qc, entanglement_pattern)
            
        return qc
        
    def _create_pennylane_parameterized_circuit(
        self, 
        n_params: int, 
        depth: int, 
        entanglement_pattern: str
    ) -> Callable:
        """Create PennyLane parameterized circuit."""
        dev = qml.device("default.qubit", wires=self.n_qubits)
        
        @qml.qnode(dev)
        def circuit(params):
            param_idx = 0
            for layer in range(depth):
                # Single-qubit rotations
                for qubit in range(self.n_qubits):
                    if param_idx < len(params):
                        qml.RY(params[param_idx], wires=qubit)
                        param_idx += 1
                    if param_idx < len(params):
                        qml.RZ(params[param_idx], wires=qubit)
                        param_idx += 1
                        
                # Entangling gates
                self._add_pennylane_entanglement(entanglement_pattern)
                
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
            
        return circuit
        
    def _add_entanglement(self, qc: QuantumCircuit, pattern: str):
        """Add entanglement gates to Qiskit circuit."""
        if pattern == "linear":
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
        elif pattern == "circular":
            for i in range(self.n_qubits):
                qc.cx(i, (i + 1) % self.n_qubits)
        elif pattern == "all-to-all":
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    qc.cx(i, j)
                    
    def _add_pennylane_entanglement(self, pattern: str):
        """Add entanglement gates to PennyLane circuit."""
        if pattern == "linear":
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        elif pattern == "circular":
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
        elif pattern == "all-to-all":
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    qml.CNOT(wires=[i, j])
                    
    def create_optimization_circuit(
        self, 
        problem_type: str = "maxcut",
        problem_params: Optional[Dict] = None
    ) -> Union[QuantumCircuit, Callable]:
        """
        Create a circuit optimized for specific problem types.
        
        Args:
            problem_type: Type of optimization problem
            problem_params: Problem-specific parameters
            
        Returns:
            Optimized quantum circuit
        """
        if problem_type == "maxcut":
            return self._create_maxcut_circuit(problem_params)
        elif problem_type == "tsp":
            return self._create_tsp_circuit(problem_params)
        elif problem_type == "portfolio":
            return self._create_portfolio_circuit(problem_params)
        else:
            return self.create_parameterized_circuit(2 * self.n_qubits)
            
    def _create_maxcut_circuit(self, params: Optional[Dict]) -> Union[QuantumCircuit, Callable]:
        """Create MaxCut optimization circuit."""
        if self.backend == "qiskit" and QISKIT_AVAILABLE:
            qc = QuantumCircuit(self.n_qubits)
            # QAOA-style circuit for MaxCut
            for layer in range(2):  # p=2 QAOA
                # Mixer layer
                for qubit in range(self.n_qubits):
                    qc.ry(Parameter(f'β_{layer}_{qubit}'), qubit)
                # Problem layer (simplified)
                for i in range(self.n_qubits):
                    for j in range(i + 1, self.n_qubits):
                        qc.rzz(Parameter(f'γ_{layer}_{i}_{j}'), i, j)
            return qc
        else:
            # Fallback to parameterized circuit
            return self.create_parameterized_circuit(2 * self.n_qubits)
            
    def _create_tsp_circuit(self, params: Optional[Dict]) -> Union[QuantumCircuit, Callable]:
        """Create TSP optimization circuit."""
        # Simplified TSP circuit
        return self.create_parameterized_circuit(3 * self.n_qubits)
        
    def _create_portfolio_circuit(self, params: Optional[Dict]) -> Union[QuantumCircuit, Callable]:
        """Create portfolio optimization circuit."""
        # Simplified portfolio optimization circuit
        return self.create_parameterized_circuit(2 * self.n_qubits)
        
    def create_measurement_circuit(
        self, 
        observables: List[str],
        basis: str = "computational"
    ) -> Union[QuantumCircuit, Callable]:
        """
        Create measurement circuit for specific observables.
        
        Args:
            observables: List of observables to measure
            basis: Measurement basis
            
        Returns:
            Circuit with measurements
        """
        if self.backend == "qiskit" and QISKIT_AVAILABLE:
            qc = QuantumCircuit(self.n_qubits, self.n_qubits)
            
            if basis == "computational":
                for i in range(self.n_qubits):
                    qc.measure(i, i)
            elif basis == "x":
                for i in range(self.n_qubits):
                    qc.h(i)
                    qc.measure(i, i)
            elif basis == "y":
                for i in range(self.n_qubits):
                    qc.sdg(i)
                    qc.h(i)
                    qc.measure(i, i)
                    
            return qc
        else:
            # PennyLane measurement
            dev = qml.device("default.qubit", wires=self.n_qubits)
            
            @qml.qnode(dev)
            def measurement_circuit():
                if basis == "x":
                    for i in range(self.n_qubits):
                        qml.Hadamard(wires=i)
                elif basis == "y":
                    for i in range(self.n_qubits):
                        qml.S(wires=i)
                        qml.Hadamard(wires=i)
                        
                return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
                
            return measurement_circuit
            
    def optimize_circuit_for_edge(
        self, 
        circuit: Union[QuantumCircuit, Callable],
        constraints: Dict[str, float]
    ) -> Union[QuantumCircuit, Callable]:
        """
        Optimize circuit for edge deployment constraints.
        
        Args:
            circuit: Circuit to optimize
            constraints: Resource constraints (depth, gate_count, etc.)
            
        Returns:
            Optimized circuit
        """
        # This would implement circuit optimization techniques
        # like gate reduction, depth optimization, etc.
        return circuit
        
    def get_circuit_metrics(self, circuit: Union[QuantumCircuit, Callable]) -> Dict[str, float]:
        """
        Get circuit resource metrics.
        
        Args:
            circuit: Circuit to analyze
            
        Returns:
            Dictionary of circuit metrics
        """
        if self.backend == "qiskit" and QISKIT_AVAILABLE and isinstance(circuit, QuantumCircuit):
            return {
                "depth": circuit.depth(),
                "gate_count": circuit.size(),
                "qubit_count": circuit.num_qubits,
                "parameter_count": len(circuit.parameters)
            }
        else:
            # Fallback metrics
            return {
                "depth": self.n_qubits * 3,
                "gate_count": self.n_qubits * 6,
                "qubit_count": self.n_qubits,
                "parameter_count": 2 * self.n_qubits
            }