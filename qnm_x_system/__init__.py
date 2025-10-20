"""
QNM-X-System: Quantum-Neuro-MOE-X
Revolutionary Edge AI System with micro-LLMs, MoE, Quantum VQE, Neuromorphic computing

This package provides a unified framework for:
- Quantum Variational Quantum Eigensolver (VQE) optimization
- Neuromorphic computing with spiking neural networks
- Mixture of Experts (MoE) for micro-LLM routing
- Edge-optimized inference and deployment
"""

__version__ = "0.1.0"
__author__ = "LERMF"
__email__ = "contact@lermf.org"

from .api import QNMXSystem
from .quantum import VQEOptimizer
from .neuromorphic import NeuromorphicProcessor
from .moe import MOERouter
from .micro_llm import MicroLLMManager
from .edge import EdgeOptimizer

__all__ = [
    "QNMXSystem",
    "VQEOptimizer", 
    "NeuromorphicProcessor",
    "MOERouter",
    "MicroLLMManager",
    "EdgeOptimizer",
]