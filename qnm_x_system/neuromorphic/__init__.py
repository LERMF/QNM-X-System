"""
Neuromorphic computing module for QNM-X-System.

Provides spiking neural networks and neuromorphic processing capabilities
optimized for edge deployment.
"""

from .neuromorphic_processor import NeuromorphicProcessor
from .spiking_networks import SpikingNeuralNetwork, LIFNeuron, STDPLearning
from .neuromorphic_utils import NeuromorphicUtils

__all__ = ["NeuromorphicProcessor", "SpikingNeuralNetwork", "LIFNeuron", "STDPLearning", "NeuromorphicUtils"]