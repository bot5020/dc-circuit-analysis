"""Модуль анализа электрических цепей постоянного тока.

Содержит классы для генерации, решения и верификации задач по анализу DC цепей.
"""

from dc_circuit.game import DCCircuitGame
from dc_circuit.verifier import DCCircuitVerifier
from dc_circuit.generator import CircuitGenerator
from dc_circuit.solver import CircuitSolver, Circuit
from dc_circuit.prompt import create_circuit_prompt

__all__ = [
    "DCCircuitGame",
    "DCCircuitVerifier",
    "CircuitGenerator",
    "CircuitSolver",
    "Circuit",
    "create_circuit_prompt",
]
