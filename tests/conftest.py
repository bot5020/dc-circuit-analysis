"""Конфигурация pytest."""

import pytest
import sys
import os

# Добавляем корневую папку в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_circuit_metadata():
    """Фикстура с примером метаданных цепи."""
    return {
        "circuit_type": "series",
        "voltage_source": 12.0,
        "resistors": {
            "R1": ("A", "B", 100.0),
            "R2": ("B", "C", 200.0)
        },
        "question_type": "current",
        "target_resistor": "R1",
        "nodes": ["A", "B", "C"],
        "source_node": "A",
        "ground_node": "C"
    }


@pytest.fixture
def sample_node_voltages():
    """Фикстура с примером узловых потенциалов."""
    return {
        "A": 12.0,
        "B": 8.0,
        "C": 0.0
    }


@pytest.fixture
def sample_circuit():
    """Фикстура с примером цепи."""
    from dc_circuit.solver import Circuit
    circuit = Circuit()
    circuit.add_voltage_source("A", "C", 12.0)
    circuit.add_resistor("A", "B", 100.0)
    circuit.add_resistor("B", "C", 200.0)
    circuit.set_ground("C")
    return circuit
