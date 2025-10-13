"""Упрощенный генератор электрических цепей согласно ТЗ"""

import random
from typing import Dict, Tuple, Optional
from dc_circuit.solver import Circuit
from config import CircuitConfig


class CircuitGenerator:
    """Упрощенный генератор электрических цепей: только последовательные и параллельные."""
    
    def __init__(self, config: CircuitConfig = None):
        self.config = config or CircuitConfig()
        self.difficulty_configs = self._build_difficulty_configs()
    
    def _build_difficulty_configs(self) -> Dict[int, Dict]:
        """Упрощенные конфигурации сложности (2 уровня)."""
        configs = {}
        # Уровень 1: Последовательные цепи (2 резистора)
        configs[1] = {"min_resistors": 2, "max_resistors": 2, "topology": "series"}
        # Уровень 2: Параллельные цепи (2 резистора)
        configs[2] = {"min_resistors": 2, "max_resistors": 2, "topology": "parallel"}
        return configs
    
    def generate_circuit(self, difficulty: int = 1, seed: Optional[int] = None, **kwargs) -> Tuple[Circuit, str, Dict]:
        """Генерирует простую цепь заданной сложности"""
        if seed is not None:
            random.seed(seed)
        
        config = self.difficulty_configs.get(difficulty, self.difficulty_configs[1])
        config = {**config, **kwargs}
        
        num_resistors = random.randint(config["min_resistors"], config["max_resistors"])
        topology = config["topology"]
        
        if topology == "series":
            return self._generate_series(num_resistors)
        elif topology == "parallel":
            return self._generate_parallel(num_resistors)
        else:
            # Fallback к последовательной цепи
            return self._generate_series(2)
    
    def _generate_series(self, num_resistors: int) -> Tuple[Circuit, str, Dict]:
        """Генерирует простую последовательную цепь"""
        circuit = Circuit()
        # Простые узлы: A, B, C
        nodes = ["A", "B", "C"]
        
        voltage = random.randint(5, 24)  # Простые значения напряжения
        circuit.add_voltage_source("A", "C", voltage)
        circuit.set_ground("C")
        
        resistors = {}
        # Простые сопротивления - только для 2 резисторов максимум
        for i in range(min(num_resistors, 2)):
            resistance = random.randint(10, 100)  # Простые значения
            resistor_name = f"R{i+1}"
            if i == 0:
                circuit.add_resistor("A", "B", resistance)
                resistors[resistor_name] = ("A", "B", resistance)
            else:
                circuit.add_resistor("B", "C", resistance)
                resistors[resistor_name] = ("B", "C", resistance)
        
        # Для последовательных цепей только напряжение (ток одинаков везде)
        question_types = ["voltage"]
        question_type = random.choice(question_types)
        target_resistor = random.choice(list(resistors.keys()))
        
        metadata = {
            "circuit_type": "series",
            "voltage_source": voltage,
            "voltage_sources": {("A", "C"): voltage},
            "resistors": resistors,
            "question_type": question_type,
            "target_resistor": target_resistor,
            "nodes": nodes,
            "source_node": "A",
            "ground_node": "C"
        }
        
        return circuit, question_type, metadata
    
    def _generate_parallel(self, num_resistors: int) -> Tuple[Circuit, str, Dict]:
        """Генерирует простую параллельную цепь"""
        circuit = Circuit()
        voltage = random.randint(5, 24)  # Простые значения
        circuit.add_voltage_source("A", "B", voltage)
        circuit.set_ground("B")
        
        resistors = {}
        for i in range(num_resistors):
            resistance = random.randint(10, 100)  # Простые значения
            resistor_name = f"R{i+1}"
            circuit.add_resistor("A", "B", resistance)
            resistors[resistor_name] = ("A", "B", resistance)
        
        # Для параллельных цепей только ток (напряжение одинаково на всех резисторах)
        question_types = ["current"]
        question_type = random.choice(question_types)
        target_resistor = random.choice(list(resistors.keys()))
        
        metadata = {
            "circuit_type": "parallel",
            "voltage_source": voltage,
            "voltage_sources": {("A", "B"): voltage},
            "resistors": resistors,
            "question_type": question_type,
            "target_resistor": target_resistor,
            "nodes": ["A", "B"],
            "source_node": "A",
            "ground_node": "B"
        }
        
        return circuit, question_type, metadata
    