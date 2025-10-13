"""Генератор электрических цепей"""

import random
from typing import Dict, Tuple, Optional
from dc_circuit.solver import Circuit
from config import CircuitConfig


class CircuitGenerator:
    """Генерирует электрические цепи: последовательные и параллельные."""
    
    def __init__(self, config: CircuitConfig = None):
        self.config = config or CircuitConfig()
        self.difficulty_configs = self._build_difficulty_configs()
    
    def _build_difficulty_configs(self) -> Dict[int, Dict]:
        """Строит конфигурации сложности (3 уровня)."""
        configs = {}
        # Уровень 1: Последовательные цепи (2-4 резистора)
        configs[1] = {"min_resistors": 2, "max_resistors": 4, "topology": "series"}
        # Уровень 2: Параллельные цепи (2-5 резисторов)
        configs[2] = {"min_resistors": 2, "max_resistors": 5, "topology": "parallel"}
        # Уровень 3: Смешанные цепи (3-6 резисторов)
        configs[3] = {"min_resistors": 3, "max_resistors": 6, "topology": "mixed"}
        return configs
    
    def generate_circuit(self, difficulty: int = 1, seed: Optional[int] = None, **kwargs) -> Tuple[Circuit, str, Dict]:
        """Генерирует цепь заданной сложности"""
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
            return self._generate_mixed(num_resistors)
    
    def _generate_series(self, num_resistors: int) -> Tuple[Circuit, str, Dict]:
        """Генерирует последовательную цепь"""
        circuit = Circuit()
        nodes = [chr(65 + i) for i in range(num_resistors + 1)]
        
        voltage = random.randint(self.config.voltage_range[0], self.config.voltage_range[1])
        circuit.add_voltage_source(nodes[0], nodes[-1], voltage)
        circuit.set_ground(nodes[-1])
        
        resistors = {}
        for i in range(num_resistors):
            resistance = random.randint(self.config.resistance_range[0], self.config.resistance_range[1])
            resistor_name = f"R{i+1}"
            circuit.add_resistor(nodes[i], nodes[i+1], resistance)
            resistors[resistor_name] = (nodes[i], nodes[i+1], resistance)
        
        # Только основные типы вопросов
        question_types = ["current", "voltage", "equivalent_resistance"]
        question_type = random.choice(question_types)
        target_resistor = random.choice(list(resistors.keys()))
        
        metadata = {
            "circuit_type": "series",
            "voltage_source": voltage,
            "voltage_sources": {(nodes[0], nodes[-1]): voltage},
            "resistors": resistors,
            "question_type": question_type,
            "target_resistor": target_resistor,
            "nodes": nodes,
            "source_node": nodes[0],
            "ground_node": nodes[-1]
        }
        
        return circuit, question_type, metadata
    
    def _generate_parallel(self, num_resistors: int) -> Tuple[Circuit, str, Dict]:
        """Генерирует параллельную цепь"""
        circuit = Circuit()
        voltage = random.randint(self.config.voltage_range[0], self.config.voltage_range[1])
        circuit.add_voltage_source("A", "B", voltage)
        circuit.set_ground("B")
        
        resistors = {}
        for i in range(num_resistors):
            resistance = random.randint(self.config.resistance_range[0], self.config.resistance_range[1])
            resistor_name = f"R{i+1}"
            circuit.add_resistor("A", "B", resistance)
            resistors[resistor_name] = ("A", "B", resistance)
        
        question_types = ["current", "voltage", "equivalent_resistance"]
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
    
    def _generate_mixed(self, num_resistors: int) -> Tuple[Circuit, str, Dict]:
        """Генерирует смешанную цепь (комбинация series/parallel)
        
        ИСПРАВЛЕНО: Теперь правильно создает параллельные резисторы.
        Circuit.resistors теперь список, поэтому можно добавлять несколько резисторов
        между одними узлами без перезаписи.
        
        Топология: A ---R1--- B ---R2--- D
                                |---R3---|
                                |---R4---|
        """
        circuit = Circuit()
        voltage = random.randint(self.config.voltage_range[0], self.config.voltage_range[1])
        
        circuit.add_voltage_source("A", "D", voltage)
        circuit.set_ground("D")
        
        resistors = {}
        
        # Последовательный резистор R1 (всегда есть)
        r1_resistance = random.randint(self.config.resistance_range[0], self.config.resistance_range[1])
        circuit.add_resistor("A", "B", r1_resistance)
        resistors["R1"] = ("A", "B", r1_resistance)
        
        # Параллельные резисторы между B и D (R2, R3, ...)
        parallel_count = num_resistors - 1  # Минус R1
        
        for i in range(parallel_count):
            resistance = random.randint(self.config.resistance_range[0], self.config.resistance_range[1])
            resistor_name = f"R{i+2}"  # R2, R3, R4...
            
            # Теперь можно просто добавлять - Circuit.resistors список!
            circuit.add_resistor("B", "D", resistance)
            resistors[resistor_name] = ("B", "D", resistance)
        
        nodes = ["A", "B", "D"]
        question_types = ["current", "voltage", "equivalent_resistance"]
        question_type = random.choice(question_types)
        target_resistor = random.choice(list(resistors.keys()))
        
        metadata = {
            "circuit_type": "mixed",
            "voltage_source": voltage,
            "voltage_sources": {("A", "D"): voltage},
            "resistors": resistors,
            "question_type": question_type,
            "target_resistor": target_resistor,
            "nodes": nodes,
            "source_node": "A",
            "ground_node": "D"
        }
        
        return circuit, question_type, metadata