"""Генератор электрических цепей согласно ТЗ"""

import random
from typing import Dict, Tuple, Optional
from dc_circuit.solver import Circuit
from config import CircuitConfig


class CircuitGenerator:
    """Генератор электрических цепей: только последовательные и параллельные."""
    
    def __init__(self, config: CircuitConfig = None):
        self.config = config or CircuitConfig()
        self.difficulty_configs = self._build_difficulty_configs()
    
    def _build_difficulty_configs(self) -> Dict[int, Dict]:
        configs = {}
        # Уровень 1: Простые последовательные цепи (2-3 резистора)
        configs[1] = {"min_resistors": 2, "max_resistors": 3, "topology": "series"}
        # Уровень 2: Простые параллельные цепи (2-3 резистора)
        configs[2] = {"min_resistors": 2, "max_resistors": 3, "topology": "parallel"}
        # Уровень 3: Средние последовательные цепи (4-6 резисторов)
        configs[3] = {"min_resistors": 4, "max_resistors": 6, "topology": "series"}
        # Уровень 4: Средние параллельные цепи (4-6 резисторов)
        configs[4] = {"min_resistors": 4, "max_resistors": 6, "topology": "parallel"}
        # Уровень 5: Сложные последовательные цепи (7-10 резисторов)
        configs[5] = {"min_resistors": 7, "max_resistors": 10, "topology": "series"}
        # Уровень 6: Сложные параллельные цепи (7-10 резисторов)
        configs[6] = {"min_resistors": 7, "max_resistors": 10, "topology": "parallel"}
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
        """Генерирует последовательную цепь с заданным количеством резисторов"""
        circuit = Circuit()
        
        # Создаем узлы: A (источник), промежуточные узлы, B (земля)
        nodes = ["A"]  # Начальный узел (источник +)
        for i in range(num_resistors - 1):
            nodes.append(f"N{i+1}")  # Промежуточные узлы
        nodes.append("B")  # Конечный узел (земля)
        
        voltage = random.randint(5, 24)  # Простые значения напряжения
        circuit.add_voltage_source("A", "B", voltage)
        circuit.set_ground("B")
        
        resistors = {}
        # Создаем цепочку резисторов: A-N1-N2-...-B
        for i in range(num_resistors):
            resistance = random.randint(10, 100)  # Простые значения
            resistor_name = f"R{i+1}"
            
            if i == 0:
                # Первый резистор: A -> N1
                circuit.add_resistor("A", "N1", resistance)
                resistors[resistor_name] = ("A", "N1", resistance)
            elif i == num_resistors - 1:
                # Последний резистор: N{i-1} -> B
                prev_node = f"N{i}"
                circuit.add_resistor(prev_node, "B", resistance)
                resistors[resistor_name] = (prev_node, "B", resistance)
            else:
                # Промежуточные резисторы: N{i} -> N{i+1}
                current_node = f"N{i}"
                next_node = f"N{i+1}"
                circuit.add_resistor(current_node, next_node, resistance)
                resistors[resistor_name] = (current_node, next_node, resistance)
        
        # Для последовательных цепей только напряжение (ток одинаков везде)
        question_types = ["voltage"]
        question_type = random.choice(question_types)
        target_resistor = random.choice(list(resistors.keys()))
        
        metadata = {
            "circuit_type": "series",
            "voltage_source": voltage,
            "voltage_sources": {("A", "B"): voltage},
            "resistors": resistors,
            "question_type": question_type,
            "target_resistor": target_resistor,
            "nodes": nodes,
            "source_node": "A",
            "ground_node": "B"
        }
        
        return circuit, question_type, metadata
    
    def _generate_parallel(self, num_resistors: int) -> Tuple[Circuit, str, Dict]:
        """Генерирует параллельную цепь с заданным количеством резисторов"""
        circuit = Circuit()
        voltage = random.randint(5, 24)  # Простые значения
        circuit.add_voltage_source("A", "B", voltage)
        circuit.set_ground("B")
        
        resistors = {}
        # Все резисторы подключены параллельно между узлами A и B
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
    