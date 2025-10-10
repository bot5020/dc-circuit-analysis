import random
from typing import Dict, Tuple, Optional
from dc_circuit.solver import Circuit


class CircuitGenerator:
    """Генерирует электрические цепи различной сложности"""
    
    def __init__(self):
        self.difficulty_configs = {
            1: {"min_resistors": 2, "max_resistors": 2, "topology": "series"},
            2: {"min_resistors": 3, "max_resistors": 3, "topology": "series"}, 
            3: {"min_resistors": 2, "max_resistors": 3, "topology": "parallel"},
            4: {"min_resistors": 3, "max_resistors": 4, "topology": "parallel"},
            5: {"min_resistors": 3, "max_resistors": 4, "topology": "mixed"},
            6: {"min_resistors": 4, "max_resistors": 5, "topology": "mixed"},
            7: {"min_resistors": 4, "max_resistors": 6, "topology": "complex"},
            8: {"min_resistors": 5, "max_resistors": 7, "topology": "complex"},
            9: {"min_resistors": 6, "max_resistors": 8, "topology": "complex"},
            10: {"min_resistors": 7, "max_resistors": 10, "topology": "complex"}
        }
    
    def generate_circuit(self, difficulty: int = 1, seed: Optional[int] = None, **kwargs) -> Tuple[Circuit, str, Dict]:
        """
        Генерирует цепь заданной сложности
        @param difficulty: уровень сложности 1-10
        @return: (circuit, question_type, metadata)
        """
        if seed is not None:
            random.seed(seed)
        base_config = self.difficulty_configs.get(difficulty, self.difficulty_configs[1])

        config: Dict = {**base_config, **kwargs}
        
        num_resistors = random.randint(config["min_resistors"], config["max_resistors"])
        topology = config["topology"]
        
        if topology == "series":
            return self._generate_series(num_resistors)
        elif topology == "parallel":
            return self._generate_parallel(num_resistors)
        elif topology == "mixed":
            return self._generate_mixed(num_resistors)
        else:
            return self._generate_complex(num_resistors, difficulty)
    
    def _generate_series(self, num_resistors: int) -> Tuple[Circuit, str, Dict]:
        """Генерирует последовательную цепь"""
        circuit = Circuit()
        
        # Создаем узлы A, B, C, D...
        nodes = [chr(65 + i) for i in range(num_resistors + 1)]
        
        # Добавляем источник напряжения
        voltage = random.randint(5, 24)
        circuit.add_voltage_source(nodes[0], nodes[-1], voltage)
        circuit.set_ground(nodes[-1])
        
        # Добавляем резисторы последовательно
        resistors = {}
        for i in range(num_resistors):
            resistance = random.randint(10, 1000)
            resistor_name = f"R{i+1}"
            circuit.add_resistor(nodes[i], nodes[i+1], resistance)
            resistors[resistor_name] = (nodes[i], nodes[i+1], resistance)
        
        # Выбираем случайный вопрос (простые типы для series)
        question_types = ["current", "voltage", "power", "total_current"]
        question_type = random.choice(question_types)
        
        target_resistor = random.choice(list(resistors.keys()))
        
        metadata = {
            "circuit_type": "series",
            "voltage_source": voltage,
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
        
        # Два основных узла A и B
        voltage = random.randint(5, 24)
        circuit.add_voltage_source("A", "B", voltage)
        circuit.set_ground("B")
        
        # Добавляем резисторы параллельно между A и B
        resistors = {}
        for i in range(num_resistors):
            resistance = random.randint(10, 1000)
            resistor_name = f"R{i+1}"
            circuit.add_resistor("A", "B", resistance)
            resistors[resistor_name] = ("A", "B", resistance)
        
        # Выбираем случайный вопрос (средние типы для parallel)
        question_types = ["current", "voltage", "power", "total_current", "equivalent_resistance"]
        question_type = random.choice(question_types)
        target_resistor = random.choice(list(resistors.keys()))
        
        metadata = {
            "circuit_type": "parallel",
            "voltage_source": voltage,
            "resistors": resistors,
            "question_type": question_type,
            "target_resistor": target_resistor,
            "nodes": ["A", "B"],
            "source_node": "A",
            "ground_node": "B"
        }
        
        return circuit, question_type, metadata
    
    def _generate_mixed(self, num_resistors: int) -> Tuple[Circuit, str, Dict]:
        """Генерирует смешанную цепь (последовательно-параллельную)"""
        circuit = Circuit()
        
        nodes = [chr(65 + i) for i in range(4)]  # A, B, C, D
        voltage = random.randint(5, 24)
        circuit.add_voltage_source(nodes[0], nodes[-1], voltage)
        circuit.set_ground(nodes[-1])
        
        resistors = {}
        resistor_count = 1
        
        # Последовательный резистор
        resistance = random.randint(10, 1000)
        circuit.add_resistor(nodes[0], nodes[1], resistance)
        resistors[f"R{resistor_count}"] = (nodes[0], nodes[1], resistance)
        resistor_count += 1
        
        # Параллельные резисторы
        parallel_count = num_resistors - 2
        for i in range(parallel_count):
            resistance = random.randint(10, 1000)
            circuit.add_resistor(nodes[1], nodes[2], resistance)
            resistors[f"R{resistor_count}"] = (nodes[1], nodes[2], resistance)
            resistor_count += 1
        
        # Еще один последовательный
        resistance = random.randint(10, 1000)
        circuit.add_resistor(nodes[2], nodes[3], resistance)
        resistors[f"R{resistor_count}"] = (nodes[2], nodes[3], resistance)
        
        # Выбираем случайный вопрос (средние типы для parallel)
        question_types = ["current", "voltage", "power", "total_current", "equivalent_resistance"]
        question_type = random.choice(question_types)
        target_resistor = random.choice(list(resistors.keys()))
        
        metadata = {
            "circuit_type": "mixed",
            "voltage_source": voltage,
            "resistors": resistors,
            "question_type": question_type,
            "target_resistor": target_resistor,
            "nodes": nodes,
            "source_node": nodes[0],
            "ground_node": nodes[-1]
        }
        
        return circuit, question_type, metadata
    
    def _generate_complex(self, num_resistors: int, difficulty: int = 1) -> Tuple[Circuit, str, Dict]:
        """Генерирует сложную многоконтурную цепь"""
        circuit = Circuit()
        
        # Создаем больше узлов для сложной топологии
        num_nodes = min(num_resistors + 1, 6)
        nodes = [chr(65 + i) for i in range(num_nodes)]
        
        voltage = random.randint(5, 24)
        circuit.add_voltage_source(nodes[0], nodes[-1], voltage)
        circuit.set_ground(nodes[-1])
        
        resistors = {}
        # Создаем связную цепь с гарантированной проводимостью
        added_resistors = 0

        # Шаг 1: Создаем базовую структуру
        if num_resistors >= 2:
            # Последовательная цепь от источника к земле
            for i in range(num_resistors - 1):
                resistance = random.randint(10, 1000)
                circuit.add_resistor(nodes[i], nodes[i+1], resistance)
                resistors[f"R{i+1}"] = (nodes[i], nodes[i+1], resistance)
                added_resistors += 1

            # Замыкаем цепь на землю
            resistance = random.randint(10, 1000)
            circuit.add_resistor(nodes[-2], nodes[-1], resistance)
            resistors[f"R{num_resistors}"] = (nodes[-2], nodes[-1], resistance)
            added_resistors += 1

        # Шаг 2: Добавляем дополнительные резисторы для сложности
        attempts = 0
        max_attempts = 50

        while added_resistors < num_resistors and attempts < max_attempts:
            n1, n2 = random.sample(nodes[:-1], 2)  # Исключаем землю
            if (n1, n2) not in circuit.resistors and (n2, n1) not in circuit.resistors:
                resistance = random.randint(10, 1000)
                circuit.add_resistor(n1, n2, resistance)
                resistors[f"R{added_resistors + 1}"] = (n1, n2, resistance)
                added_resistors += 1
            attempts += 1
        
        # Выбираем случайный вопрос (все типы для complex)
        question_types = [
            "current", "voltage", "power", "total_current",
            "equivalent_resistance", "voltage_divider", "current_divider", "power_total"
        ]
        question_type = random.choice(question_types)
        target_resistor = random.choice(list(resistors.keys())) if resistors else "R1"
        
        # Валидация корректности цепи
        if not self._validate_circuit(circuit, nodes):
            raise ValueError(f"Цепь сложности {difficulty} некорректна")

        metadata = {
            "circuit_type": "complex",
            "voltage_source": voltage,
            "resistors": resistors,
            "question_type": question_type,
            "target_resistor": target_resistor,
            "nodes": nodes,
            "source_node": nodes[0],
            "ground_node": nodes[-1]
        }

        return circuit, question_type, metadata

    def _validate_circuit(self, circuit, nodes):
        """Валидация корректности цепи"""
        try:
            from dc_circuit.solver import CircuitSolver
            solver = CircuitSolver()

            # 1. Проверяем, что есть резисторы
            if not circuit.resistors:
                return False

            # 2. Проверяем связность (есть путь от источника к земле)
            if not self._is_connected_circuit(circuit, nodes):
                return False

            # 3. Пробуем решить цепь
            voltages = solver.solve(circuit)
            if not voltages:
                return False

            # 4. Проверяем, что есть ток в цепи
            if not self._has_current_in_circuit(circuit, voltages):
                return False

            return True

        except Exception:
            return False

    def _is_connected_circuit(self, circuit, nodes):
        """Проверяет связность цепи"""
        visited = set()
        stack = [nodes[0]]

        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)

            # Добавляем соседей через резисторы
            for (n1, n2), _ in circuit.resistors.items():
                if n1 == node and n2 not in visited:
                    stack.append(n2)
                elif n2 == node and n1 not in visited:
                    stack.append(n1)

        return nodes[-1] in visited  # Земля должна быть достижима

    def _has_current_in_circuit(self, circuit, voltages):
        """Проверяет, что в цепи есть ток"""
        from dc_circuit.solver import CircuitSolver
        solver = CircuitSolver()

        # Проверяем ток хотя бы в одном резисторе
        for (n1, n2), resistance in circuit.resistors.items():
            if resistance > 0:  # Не нулевое сопротивление
                current = solver.get_current(circuit, voltages, n1, n2)
                if abs(current) > 1e-6:  # Есть заметный ток
                    return True
        return False