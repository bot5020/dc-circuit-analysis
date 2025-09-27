from typing import Optional, List

from base.game import Game
from base.data import Data
from dc_circuit.generator import CircuitGenerator
from dc_circuit.solver import CircuitSolver
from dc_circuit.verifier import DCCircuitVerifier
from dc_circuit.prompt import create_circuit_prompt


class DCCircuitGame(Game):
    """Игра для анализа цепей постоянного тока"""
    
    def __init__(self):
        super().__init__("DC Circuit Analysis", DCCircuitVerifier)
        self.generator = CircuitGenerator()
        self.solver = CircuitSolver()
        self.answer_precision = 3  # Количество знаков после запятой в ответах
    
    def generate(self, num_of_questions: int = 100, max_attempts: int = 50,
                 difficulty: Optional[int] = 1, seed: Optional[int] = None, **kwargs) -> List[Data]:
        """
        Генерирует задачи анализа DC цепей
        @param num_of_questions: количество задач
        @param max_attempts: максимальное количество попыток генерации на задачу
        @param difficulty: уровень сложности 1-10
        @return: список объектов Data

        Попытки нужны потому что:
        1. Генератор может создать некорректную цепь (без резисторов)
        2. Solver может не решить систему уравнений (особенно для сложных цепей)
        3. Вычисление ответа может завершиться неудачно
        4. Могут возникнуть непредвиденные исключения
        """
        data_list = []
        
        for _ in range(num_of_questions):
            attempts = 0
            while attempts < max_attempts:
                try:
                    # Генерируем цепь
                    circuit, question_type, metadata = self.generator.generate_circuit(
                        difficulty=difficulty, seed=seed, **kwargs
                    )

                    # Проверяем, что цепь корректная
                    if not metadata.get("resistors"):
                        attempts += 1
                        continue

                    # Решаем цепь
                    node_voltages = self.solver.solve(circuit)
                    if not node_voltages:
                        attempts += 1
                        continue

                    # Вычисляем правильный ответ
                    target_resistor = metadata["target_resistor"]
                    correct_answer = self._calculate_answer(
                        circuit, node_voltages, metadata, question_type, target_resistor
                    )

                    if correct_answer is None:
                        attempts += 1
                        continue
                    

                    question_text = create_circuit_prompt(metadata, question_type, target_resistor)
                    
                    # Создаем объект данных
                    data = Data(
                        question=question_text,
                        answer=str(correct_answer),
                        difficulty=difficulty,
                        metadata=metadata
                    )
                    
                    data_list.append(data)
                    break
                    
                except Exception as e:
                    attempts += 1
                    if attempts >= max_attempts:
                        print(f"Не удалось сгенерировать задачу после {max_attempts} попыток: {e}")
        
        return data_list
    
    def _calculate_answer(self, circuit, node_voltages, metadata, question_type, target_resistor):
        """Вычисляет правильный ответ для заданного вопроса"""
        try:
            # Находим узлы целевого резистора
            resistor_info = metadata["resistors"][target_resistor]
            node1, node2, resistance = resistor_info
            
            # Вычисляем ток через резистор
            current = self.solver.get_current(circuit, node_voltages, node1, node2)
            current = abs(current)  # Берем модуль тока
            
            if question_type == "current":
                return round(current, self.answer_precision)
            elif question_type == "voltage":
                voltage = current * resistance
                return round(voltage, self.answer_precision)
            elif question_type == "power":
                power = current * current * resistance
                return round(power, self.answer_precision)
            elif question_type == "total_current":
                # Общий ток от источника
                source_node = metadata.get("source_node", "A")
                ground_node = metadata.get("ground_node", "C")
                total_current = self.solver.get_current(circuit, node_voltages, source_node, ground_node)
                return round(abs(total_current), self.answer_precision)
            elif question_type == "equivalent_resistance":
                # Эквивалентное сопротивление всей цепи
                voltage_source = metadata.get("voltage_source", 10)
                total_current = self.solver.get_current(circuit, node_voltages,
                    metadata.get("source_node", "A"), metadata.get("ground_node", "C"))
                if abs(total_current) > 1e-6:
                    eq_resistance = voltage_source / abs(total_current)
                    return round(eq_resistance, self.answer_precision)
                return None
            elif question_type == "voltage_divider":
                # Напряжение на конкретном резисторе в делителе
                total_resistance = sum(r for _, _, r in metadata["resistors"].values())
                if total_resistance > 0:
                    voltage_source = metadata.get("voltage_source", 10)
                    resistor_info = metadata["resistors"][target_resistor]
                    _, _, resistance = resistor_info
                    voltage = voltage_source * resistance / total_resistance
                    return round(voltage, self.answer_precision)
                return None
            elif question_type == "current_divider":
                # Ток через конкретный резистор в параллельном соединении
                # Находим параллельные резисторы
                parallel_resistors = []
                target_info = metadata["resistors"][target_resistor]
                target_nodes = set(target_info[:2])

                for r_name, (n1, n2, _) in metadata["resistors"].items():
                    if r_name != target_resistor and set([n1, n2]) == target_nodes:
                        parallel_resistors.append((n1, n2, metadata["resistors"][r_name][2]))

                if parallel_resistors:
                    total_conductance = sum(1.0 / r for _, _, r in parallel_resistors)
                    target_conductance = 1.0 / resistance
                    total_current = sum(abs(self.solver.get_current(circuit, node_voltages, n1, n2))
                                      for n1, n2, _ in parallel_resistors[:1])  # Примерный расчет

                    if total_current > 0:
                        current = total_current * (target_conductance / (target_conductance + total_conductance))
                        return round(current, self.answer_precision)
                return None
            elif question_type == "power_total":
                # Общая мощность всех резисторов
                total_power = 0
                for r_name, (n1, n2, r_val) in metadata["resistors"].items():
                    current = abs(self.solver.get_current(circuit, node_voltages, n1, n2))
                    total_power += current * current * r_val
                return round(total_power, self.answer_precision)
            else:
                return round(current, self.answer_precision)
                
        except Exception:
            return None
    
    def extract_answer(self, test_solution: str) -> str:
        """Извлекает ответ из решения агента"""
        return self.verifier.extract_answer(test_solution)