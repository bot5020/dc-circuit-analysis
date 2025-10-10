"""Модуль реализации игры анализа DC цепей.

Содержит класс DCCircuitGame для генерации задач по анализу электрических цепей
постоянного тока с использованием законов Ома и Кирхгофа.
"""

from typing import Optional, List, Dict, Any

from base.game import Game
from base.data import Data
from base.utils import extract_answer
from dc_circuit.generator import CircuitGenerator
from dc_circuit.solver import CircuitSolver, Circuit
from dc_circuit.verifier import DCCircuitVerifier
from dc_circuit.prompt import create_circuit_prompt


class DCCircuitGame(Game):
    """Игра для анализа цепей постоянного тока.
    
    Реализует генерацию задач по анализу DC цепей с различными типами вопросов:
    - Расчет тока через резистор
    - Расчет напряжения на резисторе
    - Расчет мощности
    - И другие типы вопросов
    
    Attributes:
        generator: Генератор электрических цепей
        solver: Решатель цепей (вычисление узловых потенциалов)
        answer_precision: Количество знаков после запятой в ответах
    """
    
    def __init__(self) -> None:
        """Инициализирует игру DC Circuit Analysis."""
        super().__init__("DC Circuit Analysis", DCCircuitVerifier)
        self.generator: CircuitGenerator = CircuitGenerator()
        self.solver: CircuitSolver = CircuitSolver()
        self.answer_precision: int = 3
    
    def generate(
        self, 
        num_of_questions: int = 100, 
        max_attempts: int = 50,
        difficulty: Optional[int] = 1, 
        seed: Optional[int] = None, 
        **kwargs: Any
    ) -> List[Data]:
        """Генерирует задачи анализа DC цепей.
        
        Метод генерирует заданное количество задач по анализу электрических цепей
        с использованием случайной генерации и валидации корректности.
        
        Args:
            num_of_questions: Количество задач для генерации
            max_attempts: Максимальное количество попыток генерации одной задачи
                         (нужно т.к. не все случайные цепи корректны)
            difficulty: Уровень сложности от 1 до 10 (влияет на количество резисторов,
                       топологию цепи)
            seed: Seed для воспроизводимости генерации (опционально)
            **kwargs: Дополнительные гиперпараметры для прямой настройки генерации
                     (например, min_resistors=5, max_resistors=10, topology="mixed")
        
        Returns:
            Список объектов Data с сгенерированными задачами
            
        Note:
            Попытки нужны потому что:
            1. Генератор может создать некорректную цепь (без резисторов)
            2. Solver может не решить систему уравнений (особенно для сложных цепей)
            3. Вычисление ответа может завершиться неудачно
            4. Могут возникнуть непредвиденные исключения
            
        Example:
            >>> game = DCCircuitGame()
            >>> tasks = game.generate(num_of_questions=10, difficulty=5)
            >>> print(len(tasks))  # 10
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
    
    def _calculate_answer(
        self, 
        circuit: Circuit, 
        node_voltages: Dict[str, float], 
        metadata: Dict[str, Any], 
        question_type: str, 
        target_resistor: str
    ) -> Optional[float]:
        """Вычисляет правильный ответ для заданного типа вопроса.
        
        Использует Strategy pattern с отдельными калькуляторами для каждого типа вопроса.
        Калькуляторы определены в модуле dc_circuit.calculators.
        
        Args:
            circuit: Объект Circuit с электрической цепью
            node_voltages: Словарь узловых потенциалов {node: voltage}
            metadata: Метаданные цепи (резисторы, источники напряжения и т.д.)
            question_type: Тип вопроса ("current", "voltage", "power" и т.д.)
            target_resistor: Название целевого резистора (например, "R1")
        
        Returns:
            Вычисленное значение ответа с заданной точностью или None при ошибке
            
        Note:
            Метод lazy-инициализирует реестр калькуляторов при первом вызове.
            Это позволяет избежать циклических импортов и ускоряет инициализацию.
        """

        # CHECK:
        # Lazy initialization калькуляторов для избежания циклических импортов
        if not hasattr(self, '_calculators'):
            from dc_circuit.calculators import get_calculator_registry
            self._calculators = get_calculator_registry(self.solver, self.answer_precision)
        
        # Получаем калькулятор для данного типа вопроса
        calculator = self._calculators.get(question_type)
        if calculator:
            return calculator.calculate(circuit, node_voltages, metadata, target_resistor)
        else: 
            raise ValueError(f"Unknown question type: {question_type}")
    
    def extract_answer(self, test_solution: str) -> str:
        """Извлекает ответ из решения агента.
        
        Использует унифицированную функцию из base.utils для консистентности
        извлечения ответов во всем проекте.
        
        Args:
            test_solution: Полное решение агента (может содержать теги, рассуждения)
        
        Returns:
            Извлеченный ответ как строка
        """
        return extract_answer(test_solution)