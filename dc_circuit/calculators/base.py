"""Базовый класс для калькуляторов ответов."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
from dc_circuit.solver import CircuitSolver, Circuit


class AnswerCalculator(ABC):
    """Базовый абстрактный класс для калькуляторов ответов.
    
    Определяет интерфейс для вычисления ответа на вопрос определенного типа.
    Все калькуляторы должны наследоваться от этого класса.
    
    Attributes:
        solver: Экземпляр CircuitSolver для вычисления токов и напряжений
        precision: Количество знаков после запятой в ответе
    """
    
    def __init__(self, solver: CircuitSolver, precision: int = 3):
        """Инициализирует калькулятор.
        
        Args:
            solver: Решатель цепей для вычислений
            precision: Количество знаков после запятой
        """
        self.solver = solver
        self.precision = precision
    
    @abstractmethod
    def calculate(
        self,
        circuit: Circuit,
        node_voltages: Dict[str, float],
        metadata: Dict[str, Any],
        target_resistor: str
    ) -> Optional[float]:
        """Вычисляет ответ для заданного типа вопроса.
        
        Args:
            circuit: Объект Circuit с цепью
            node_voltages: Словарь узловых потенциалов {node: voltage}
            metadata: Метаданные цепи (резисторы, источники и т.д.)
            target_resistor: Название целевого резистора (например, "R1")
        
        Returns:
            Вычисленное значение ответа или None если не удалось вычислить
        """
        raise NotImplementedError("AnswerCalculator.calculate() не реализован")
    
    def _round_result(self, value: float) -> float:
        """Округляет результат до заданной точности.
        
        Args:
            value: Значение для округления
        
        Returns:
            Округленное значение
        """
        return round(value, self.precision)
