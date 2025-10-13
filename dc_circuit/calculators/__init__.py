"""Упрощенный модуль калькуляторов для анализа DC цепей.

Содержит только базовые калькуляторы:
- Ток через резистор
- Напряжение на резисторе
"""

from .base import AnswerCalculator
from .current import CurrentCalculator
from .voltage import VoltageCalculator


def get_calculator_registry(solver, precision=3):
    """Создает реестр калькуляторов для базовых типов вопросов.
    
    Args:
        solver: Экземпляр CircuitSolver для вычислений
        precision: Количество знаков после запятой
    
    Returns:
        Словарь {question_type: calculator_instance}
    """
    return {
        "current": CurrentCalculator(solver, precision),
        "voltage": VoltageCalculator(solver, precision),
    }


__all__ = [
    "AnswerCalculator",
    "CurrentCalculator", 
    "VoltageCalculator",
    "get_calculator_registry"
]