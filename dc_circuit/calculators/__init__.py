"""Упрощенный модуль калькуляторов для анализа DC цепей.

Содержит калькуляторы:
- Ток через резистор
- Напряжение на резисторе  
- Общее сопротивление цепи
"""

from .base import AnswerCalculator
from .current import CurrentCalculator
from .voltage import VoltageCalculator
from .equivalent_resistance import EquivalentResistanceCalculator


def get_calculator_registry(solver, precision=3):
    """Создает реестр калькуляторов для основных типов вопросов.
    
    Args:
        solver: Экземпляр CircuitSolver для вычислений
        precision: Количество знаков после запятой
    
    Returns:
        Словарь {question_type: calculator_instance}
    """
    return {
        "current": CurrentCalculator(solver, precision),
        "voltage": VoltageCalculator(solver, precision),
        "equivalent_resistance": EquivalentResistanceCalculator(solver, precision),
    }


__all__ = [
    "AnswerCalculator",
    "CurrentCalculator", 
    "VoltageCalculator",
    "EquivalentResistanceCalculator",
    "get_calculator_registry"
]