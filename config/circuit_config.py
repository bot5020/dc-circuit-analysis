"""Упрощенная конфигурация для генерации цепей согласно ТЗ"""

from dataclasses import dataclass
from typing import List


@dataclass
class CircuitConfig:
    """Упрощенная конфигурация для генерации электрических цепей."""
    
    # Сложность - только 2 уровня
    difficulties: List[int] = None
    max_attempts: int = 50
    
    # Простые параметры генерации
    voltage_range: tuple = (5, 24)
    resistance_range: tuple = (10, 100)
    
    def __post_init__(self):
        if self.difficulties is None:
            self.difficulties = [1, 2]  # Только 2 уровня сложности