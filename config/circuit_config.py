"""Упрощенная конфигурация для генерации цепей согласно"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class CircuitConfig:
    """Упрощенная конфигурация для генерации электрических цепей."""
    
    # Сложность
    difficulties: List[int] = None
    max_attempts: int = 50
    
    # Параметры генерации
    voltage_range: tuple = (5, 24)
    resistance_range: tuple = (10, 1000)
    
    # Топологии
    topology_configs: Dict[str, Dict] = None
    
    def __post_init__(self):
        if self.difficulties is None:
            self.difficulties = [1, 2, 3]  # Только 3 уровня сложности
        
        if self.topology_configs is None:
            self.topology_configs = {
                "series": {
                    "min_resistors": 2,
                    "max_resistors": 4,
                    "question_types": ["current", "voltage", "equivalent_resistance"]
                },
                "parallel": {
                    "min_resistors": 2,
                    "max_resistors": 5,
                    "question_types": ["current", "voltage", "equivalent_resistance"]
                },
                "mixed": {
                    "min_resistors": 3,
                    "max_resistors": 6,
                    "question_types": ["current", "voltage", "equivalent_resistance"]
                }
            }