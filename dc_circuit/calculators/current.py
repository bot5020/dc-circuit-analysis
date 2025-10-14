"""Калькулятор тока через резистор."""

from typing import Dict, Optional, Any
from .base import AnswerCalculator
from dc_circuit.solver import Circuit


class CurrentCalculator(AnswerCalculator):
    """Вычисляет ток через заданный резистор."""
    
    def calculate(
        self,
        circuit: Circuit,
        node_voltages: Dict[str, float],
        metadata: Dict[str, Any],
        target_resistor: str
    ) -> Optional[float]:
        """Вычисляет ток через резистор по закону Ома: I = (V1 - V2) / R.
        
        Args:
            circuit: Объект Circuit с цепью
            node_voltages: Словарь узловых потенциалов
            metadata: Метаданные цепи
            target_resistor: Название резистора (например, "R1")
        
        Returns:
            Ток через резистор в амперах или None если не удалось вычислить
        """
        try:
            # Получаем информацию о резисторе
            resistors = metadata.get("resistors", {})
            if target_resistor not in resistors:
                return None
            
            node1, node2, resistance = resistors[target_resistor]
            
            # Получаем напряжения на узлах
            v1 = node_voltages.get(node1, 0.0)
            v2 = node_voltages.get(node2, 0.0)
            
            # Вычисляем ток по закону Ома
            current = (v1 - v2) / resistance
            
            return self._round_result(abs(current))
            
        except (KeyError, ZeroDivisionError, TypeError):
            return None
