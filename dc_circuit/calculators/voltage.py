"""Калькулятор напряжения на резисторе."""

from typing import Dict, Optional, Any
from .base import AnswerCalculator
from dc_circuit.solver import CircuitSolver, Circuit


class VoltageCalculator(AnswerCalculator):
    """Вычисляет напряжение на заданном резисторе."""
    
    def calculate(
        self,
        circuit: Circuit,
        node_voltages: Dict[str, float],
        metadata: Dict[str, Any],
        target_resistor: str
    ) -> Optional[float]:
        """Вычисляет напряжение на резисторе: V = V1 - V2.
        
        Args:
            circuit: Объект Circuit с цепью
            node_voltages: Словарь узловых потенциалов
            metadata: Метаданные цепи
            target_resistor: Название резистора (например, "R1")
        
        Returns:
            Напряжение на резисторе в вольтах или None если не удалось вычислить
        """
        try:
            # Получаем информацию о резисторе
            resistors = metadata.get("resistors", {})
            if target_resistor not in resistors:
                return None
            
            node1, node2, _ = resistors[target_resistor]
            
            # Получаем напряжения на узлах
            v1 = node_voltages.get(node1, 0.0)
            v2 = node_voltages.get(node2, 0.0)
            
            # Вычисляем напряжение
            voltage = abs(v1 - v2)
            
            return self._round_result(voltage)
            
        except (KeyError, TypeError):
            return None
