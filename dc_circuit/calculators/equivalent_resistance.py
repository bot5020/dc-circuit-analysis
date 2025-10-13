"""Калькулятор эквивалентного сопротивления."""

from typing import Dict, Optional, Any
from .base import AnswerCalculator
from dc_circuit.solver import CircuitSolver, Circuit


class EquivalentResistanceCalculator(AnswerCalculator):
    """Вычисляет эквивалентное сопротивление цепи."""
    
    def calculate(
        self,
        circuit: Circuit,
        node_voltages: Dict[str, float],
        metadata: Dict[str, Any],
        target_resistor: str
    ) -> Optional[float]:
        """Вычисляет эквивалентное сопротивление цепи.
        
        ИСПРАВЛЕНО: Теперь правильно вычисляет общий ток через источник,
        суммируя токи через все резисторы.
        
        Args:
            circuit: Объект Circuit с цепью
            node_voltages: Словарь узловых потенциалов
            metadata: Метаданные цепи
            target_resistor: Не используется (для совместимости)
        
        Returns:
            Эквивалентное сопротивление в омах или None если не удалось вычислить
        """
        try:
            # Получаем информацию об источнике
            voltage_sources = metadata.get("voltage_sources", {})
            if not voltage_sources:
                return None
            
            # Берем первый источник
            source_key = list(voltage_sources.keys())[0]
            source_voltage = voltage_sources[source_key]
            
            if source_voltage == 0:
                return None
            
            # Вычисляем общий ток через источник
            # Используем метод мощности: P_total = V_source × I_total
            # P_total = сумма мощностей на всех резисторах
            resistors = metadata.get("resistors", {})
            total_power = 0.0
            
            for resistor_name, (node1, node2, resistance) in resistors.items():
                if resistance == 0:
                    continue
                    
                # Получаем напряжения на узлах резистора
                v1 = node_voltages.get(node1, 0.0)
                v2 = node_voltages.get(node2, 0.0)
                voltage_across = abs(v1 - v2)
                
                # Мощность на резисторе: P = V² / R
                power = (voltage_across ** 2) / resistance
                total_power += power
            
            if total_power == 0:
                return None
            
            # Общий ток: I_total = P_total / V_source
            total_current = total_power / source_voltage
            
            # Эквивалентное сопротивление = напряжение источника / общий ток
            equivalent_resistance = source_voltage / total_current
            
            return self._round_result(equivalent_resistance)
            
        except (KeyError, ZeroDivisionError, TypeError):
            return None
