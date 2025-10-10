"""Модуль калькуляторов ответов для DC Circuit Analysis.

Содержит реализации Strategy pattern для вычисления различных типов ответов.
Каждый тип вопроса (current, voltage, power и т.д.) имеет свой калькулятор.
"""

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
        pass


class CurrentCalculator(AnswerCalculator):
    """Калькулятор тока через резистор."""
    
    def calculate(
        self,
        circuit: Circuit,
        node_voltages: Dict[str, float],
        metadata: Dict[str, Any],
        target_resistor: str
    ) -> Optional[float]:
        """Вычисляет ток через целевой резистор.
        
        Использует закон Ома: I = (V1 - V2) / R
        """
        try:
            resistor_info = metadata["resistors"][target_resistor]
            node1, node2, _ = resistor_info
            
            current = self.solver.get_current(circuit, node_voltages, node1, node2)
            return round(abs(current), self.precision)
        except Exception:
            return None


class VoltageCalculator(AnswerCalculator):
    """Калькулятор напряжения на резисторе."""
    
    def calculate(
        self,
        circuit: Circuit,
        node_voltages: Dict[str, float],
        metadata: Dict[str, Any],
        target_resistor: str
    ) -> Optional[float]:
        """Вычисляет напряжение на целевом резисторе.
        
        Использует закон Ома: V = I × R
        """
        try:
            resistor_info = metadata["resistors"][target_resistor]
            node1, node2, resistance = resistor_info
            
            current = self.solver.get_current(circuit, node_voltages, node1, node2)
            voltage = abs(current) * resistance
            return round(voltage, self.precision)
        except Exception:
            return None


class PowerCalculator(AnswerCalculator):
    """Калькулятор мощности резистора."""
    
    def calculate(
        self,
        circuit: Circuit,
        node_voltages: Dict[str, float],
        metadata: Dict[str, Any],
        target_resistor: str
    ) -> Optional[float]:
        """Вычисляет мощность, рассеиваемую резистором.
        
        Использует формулу: P = I²R
        """
        try:
            resistor_info = metadata["resistors"][target_resistor]
            node1, node2, resistance = resistor_info
            
            current = self.solver.get_current(circuit, node_voltages, node1, node2)
            power = (current ** 2) * resistance
            return round(power, self.precision)
        except Exception:
            return None


class TotalCurrentCalculator(AnswerCalculator):
    """Калькулятор общего тока от источника."""
    
    def calculate(
        self,
        circuit: Circuit,
        node_voltages: Dict[str, float],
        metadata: Dict[str, Any],
        target_resistor: str
    ) -> Optional[float]:
        """Вычисляет общий ток от источника напряжения.
        
        Вычисляет ток между узлом источника и землей.
        """
        try:
            source_node = metadata.get("source_node", "A")
            ground_node = metadata.get("ground_node", "C")
            
            total_current = self.solver.get_current(circuit, node_voltages, source_node, ground_node)
            return round(abs(total_current), self.precision)
        except Exception:
            return None


class EquivalentResistanceCalculator(AnswerCalculator):
    """Калькулятор эквивалентного сопротивления цепи."""
    
    def calculate(
        self,
        circuit: Circuit,
        node_voltages: Dict[str, float],
        metadata: Dict[str, Any],
        target_resistor: str
    ) -> Optional[float]:
        """Вычисляет эквивалентное сопротивление всей цепи.
        
        Использует формулу: R_eq = V / I_total
        """
        try:
            voltage_source = metadata.get("voltage_source", 10)
            source_node = metadata.get("source_node", "A")
            ground_node = metadata.get("ground_node", "C")
            
            total_current = self.solver.get_current(circuit, node_voltages, source_node, ground_node)
            
            if abs(total_current) > 1e-6:
                eq_resistance = voltage_source / abs(total_current)
                return round(eq_resistance, self.precision)
            return None
        except Exception:
            return None


class VoltageDividerCalculator(AnswerCalculator):
    """Калькулятор напряжения в делителе напряжения."""
    
    def calculate(
        self,
        circuit: Circuit,
        node_voltages: Dict[str, float],
        metadata: Dict[str, Any],
        target_resistor: str
    ) -> Optional[float]:
        """Вычисляет напряжение на резисторе в делителе напряжения.
        
        Использует формулу делителя: V_R = V_source × (R / R_total)
        """
        try:
            total_resistance = sum(r for _, _, r in metadata["resistors"].values())
            
            if total_resistance > 0:
                voltage_source = metadata.get("voltage_source", 10)
                resistor_info = metadata["resistors"][target_resistor]
                _, _, resistance = resistor_info
                
                voltage = voltage_source * resistance / total_resistance
                return round(voltage, self.precision)
            return None
        except Exception:
            return None


class CurrentDividerCalculator(AnswerCalculator):
    """Калькулятор тока в делителе тока."""
    
    def calculate(
        self,
        circuit: Circuit,
        node_voltages: Dict[str, float],
        metadata: Dict[str, Any],
        target_resistor: str
    ) -> Optional[float]:
        """Вычисляет ток через резистор в параллельном соединении.
        
        Использует формулу делителя тока: I_R = I_total × (G_R / G_total)
        где G - проводимость (1/R).
        """
        try:
            resistor_info = metadata["resistors"][target_resistor]
            node1, node2, resistance = resistor_info
            target_nodes = set([node1, node2])
            
            # Находим параллельные резисторы (с теми же узлами)
            parallel_resistors = []
            for r_name, (n1, n2, _) in metadata["resistors"].items():
                if r_name != target_resistor and set([n1, n2]) == target_nodes:
                    parallel_resistors.append((n1, n2, metadata["resistors"][r_name][2]))
            
            if parallel_resistors:
                total_conductance = sum(1.0 / r for _, _, r in parallel_resistors)
                target_conductance = 1.0 / resistance
                total_current = sum(
                    abs(self.solver.get_current(circuit, node_voltages, n1, n2))
                    for n1, n2, _ in parallel_resistors[:1]
                )
                
                if total_current > 0:
                    current = total_current * (target_conductance / (target_conductance + total_conductance))
                    return round(current, self.precision)
            return None
        except Exception:
            return None


class TotalPowerCalculator(AnswerCalculator):
    """Калькулятор общей мощности всех резисторов."""
    
    def calculate(
        self,
        circuit: Circuit,
        node_voltages: Dict[str, float],
        metadata: Dict[str, Any],
        target_resistor: str
    ) -> Optional[float]:
        """Вычисляет общую мощность, рассеиваемую всеми резисторами.
        
        Суммирует P = I²R для всех резисторов в цепи.
        """
        try:
            total_power = 0.0
            for _, (n1, n2, r_val) in metadata["resistors"].items():
                current = abs(self.solver.get_current(circuit, node_voltages, n1, n2))
                total_power += (current ** 2) * r_val
            
            return round(total_power, self.precision)
        except Exception:
            return None


def get_calculator_registry(solver: CircuitSolver, precision: int = 3) -> Dict[str, AnswerCalculator]:
    """Создает реестр калькуляторов для всех типов вопросов.
    
    Args:
        solver: Решатель цепей
        precision: Количество знаков после запятой
    
    Returns:
        Словарь {тип_вопроса: калькулятор}
    """
    return {
        "current": CurrentCalculator(solver, precision),
        "voltage": VoltageCalculator(solver, precision),
        "power": PowerCalculator(solver, precision),
        "total_current": TotalCurrentCalculator(solver, precision),
        "equivalent_resistance": EquivalentResistanceCalculator(solver, precision),
        "voltage_divider": VoltageDividerCalculator(solver, precision),
        "current_divider": CurrentDividerCalculator(solver, precision),
        "power_total": TotalPowerCalculator(solver, precision),
    }
