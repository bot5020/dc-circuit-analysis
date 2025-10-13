"""Реальные тесты вычислений для проверки физической корректности."""

import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dc_circuit.calculators import CurrentCalculator, VoltageCalculator
from dc_circuit.solver import CircuitSolver, Circuit


class TestRealCalculations:
    """Тесты реальных вычислений с проверкой физической корректности."""
    
    def setup_method(self):
        """Настройка с реальной цепью."""
        self.solver = CircuitSolver()
        self.circuit = Circuit()
        
        # Создаем простую последовательную цепь: V=12V, R1=4Ω, R2=6Ω
        self.circuit.add_voltage_source("A", "C", 12.0)
        self.circuit.add_resistor("A", "B", 4.0)
        self.circuit.add_resistor("B", "C", 6.0)
        self.circuit.set_ground("C")
        
        # Решаем цепь
        self.node_voltages = self.solver.solve(self.circuit)
        
        self.metadata = {
            "resistors": {
                "R1": ("A", "B", 4.0),
                "R2": ("B", "C", 6.0)
            },
            "voltage_sources": {
                ("A", "C"): 12.0
            }
        }
    
    def test_series_circuit_current(self):
        """Тест тока в последовательной цепи."""
        calc = CurrentCalculator(self.solver)
        
        # В последовательной цепи ток одинаковый через все резисторы
        current_r1 = calc.calculate(self.circuit, self.node_voltages, self.metadata, "R1")
        current_r2 = calc.calculate(self.circuit, self.node_voltages, self.metadata, "R2")
        
        # Ток должен быть одинаковым
        assert current_r1 is not None
        assert current_r2 is not None
        assert abs(current_r1 - current_r2) < 0.001  # Точность до 0.001A
        
        # Проверяем закон Ома: I = V_total / R_total = 12V / 10Ω = 1.2A
        expected_current = 12.0 / 10.0  # 1.2A
        assert abs(current_r1 - expected_current) < 0.001
    
    def test_series_circuit_voltage(self):
        """Тест напряжения в последовательной цепи."""
        calc = VoltageCalculator(self.solver)
        
        # Напряжение на R1: V_R1 = I * R1 = 1.2A * 4Ω = 4.8V
        voltage_r1 = calc.calculate(self.circuit, self.node_voltages, self.metadata, "R1")
        assert voltage_r1 is not None
        
        expected_voltage_r1 = 1.2 * 4.0  # 4.8V
        assert abs(voltage_r1 - expected_voltage_r1) < 0.001
        
        # Напряжение на R2: V_R2 = I * R2 = 1.2A * 6Ω = 7.2V
        voltage_r2 = calc.calculate(self.circuit, self.node_voltages, self.metadata, "R2")
        assert voltage_r2 is not None
        
        expected_voltage_r2 = 1.2 * 6.0  # 7.2V
        assert abs(voltage_r2 - expected_voltage_r2) < 0.001
        
        # Сумма напряжений должна равняться общему напряжению
        total_voltage = voltage_r1 + voltage_r2
        assert abs(total_voltage - 12.0) < 0.001
    
    
    def test_parallel_circuit(self):
        """Тест параллельной цепи."""
        # Создаем параллельную цепь: V=12V, R1=4Ω, R2=6Ω параллельно
        circuit = Circuit()
        circuit.add_voltage_source("A", "B", 12.0)
        circuit.add_resistor("A", "B", 4.0)
        circuit.add_resistor("A", "B", 6.0)
        circuit.set_ground("B")
        
        node_voltages = self.solver.solve(circuit)
        
        metadata = {
            "resistors": {
                "R1": ("A", "B", 4.0),
                "R2": ("A", "B", 6.0)
            },
            "voltage_sources": {
                ("A", "B"): 12.0
            }
        }
        
        # В параллельной цепи напряжение одинаковое на всех резисторах
        voltage_calc = VoltageCalculator(self.solver)
        voltage_r1 = voltage_calc.calculate(circuit, node_voltages, metadata, "R1")
        voltage_r2 = voltage_calc.calculate(circuit, node_voltages, metadata, "R2")
        
        assert voltage_r1 is not None
        assert voltage_r2 is not None
        assert abs(voltage_r1 - 12.0) < 0.001  # Напряжение источника
        assert abs(voltage_r2 - 12.0) < 0.001  # Напряжение источника
        
        # Токи должны быть разными: I1 = 12V/4Ω = 3A, I2 = 12V/6Ω = 2A
        current_calc = CurrentCalculator(self.solver)
        current_r1 = current_calc.calculate(circuit, node_voltages, metadata, "R1")
        current_r2 = current_calc.calculate(circuit, node_voltages, metadata, "R2")
        
        assert current_r1 is not None
        assert current_r2 is not None
        assert abs(current_r1 - 3.0) < 0.001  # 3A
        assert abs(current_r2 - 2.0) < 0.001  # 2A
    
    def test_edge_cases(self):
        """Тест граничных случаев."""
        calc = CurrentCalculator(self.solver)
        
        # Тест с нулевым сопротивлением (короткое замыкание)
        circuit = Circuit()
        circuit.add_voltage_source("A", "B", 12.0)
        circuit.add_resistor("A", "B", 0.001)  # Очень маленькое сопротивление
        circuit.set_ground("B")
        
        node_voltages = self.solver.solve(circuit)
        metadata = {
            "resistors": {"R1": ("A", "B", 0.001)},
            "voltage_sources": {("A", "B"): 12.0}
        }
        
        current = calc.calculate(circuit, node_voltages, metadata, "R1")
        assert current is not None
        assert current > 1000  # Очень большой ток при коротком замыкании
        
        # Тест с очень большим сопротивлением (обрыв)
        circuit = Circuit()
        circuit.add_voltage_source("A", "B", 12.0)
        circuit.add_resistor("A", "B", 1000000.0)  # Очень большое сопротивление
        circuit.set_ground("B")
        
        node_voltages = self.solver.solve(circuit)
        metadata = {
            "resistors": {"R1": ("A", "B", 1000000.0)},
            "voltage_sources": {("A", "B"): 12.0}
        }
        
        current = calc.calculate(circuit, node_voltages, metadata, "R1")
        assert current is not None
        assert current < 0.001  # Очень маленький ток при обрыве
