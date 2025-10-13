"""Тесты физической корректности вычислений."""

import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dc_circuit.calculators import CurrentCalculator, VoltageCalculator
from dc_circuit.solver import CircuitSolver, Circuit


class TestPhysics:
    """Тесты физической корректности."""
    
    def setup_method(self):
        """Настройка для физических тестов."""
        self.solver = CircuitSolver()
    
    def test_kirchhoff_current_law(self):
        """Тест закона Кирхгофа для токов (KCL)."""
        # Создаем простую параллельную цепь для проверки KCL
        circuit = Circuit()
        circuit.add_voltage_source("A", "B", 12.0)
        circuit.add_resistor("A", "B", 4.0)  # I1
        circuit.add_resistor("A", "B", 6.0)  # I2
        circuit.add_resistor("A", "B", 12.0) # I3
        circuit.set_ground("B")
        
        node_voltages = self.solver.solve(circuit)
        
        metadata = {
            "resistors": {
                "R1": ("A", "B", 4.0),
                "R2": ("A", "B", 6.0),
                "R3": ("A", "B", 12.0)
            },
            "voltage_sources": {("A", "B"): 12.0}
        }
        
        calc = CurrentCalculator(self.solver)
        
        # Токи через резисторы
        i1 = calc.calculate(circuit, node_voltages, metadata, "R1")
        i2 = calc.calculate(circuit, node_voltages, metadata, "R2")
        i3 = calc.calculate(circuit, node_voltages, metadata, "R3")
        
        # В параллельной цепи: I1 = 12V/4Ω = 3A, I2 = 12V/6Ω = 2A, I3 = 12V/12Ω = 1A
        assert abs(i1 - 3.0) < 0.001, f"Ток через R1 должен быть 3A, получен {i1}"
        assert abs(i2 - 2.0) < 0.001, f"Ток через R2 должен быть 2A, получен {i2}"
        assert abs(i3 - 1.0) < 0.001, f"Ток через R3 должен быть 1A, получен {i3}"
        
        # KCL: I1 + I2 + I3 = I_total (ток от источника)
        total_current = i1 + i2 + i3
        expected_total = 3.0 + 2.0 + 1.0  # 6A
        
        assert abs(total_current - expected_total) < 0.001, f"Общий ток должен быть 6A, получен {total_current}"
    
    def test_kirchhoff_voltage_law(self):
        """Тест закона Кирхгофа для напряжений (KVL)."""
        # Создаем контур с несколькими элементами
        circuit = Circuit()
        circuit.add_voltage_source("A", "D", 12.0)
        circuit.add_resistor("A", "B", 2.0)
        circuit.add_resistor("B", "C", 3.0)
        circuit.add_resistor("C", "D", 1.0)
        circuit.set_ground("D")
        
        node_voltages = self.solver.solve(circuit)
        
        metadata = {
            "resistors": {
                "R1": ("A", "B", 2.0),
                "R2": ("B", "C", 3.0),
                "R3": ("C", "D", 1.0)
            },
            "voltage_sources": {("A", "D"): 12.0}
        }
        
        calc = VoltageCalculator(self.solver)
        
        # Напряжения на резисторах
        v1 = calc.calculate(circuit, node_voltages, metadata, "R1")
        v2 = calc.calculate(circuit, node_voltages, metadata, "R2")
        v3 = calc.calculate(circuit, node_voltages, metadata, "R3")
        
        # KVL: V_source = V1 + V2 + V3
        total_voltage = v1 + v2 + v3
        assert abs(total_voltage - 12.0) < 0.001
    
    def test_ohm_law(self):
        """Тест закона Ома."""
        circuit = Circuit()
        circuit.add_voltage_source("A", "B", 10.0)
        circuit.add_resistor("A", "B", 5.0)
        circuit.set_ground("B")
        
        node_voltages = self.solver.solve(circuit)
        
        metadata = {
            "resistors": {"R1": ("A", "B", 5.0)},
            "voltage_sources": {("A", "B"): 10.0}
        }
        
        current_calc = CurrentCalculator(self.solver)
        voltage_calc = VoltageCalculator(self.solver)
        
        # V = I * R
        current = current_calc.calculate(circuit, node_voltages, metadata, "R1")
        voltage = voltage_calc.calculate(circuit, node_voltages, metadata, "R1")
        
        # Проверяем закон Ома: V = I * R
        expected_voltage = current * 5.0
        assert abs(voltage - expected_voltage) < 0.001
        
        # Проверяем закон Ома: I = V / R
        expected_current = voltage / 5.0
        assert abs(current - expected_current) < 0.001
    
    
    def test_series_resistance(self):
        """Тест эквивалентного сопротивления последовательной цепи."""
        circuit = Circuit()
        circuit.add_voltage_source("A", "C", 12.0)
        circuit.add_resistor("A", "B", 3.0)
        circuit.add_resistor("B", "C", 2.0)
        circuit.set_ground("C")
        
        node_voltages = self.solver.solve(circuit)
        
        metadata = {
            "resistors": {
                "R1": ("A", "B", 3.0),
                "R2": ("B", "C", 2.0)
            },
            "voltage_sources": {("A", "C"): 12.0}
        }
        
        current_calc = CurrentCalculator(self.solver)
        
        # В последовательной цепи ток одинаковый
        current = current_calc.calculate(circuit, node_voltages, metadata, "R1")
        
        # Эквивалентное сопротивление: R_eq = R1 + R2 = 3 + 2 = 5Ω
        # Ток: I = V / R_eq = 12V / 5Ω = 2.4A
        expected_current = 12.0 / 5.0
        assert abs(current - expected_current) < 0.001
    
    def test_parallel_resistance(self):
        """Тест эквивалентного сопротивления параллельной цепи."""
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
            "voltage_sources": {("A", "B"): 12.0}
        }
        
        current_calc = CurrentCalculator(self.solver)
        
        # Токи через резисторы
        i1 = current_calc.calculate(circuit, node_voltages, metadata, "R1")
        i2 = current_calc.calculate(circuit, node_voltages, metadata, "R2")
        
        # В параллельной цепи: I1 = V/R1 = 12V/4Ω = 3A, I2 = V/R2 = 12V/6Ω = 2A
        assert abs(i1 - 3.0) < 0.001
        assert abs(i2 - 2.0) < 0.001
        
        # Общий ток: I_total = I1 + I2 = 3A + 2A = 5A
        total_current = i1 + i2
        assert abs(total_current - 5.0) < 0.001
        
        # Эквивалентное сопротивление: R_eq = V / I_total = 12V / 5A = 2.4Ω
        # Или: 1/R_eq = 1/R1 + 1/R2 = 1/4 + 1/6 = 5/12, R_eq = 12/5 = 2.4Ω
        expected_req = 1.0 / (1.0/4.0 + 1.0/6.0)
        actual_req = 12.0 / total_current
        assert abs(actual_req - expected_req) < 0.001
