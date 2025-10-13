"""Тесты для калькуляторов."""

import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dc_circuit.calculators import (
    CurrentCalculator,
    VoltageCalculator,
    EquivalentResistanceCalculator
)
from dc_circuit.solver import CircuitSolver, Circuit


class TestCalculators:
    """Тесты для калькуляторов."""
    
    def setup_method(self):
        """Настройка перед каждым тестом."""
        self.solver = CircuitSolver()
        self.circuit = Circuit()
        self.node_voltages = {"A": 10.0, "B": 5.0, "C": 0.0}
        self.metadata = {
            "resistors": {
                "R1": ("A", "B", 100.0),
                "R2": ("B", "C", 200.0)
            },
            "voltage_sources": {
                ("A", "C"): 10.0
            }
        }
    
    def test_current_calculator(self):
        """Тест калькулятора тока."""
        calc = CurrentCalculator(self.solver)
        
        # Тест с правильными данными
        result = calc.calculate(self.circuit, self.node_voltages, self.metadata, "R1")
        assert result is not None
        assert isinstance(result, float)
        assert result > 0
        
        # Тест с несуществующим резистором
        result = calc.calculate(self.circuit, self.node_voltages, self.metadata, "R999")
        assert result is None
    
    def test_voltage_calculator(self):
        """Тест калькулятора напряжения."""
        calc = VoltageCalculator(self.solver)
        
        # Тест с правильными данными
        result = calc.calculate(self.circuit, self.node_voltages, self.metadata, "R1")
        assert result is not None
        assert isinstance(result, float)
        assert result > 0
        
        # Тест с несуществующим резистором
        result = calc.calculate(self.circuit, self.node_voltages, self.metadata, "R999")
        assert result is None
    
    
    def test_equivalent_resistance_calculator(self):
        """Тест калькулятора эквивалентного сопротивления."""
        calc = EquivalentResistanceCalculator(self.solver)
        
        # Тест с правильными данными
        result = calc.calculate(self.circuit, self.node_voltages, self.metadata, "R1")
        assert result is not None
        assert isinstance(result, float)
        assert result > 0
    
    def test_precision(self):
        """Тест точности вычислений."""
        calc = CurrentCalculator(self.solver, precision=2)
        
        result = calc.calculate(self.circuit, self.node_voltages, self.metadata, "R1")
        if result is not None:
            # Проверяем, что результат округлен до 2 знаков
            assert len(str(result).split('.')[-1]) <= 2
