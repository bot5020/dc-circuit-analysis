"""Тесты для конфигурации верификатора."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import VerifierConfig


class TestVerifierConfig:
    """Тесты для VerifierConfig."""
    
    def test_default_config(self):
        """Тест конфигурации по умолчанию."""
        config = VerifierConfig()
        
        # Проверяем значения по умолчанию
        assert config.relative_tolerance == 1e-3
        assert config.absolute_tolerance == 1e-6
        assert config.answer_precision == 3
        assert config.threshold_perfect == 0.01
        assert config.threshold_good == 0.05
        assert config.threshold_ok == 0.10
        assert config.threshold_fair == 0.20
        assert config.min_divisor == 1e-12
        assert config.max_attempts == 3
        assert config.fallback_tolerance == 0.5
    
    def test_custom_config(self):
        """Тест кастомной конфигурации."""
        config = VerifierConfig(
            relative_tolerance=1e-4,
            absolute_tolerance=1e-7,
            answer_precision=4,
            threshold_perfect=0.005,
            threshold_good=0.02,
            threshold_ok=0.05,
            threshold_fair=0.15,
            min_divisor=1e-15,
            max_attempts=5,
            fallback_tolerance=0.3
        )
        
        assert config.relative_tolerance == 1e-4
        assert config.absolute_tolerance == 1e-7
        assert config.answer_precision == 4
        assert config.threshold_perfect == 0.005
        assert config.threshold_good == 0.02
        assert config.threshold_ok == 0.05
        assert config.threshold_fair == 0.15
        assert config.min_divisor == 1e-15
        assert config.max_attempts == 5
        assert config.fallback_tolerance == 0.3
    
    def test_config_validation(self):
        """Тест валидации конфигурации."""
        # Тест с разумными значениями
        config = VerifierConfig(
            relative_tolerance=0.001,
            absolute_tolerance=0.000001,
            answer_precision=2,
            threshold_perfect=0.01,
            threshold_good=0.05,
            threshold_ok=0.10,
            threshold_fair=0.20
        )
        
        # Проверяем, что значения сохраняются
        assert config.relative_tolerance == 0.001
        assert config.absolute_tolerance == 0.000001
        assert config.answer_precision == 2
        assert config.threshold_perfect == 0.01
        assert config.threshold_good == 0.05
        assert config.threshold_ok == 0.10
        assert config.threshold_fair == 0.20
