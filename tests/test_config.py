"""Тесты для конфигураций."""

import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TrainingConfig, CircuitConfig


class TestTrainingConfig:
    """Тесты для конфигурации обучения."""
    
    def test_default_config(self):
        """Тест конфигурации по умолчанию."""
        config = TrainingConfig()
        
        # Проверяем основные параметры
        assert config.model_name == "Qwen/Qwen3-4B-Instruct-2507"
        assert config.output_dir == "./dc_circuit_model_rl"
        assert config.lora_r == 32
        assert config.lora_alpha == 32
        assert config.learning_rate == 1e-5
        assert config.max_steps == 500
        assert config.batch_size == 2
        assert config.max_seq_length == 8192
        assert config.gpu_memory_utilization == 0.25
    
    def test_difficulties_default(self):
        """Тест значений по умолчанию для сложности."""
        config = TrainingConfig()
        
        # Проверяем, что difficulties инициализируется правильно
        assert config.difficulties == [1, 2, 3]
    
    def test_lora_target_modules_default(self):
        """Тест значений по умолчанию для LoRA модулей."""
        config = TrainingConfig()
        
        expected_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        assert config.lora_target_modules == expected_modules
    
    def test_custom_config(self):
        """Тест кастомной конфигурации."""
        config = TrainingConfig(
            model_name="custom/model",
            lora_r=64,
            batch_size=4,
            max_steps=1000
        )
        
        assert config.model_name == "custom/model"
        assert config.lora_r == 64
        assert config.batch_size == 4
        assert config.max_steps == 1000
        
        # Проверяем, что остальные параметры остались по умолчанию
        assert config.learning_rate == 1e-5
        assert config.max_seq_length == 8192


class TestCircuitConfig:
    """Тесты для конфигурации генерации цепей."""
    
    def test_default_config(self):
        """Тест конфигурации по умолчанию."""
        config = CircuitConfig()
        
        # Проверяем основные параметры
        assert config.difficulties == [1, 2, 3]
        assert config.max_attempts == 50
        assert config.voltage_range == (5, 24)
        assert config.resistance_range == (10, 1000)
        assert config.topology_configs is not None
    
    def test_topology_configs_structure(self):
        """Тест структуры конфигураций топологий."""
        config = CircuitConfig()
        topology_configs = config.topology_configs
        
        # Проверяем наличие основных топологий
        assert 'series' in topology_configs
        assert 'parallel' in topology_configs
        assert 'mixed' in topology_configs
        # complex убран, остались только основные топологии
        
        # Проверяем структуру каждой топологии
        for topology, config_dict in topology_configs.items():
            assert 'min_resistors' in config_dict
            assert 'max_resistors' in config_dict
            assert 'question_types' in config_dict
            assert config_dict['min_resistors'] <= config_dict['max_resistors']
            assert isinstance(config_dict['question_types'], list)
            assert len(config_dict['question_types']) > 0
    
    def test_custom_config(self):
        """Тест кастомной конфигурации."""
        config = CircuitConfig(
            difficulties=[1, 2, 3],
            max_attempts=100,
            voltage_range=(10, 30),
            resistance_range=(50, 500)
        )
        
        assert config.difficulties == [1, 2, 3]
        assert config.max_attempts == 100
        assert config.voltage_range == (10, 30)
        assert config.resistance_range == (50, 500)
    
    def test_question_types_coverage(self):
        """Тест покрытия типов вопросов."""
        config = CircuitConfig()
        topology_configs = config.topology_configs
        
        all_question_types = set()
        for config_dict in topology_configs.values():
            all_question_types.update(config_dict['question_types'])
        
        # Проверяем, что есть основные типы вопросов
        expected_types = ['current', 'voltage', 'equivalent_resistance']
        for question_type in expected_types:
            assert question_type in all_question_types
    
    def test_resistance_range_validity(self):
        """Тест валидности диапазона сопротивлений."""
        config = CircuitConfig()
        
        min_res, max_res = config.resistance_range
        assert min_res > 0
        assert max_res > min_res
        assert min_res < 1000  # Разумный минимум
        assert max_res > 100   # Разумный максимум
    
    def test_voltage_range_validity(self):
        """Тест валидности диапазона напряжений."""
        config = CircuitConfig()
        
        min_voltage, max_voltage = config.voltage_range
        assert min_voltage > 0
        assert max_voltage > min_voltage
        assert min_voltage < 50  # Разумный минимум
        assert max_voltage > 10  # Разумный максимум
