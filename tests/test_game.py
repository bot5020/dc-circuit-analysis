"""Тесты для игры DCCircuitGame."""

import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dc_circuit.game import DCCircuitGame
from config import CircuitConfig, VerifierConfig


class TestDCCircuitGame:
    """Тесты для игры DCCircuitGame."""
    
    def setup_method(self):
        """Настройка перед каждым тестом."""
        self.circuit_config = CircuitConfig()
        self.verifier_config = VerifierConfig()
        self.game = DCCircuitGame(self.circuit_config, self.verifier_config)
    
    def test_game_initialization(self):
        """Тест инициализации игры."""
        assert self.game.name == "DC Circuit Analysis"
        assert hasattr(self.game, 'generator')
        assert hasattr(self.game, 'solver')
        assert hasattr(self.game, 'verifier')
        assert self.game.answer_precision == 3
    
    def test_generate_simple_task(self):
        """Тест генерации простой задачи."""
        data_list = self.game.generate(num_of_questions=1, difficulty=1)
        
        assert len(data_list) == 1
        data = data_list[0]
        
        assert hasattr(data, 'question')
        assert hasattr(data, 'answer')
        assert hasattr(data, 'difficulty')
        assert hasattr(data, 'metadata')
        
        assert data.difficulty == 1
        assert isinstance(data.question, str)
        assert len(data.question) > 0
        assert isinstance(data.answer, str)
        assert len(data.answer) > 0
    
    def test_generate_multiple_tasks(self):
        """Тест генерации нескольких задач."""
        data_list = self.game.generate(num_of_questions=3, difficulty=2)
        
        assert len(data_list) == 3
        
        for data in data_list:
            assert data.difficulty == 2
            assert isinstance(data.question, str)
            assert isinstance(data.answer, str)
    
    def test_different_difficulties(self):
        """Тест генерации задач разной сложности."""
        for difficulty in [1, 3, 5]:
            data_list = self.game.generate(num_of_questions=1, difficulty=difficulty)
            
            assert len(data_list) == 1
            assert data_list[0].difficulty == difficulty
    
    def test_metadata_structure(self):
        """Тест структуры метаданных."""
        data_list = self.game.generate(num_of_questions=1, difficulty=1)
        data = data_list[0]
        
        metadata = data.metadata
        assert isinstance(metadata, dict)
        
        # Проверяем наличие ключевых полей
        assert 'circuit_type' in metadata
        assert 'question_type' in metadata
        assert 'target_resistor' in metadata
        assert 'resistors' in metadata
        assert 'voltage_sources' in metadata
    
    def test_verify_correct_answer(self):
        """Тест верификации правильного ответа."""
        data_list = self.game.generate(num_of_questions=1, difficulty=1)
        data = data_list[0]
        
        # Тест с правильным ответом в тегах
        correct_answer = f"<answer>{data.answer}</answer>"
        assert self.game.verify(data, correct_answer) == True
        
        # Тест с правильным ответом без тегов
        assert self.game.verify(data, data.answer) == True
    
    def test_verify_incorrect_answer(self):
        """Тест верификации неправильного ответа."""
        data_list = self.game.generate(num_of_questions=1, difficulty=1)
        data = data_list[0]
        
        # Тест с неправильным ответом
        incorrect_answer = "<answer>999.999</answer>"
        assert self.game.verify(data, incorrect_answer) == False
        
        # Тест с нечисловым ответом
        non_numeric_answer = "<answer>not_a_number</answer>"
        assert self.game.verify(data, non_numeric_answer) == False
    
    def test_extract_answer(self):
        """Тест извлечения ответа."""
        # Тест с тегами
        answer_with_tags = "<answer>1.234</answer>"
        extracted = self.game.extract_answer(answer_with_tags)
        assert extracted == "1.234"
        
        # Тест с префиксом
        answer_with_prefix = "The answer is 2.567"
        extracted = self.game.extract_answer(answer_with_prefix)
        assert extracted == "2.567"
        
        # Тест с нечисловым ответом
        non_numeric = "<answer>not_a_number</answer>"
        extracted = self.game.extract_answer(non_numeric)
        assert extracted == ""
    
    def test_calculators_initialization(self):
        """Тест инициализации калькуляторов."""
        # Генерируем задачу, чтобы инициализировать калькуляторы
        data_list = self.game.generate(num_of_questions=1, difficulty=1)
        
        assert hasattr(self.game, '_calculators')
        assert self.game._calculators is not None
        assert len(self.game._calculators) > 0
        
        # Проверяем наличие основных типов калькуляторов
        expected_types = ['current', 'voltage', 'equivalent_resistance']
        for calc_type in expected_types:
            assert calc_type in self.game._calculators
