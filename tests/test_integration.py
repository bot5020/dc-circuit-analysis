"""Интеграционные тесты для проверки работы всей системы."""

import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dc_circuit.game import DCCircuitGame
from config import CircuitConfig


class TestIntegration:
    """Интеграционные тесты всей системы."""
    
    def setup_method(self):
        """Настройка для интеграционных тестов."""
        self.config = CircuitConfig()
        self.game = DCCircuitGame(self.config)
    
    def test_full_pipeline_series_circuit(self):
        """Тест полного пайплайна для последовательной цепи."""
        # Генерируем задачу
        data_list = self.game.generate(num_of_questions=1, difficulty=1)
        assert len(data_list) == 1
        
        data = data_list[0]
        
        # Проверяем, что это действительно последовательная цепь
        assert data.metadata['circuit_type'] == 'series'
        
        # Проверяем, что вопрос имеет смысл
        question = data.question
        assert 'circuit' in question.lower()
        assert 'voltage' in question.lower() or 'current' in question.lower() or 'power' in question.lower()
        
        # Проверяем, что ответ числовой
        answer = float(data.answer)
        assert isinstance(answer, float)
        assert answer > 0  # Физически разумный ответ
        
        # Проверяем верификацию правильного ответа
        correct_response = f"<answer>{data.answer}</answer>"
        is_correct = self.game.verify(data, correct_response)
        assert is_correct == True, "Правильный ответ должен быть верифицирован как правильный"
        
        # Проверяем верификацию неправильного ответа
        wrong_response = "<answer>999.999</answer>"
        is_wrong = self.game.verify(data, wrong_response)
        assert is_wrong == False, "Неправильный ответ должен быть верифицирован как неправильный"
    
    def test_full_pipeline_parallel_circuit(self):
        """Тест полного пайплайна для параллельной цепи."""
        # Генерируем задачу
        data_list = self.game.generate(num_of_questions=1, difficulty=3)
        assert len(data_list) == 1
        
        data = data_list[0]
        
        # Проверяем, что это действительно параллельная цепь
        assert data.metadata['circuit_type'] == 'mixed'  # Уровень 3 = mixed
        
        # Проверяем физическую корректность
        answer = float(data.answer)
        assert answer > 0
        
        # Проверяем верификацию
        correct_response = f"<answer>{data.answer}</answer>"
        is_correct = self.game.verify(data, correct_response)
        assert is_correct == True
    
    def test_different_question_types(self):
        """Тест разных типов вопросов."""
        question_types = ['current', 'voltage', 'power', 'total_current']
        
        for _ in range(10):  # Генерируем несколько задач
            data_list = self.game.generate(num_of_questions=1, difficulty=2)
            if data_list:
                data = data_list[0]
                question_type = data.metadata.get('question_type')
                
                if question_type in question_types:
                    # Проверяем, что ответ физически разумен
                    answer = float(data.answer)
                    assert answer > 0
                    
                    # Проверяем верификацию
                    correct_response = f"<answer>{data.answer}</answer>"
                    is_correct = self.game.verify(data, correct_response)
                    assert is_correct == True
    
    def test_difficulty_progression(self):
        """Тест прогрессии сложности."""
        difficulties = [1, 2, 3, 4, 5]
        
        for difficulty in difficulties:
            data_list = self.game.generate(num_of_questions=1, difficulty=difficulty)
            if data_list:
                data = data_list[0]
                
                # Проверяем, что сложность соответствует
                assert data.difficulty == difficulty
                
                # Проверяем, что ответ физически разумен
                answer = float(data.answer)
                assert answer > 0
                
                # Проверяем верификацию
                correct_response = f"<answer>{data.answer}</answer>"
                is_correct = self.game.verify(data, correct_response)
                assert is_correct == True
    
    def test_metadata_consistency(self):
        """Тест консистентности метаданных."""
        data_list = self.game.generate(num_of_questions=5, difficulty=2)
        
        for data in data_list:
            metadata = data.metadata
            
            # Проверяем обязательные поля
            required_fields = ['circuit_type', 'question_type', 'target_resistor', 'resistors', 'voltage_sources']
            for field in required_fields:
                assert field in metadata, f"Отсутствует поле {field}"
            
            # Проверяем, что target_resistor существует в resistors
            target_resistor = metadata['target_resistor']
            assert target_resistor in metadata['resistors'], f"target_resistor {target_resistor} не найден в resistors"
            
            # Проверяем, что resistors не пустой
            assert len(metadata['resistors']) > 0, "resistors не должен быть пустым"
            
            # Проверяем, что voltage_sources не пустой
            assert len(metadata['voltage_sources']) > 0, "voltage_sources не должен быть пустым"
    
    def test_answer_precision(self):
        """Тест точности ответов."""
        data_list = self.game.generate(num_of_questions=10, difficulty=1)
        
        for data in data_list:
            answer = data.answer
            
            # Проверяем, что ответ имеет правильный формат
            assert '.' in answer, "Ответ должен содержать десятичную точку"
            
            # Проверяем количество знаков после запятой (может быть 1-3 знака)
            decimal_part = answer.split('.')[1]
            assert 1 <= len(decimal_part) <= 3, f"Ответ должен иметь 1-3 знака после запятой, получено {len(decimal_part)}"
            
            # Проверяем, что ответ можно преобразовать в float
            float_answer = float(answer)
            assert isinstance(float_answer, float)
            assert float_answer > 0
    
    def test_error_handling(self):
        """Тест обработки ошибок."""
        # Тест с несуществующим резистором
        data_list = self.game.generate(num_of_questions=1, difficulty=1)
        if data_list:
            data = data_list[0]
            
            # Тест с пустым ответом
            empty_response = ""
            is_correct = self.game.verify(data, empty_response)
            assert is_correct == False
            
            # Тест с нечисловым ответом
            non_numeric_response = "<answer>not_a_number</answer>"
            is_correct = self.game.verify(data, non_numeric_response)
            assert is_correct == False
            
            # Тест с отрицательным ответом (физически невозможным)
            negative_response = "<answer>-1.000</answer>"
            is_correct = self.game.verify(data, negative_response)
            assert is_correct == False
