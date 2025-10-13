"""Тесты для утилит."""

import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base.utils import extract_answer, get_system_prompt


class TestExtractAnswer:
    """Тесты для функции extract_answer."""
    
    def test_extract_from_tags(self):
        """Тест извлечения ответа из тегов."""
        test_cases = [
            ("<answer>1.234</answer>", "1.234"),
            ("<answer>0.500</answer>", "0.500"),
            ("<answer>999.999</answer>", "999.999"),
            ("<answer>0</answer>", "0"),
        ]
        
        for input_text, expected in test_cases:
            result = extract_answer(input_text)
            assert result == expected, f"Failed for input: {input_text}"
    
    def test_extract_from_equals(self):
        """Тест извлечения ответа после знака =."""
        test_cases = [
            ("Current = 1.234 A", "1.234"),
            ("Voltage = 5.678 V", "5.678"),
            ("Answer = 0.500", "0.500"),
            ("Result = 999.999", "999.999"),
        ]
        
        for input_text, expected in test_cases:
            result = extract_answer(input_text)
            assert result == expected, f"Failed for input: {input_text}"
    
    def test_extract_from_prefix(self):
        """Тест извлечения ответа с префиксом."""
        test_cases = [
            ("Answer: 1.234", "1.234"),
            ("Ответ: 5.678", "5.678"),
            ("Result: 0.500", "0.500"),
            ("Final answer: 999.999", "999.999"),
        ]
        
        for input_text, expected in test_cases:
            result = extract_answer(input_text)
            assert result == expected, f"Failed for input: {input_text}"
    
    def test_extract_last_number(self):
        """Тест извлечения последнего числа."""
        test_cases = [
            ("The current is 1.234 amperes", "1.234"),
            ("Voltage across R1 is 5.678 volts", "5.678"),
            ("Power dissipated is 0.500 watts", "0.500"),
            ("Total resistance equals 999.999 ohms", "999.999"),
        ]
        
        for input_text, expected in test_cases:
            result = extract_answer(input_text)
            assert result == expected, f"Failed for input: {input_text}"
    
    def test_no_number_found(self):
        """Тест случая, когда число не найдено."""
        test_cases = [
            "No number here",
            "<answer>not_a_number</answer>",
            "Current = not_a_number",
            "Answer: text_only",
            "",
        ]
        
        for input_text in test_cases:
            result = extract_answer(input_text)
            assert result == "", f"Expected empty string for input: {input_text}"
    
    def test_complex_text(self):
        """Тест извлечения из сложного текста."""
        complex_text = """
        <think>
        Step 1: Calculate the current through R1
        Step 2: Apply Ohm's law: I = V/R
        Step 3: I = 10V / 5Ω = 2.000A
        </think>
        <answer>2.000</answer>
        """
        
        result = extract_answer(complex_text)
        assert result == "2.000"
    
    def test_multiple_numbers(self):
        """Тест извлечения при наличии нескольких чисел."""
        test_cases = [
            ("R1 = 100Ω, R2 = 200Ω, Current = 1.234A", "1.234"),
            ("Voltage = 5.678V, Power = 2.345W", "2.345"),
            ("<answer>0.500</answer> and <answer>1.000</answer>", "0.500"),
        ]
        
        for input_text, expected in test_cases:
            result = extract_answer(input_text)
            assert result == expected, f"Failed for input: {input_text}"


class TestGetSystemPrompt:
    """Тесты для функции get_system_prompt."""
    
    def test_system_prompt_structure(self):
        """Тест структуры системного промпта."""
        prompt = get_system_prompt()
        
        # Проверяем, что промпт не пустой
        assert len(prompt) > 0
        assert isinstance(prompt, str)
        
        # Проверяем наличие ключевых секций
        assert "You are an expert electrical engineer" in prompt
        assert "FUNDAMENTAL LAWS" in prompt
        assert "Ohm's Law" in prompt
        assert "Kirchhoff's Current Law" in prompt
        assert "Kirchhoff's Voltage Law" in prompt
        assert "APPROACH" in prompt
        assert "RESPONSE FORMAT" in prompt
        assert "<think>" in prompt
        assert "<answer>" in prompt
        assert "EXAMPLE" in prompt
        assert "IMPORTANT RULES" in prompt
    
    def test_system_prompt_examples(self):
        """Тест наличия примеров в системном промпте."""
        prompt = get_system_prompt()
        
        # Проверяем наличие примеров
        assert "EXAMPLE 1" in prompt
        assert "EXAMPLE 2" in prompt
        assert "Series Circuit" in prompt
        assert "Parallel Circuit" in prompt
    
    def test_system_prompt_laws(self):
        """Тест наличия законов в системном промпте."""
        prompt = get_system_prompt()
        
        # Проверяем наличие основных законов
        assert "V = I × R" in prompt
        assert "I = V/R" in prompt
        assert "R = V/I" in prompt
        assert "ΣI_in = ΣI_out" in prompt
        assert "ΣV = 0" in prompt
        
    
    def test_system_prompt_format(self):
        """Тест формата ответа в системном промпте."""
        prompt = get_system_prompt()
        
        # Проверяем формат ответа
        assert "<think>" in prompt
        assert "</think>" in prompt
        assert "<answer>" in prompt
        assert "</answer>" in prompt
        assert "3 decimal places" in prompt
    
    def test_system_prompt_consistency(self):
        """Тест консистентности системного промпта."""
        prompt1 = get_system_prompt()
        prompt2 = get_system_prompt()
        
        # Проверяем, что промпт одинаковый при повторных вызовах
        assert prompt1 == prompt2
    
    def test_system_prompt_length(self):
        """Тест длины системного промпта."""
        prompt = get_system_prompt()
        
        # Проверяем, что промпт достаточно длинный (содержит детальную информацию)
        assert len(prompt) > 1000  # Минимальная длина для детального промпта
        assert len(prompt) < 10000  # Максимальная длина для разумного промпта
