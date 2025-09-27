import re
import math
from base.verifier import Verifier
from base.data import Data
from dc_circuit.solver import CircuitSolver


class DCCircuitVerifier(Verifier):
    """Верификатор для задач анализа DC цепей"""
    
    def __init__(self):
        super().__init__()
        self.solver = CircuitSolver()
        self.rtol = 1e-3  # 0.1% относительная погрешность для LLM
        self.atol = 1e-6  # 1мк абсолютная погрешность
        self.precision = 3  # Количество знаков после запятой
    
    def verify(self, data: Data, test_answer: str) -> bool:
        """
        Проверяет правильность ответа агента
        @param data: данные задачи
        @param test_answer: ответ агента
        @return: True если ответ правильный
        """
        # Извлекаем численный ответ из текста агента
        extracted_answer = self.extract_answer(test_answer)
        if extracted_answer is None:
            return False

        try:
            agent_value = float(extracted_answer)
            correct_value = float(data.answer)

            # Проверка на NaN/inf
            if any([
                math.isnan(agent_value), math.isinf(agent_value),
                math.isnan(correct_value), math.isinf(correct_value)
            ]):
                return False

            # Округляем правильный ответ до 2 знаков после запятой
            rounded_correct = round(correct_value, self.precision)

            # Проверка с относительной и абсолютной погрешностью (используем округленный ответ)
            return abs(agent_value - rounded_correct) <= (self.atol + self.rtol * abs(rounded_correct))

        except (ValueError, TypeError):
            return False

    def get_accuracy_score(self, data: Data, test_answer: str) -> float:
        """
        Возвращает точность ответа агента (от 0 до 1)
        @param data: данные задачи
        @param test_answer: ответ агента
        @return: точность от 0 до 1
        """
        # Извлекаем численный ответ из текста агента
        extracted_answer = self.extract_answer(test_answer)
        if extracted_answer is None:
            return 0.0

        try:
            agent_value = float(extracted_answer)
            correct_value = float(data.answer)

            # Проверка на NaN/inf
            if any([
                math.isnan(agent_value), math.isinf(agent_value),
                math.isnan(correct_value), math.isinf(correct_value)
            ]):
                return 0.0

            # Округляем правильный ответ до 2 знаков после запятой
            rounded_correct = round(correct_value, self.precision)

            # Вычисляем относительную погрешность
            if abs(rounded_correct) < 1e-12:  # Избегаем деления на ноль
                relative_error = abs(agent_value - rounded_correct)
            else:
                relative_error = abs(agent_value - rounded_correct) / abs(rounded_correct)

            # Градиентная оценка на основе точности (более строгая)
            if relative_error <= 0.001:  # 0.1%
                return 1.0  # Максимальная оценка за очень высокую точность
            elif relative_error <= 0.002:  # 0.2%
                return 0.75
            elif relative_error <= 0.003:  # 0.3%
                return 0.5
            elif relative_error <= 0.005:  # 0.5%
                return 0.25
            else:
                return 0.0

        except (ValueError, TypeError):
            return 0.0
    
    def extract_answer(self, test_solution: str) -> str:
        """
        Извлекает численный ответ из решения агента
        @param test_solution: текст решения
        @return: численное значение как строка
        """
        # Чистим текст
        text = test_solution.strip()

        # 1. Ищем число в <answer> тегах (основной формат)
        tag_match = re.search(r'<answer>\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*</answer>', text)
        if tag_match:
            return tag_match.group(1)

        # 2. Ищем "= число" в конце (альтернативный формат)
        equals_match = re.search(r'=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(?:A|V|W|Ω)?\s*$', text, re.MULTILINE)
        if equals_match:
            return equals_match.group(1)

        # 3. Ищем первое число в тексте (fallback)
        numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text)
        if numbers:
            return numbers[0]

        return None