"""Модуль верификации решений для DC Circuit Analysis.

Содержит DCCircuitVerifier для проверки правильности ответов на задачи
по анализу электрических цепей с учётом допустимых погрешностей.
"""

import math
from typing import Optional
from base.verifier import Verifier
from base.data import Data
from base.utils import extract_answer
from config import VerifierConfig


class DCCircuitVerifier(Verifier):
    """Верификатор для задач анализа DC цепей.
    
    Проверяет правильность численных ответов с учётом относительной и
    абсолютной погрешности. Поддерживает градиентную оценку точности.
    
    Attributes:
        rtol: Относительная погрешность
        atol: Абсолютная погрешность
        precision: Количество знаков после запятой в ответах
    """
    
    def __init__(self, config: VerifierConfig = None) -> None:
        super().__init__()
        self.config = config or VerifierConfig()
        self.rtol: float = self.config.relative_tolerance
        self.atol: float = self.config.absolute_tolerance
        self.precision: int = self.config.answer_precision
    
    def verify(self, data: Data, test_answer: str) -> bool:
        """Проверяет правильность ответа агента.
        
        Использует комбинированную проверку с абсолютной и относительной погрешностью.
        Формула: |agent - correct| <= atol + rtol * |correct|
        
        Args:
            data: Данные задачи с правильным ответом
            test_answer: Ответ агента (может содержать теги, рассуждения)
        
        Returns:
            True если ответ правильный в пределах допустимой погрешности
        """
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

            # Округление значений до заданной точности
            rounded_correct = round(correct_value, self.precision)
            rounded_agent = round(agent_value, self.precision)

            # Комбинированная проверка абсолютной и относительной погрешности
            return abs(rounded_agent - rounded_correct) <= (self.atol + self.rtol * abs(rounded_correct))

        except (ValueError, TypeError):
            return False
    
    def get_accuracy_score(self, data: Data, test_answer: str) -> float:
        """
        Возвращает точность ответа агента (от 0 до 1)

        Args:
            data: Данные задачи с правильным ответом
            test_answer: Ответ агента (может содержать теги, рассуждения)
        
        Returns:
            Точность от 0 до 1
        """
        # Извлечение численного ответа из текста агента
        extracted_answer = self.extract_answer(test_answer)
        if extracted_answer is None:
            return 0.0

        try:
            agent_value = float(extracted_answer)
            correct_value = float(data.answer)

            # Проверка на NaN/inf в значениях
            if any([
                math.isnan(agent_value), math.isinf(agent_value),
                math.isnan(correct_value), math.isinf(correct_value)
            ]):
                return 0.0

            # Округление значений до одинаковой точности
            rounded_correct = round(correct_value, self.precision)
            rounded_agent = round(agent_value, self.precision)

            # Вычисление относительной погрешности
            if abs(rounded_correct) < self.config.min_divisor:  # Избегаем деления на ноль
                relative_error = abs(rounded_agent - rounded_correct)
            else:
                relative_error = abs(rounded_agent - rounded_correct) / abs(rounded_correct)

            # Градиентная оценка точности
            if relative_error <= self.config.threshold_perfect:
                return 1.0
            elif relative_error <= self.config.threshold_good:
                return 0.75
            elif relative_error <= self.config.threshold_ok:
                return 0.5
            elif relative_error <= self.config.threshold_fair:
                return 0.25
            else:
                return 0.0

        except (ValueError, TypeError):
            return 0.0
    
    def extract_answer(self, test_solution: str) -> Optional[str]:
        """Извлекает численный ответ из решения агента.
        
        Использует функцию из base.utils.
        
        Args:
            test_solution: Текст решения (может содержать теги, рассуждения)
        
        Returns:
            Извлечённый ответ как строку
        """
        return extract_answer(test_solution)