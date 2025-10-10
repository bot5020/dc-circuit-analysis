"""Базовый модуль для верификации решений.

Этот модуль содержит абстрактный класс Verifier, который определяет интерфейс
для проверки правильности решений. 
"""

from abc import ABC, abstractmethod
from typing import Optional
from base.data import Data


class Verifier(ABC):
    """Класс для верификатора решений
    
    Определяет интерфейс для проверки правильности ответов и извлечения
    финальных ответов из решений. Все верификаторы должны наследоваться
    от этого класса.
    """
    
    def __init__(self) -> None:
        """Инициализирует верификатор"""
        pass
    
    @abstractmethod
    def verify(self, data: Data, test_answer: str) -> bool:
        """Проверяет правильность ответа
        
        Метод должен извлечь ответ из test_answer с помощью extract_answer()
        и сравнить его с правильным ответом из data.answer. Логика сравнения
        может быть сложнее простого ==, например, с учетом погрешности.
        
        Args:
            data: Объект Data с вопросом и правильным ответом
            test_answer: Ответ для проверки (может содержать рассуждения, теги)
        
        Returns:
            True если ответ правильный, False иначе
            
        Raises:
            NotImplementedError: Если метод не реализован в подклассе
            
        Note:
            Рекомендуется использовать extract_answer() для извлечения
            финального ответа перед сравнением.
        """
        raise NotImplementedError("Verifier.verify() не реализован")

    @abstractmethod
    def extract_answer(self, test_solution: str) -> Optional[str]:
        """Извлекает финальный ответ из решения.
        
        Решение может содержать цепочку рассуждений, теги, дополнительный текст.
        Этот метод должен извлечь только финальный ответ для сравнения.
        
        Args:
            test_solution: Полное решение (может включать <think>, <answer> теги)
        
        Returns:
            Извлеченный ответ как строка или None если не удалось извлечь
            
        Raises:
            NotImplementedError: Если метод не реализован в подклассе
            
        Example:
            >>> solution = "<think>Reasoning...</think><answer>1.234</answer>"
            >>> verifier.extract_answer(solution)
            '1.234'
        """
        raise NotImplementedError("Verifier.extract_answer() не реализован")