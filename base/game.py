"""Базовый модуль для игровой логики.

Этот модуль содержит абстрактный класс Game, который определяет интерфейс
для всех игр.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Type, Any
from base.verifier import Verifier
from base.data import Data


class Game(ABC):
    """Базовый абстрактный класс для игры.
    
    Определяет интерфейс для генерации задач, верификации решений
    и извлечения ответов. Все игры должны наследоваться от этого класса.
    
    Attributes:
        name: Название игры
        verifier: Экземпляр верификатора для проверки ответов
    """
    
    def __init__(self, name: str, verifier: Type[Verifier]) -> None:
        """Инициализирует игру с названием и верификатором.
        
        Args:
            name: Название игры
            verifier: Класс верификатора (будет инстанцирован)
        """
        self.name: str = name
        self.verifier: Verifier = verifier()

    @abstractmethod
    def generate(
        self, 
        num_of_questions: int = 100, 
        max_attempts: int = 100, 
        difficulty: Optional[int] = 1, 
        **kwargs: Any
    ) -> List[Data]:
        """Генерирует вопросы и ответы для игры.
        
        Метод должен генерировать заданное количество задач определенной сложности.
        Поддерживает как передачу difficulty, так и прямую передачу гиперпараметров
        через **kwargs.
        
        Args:
            num_of_questions: Количество вопросов для генерации
            max_attempts: Максимальное количество попыток генерации одной задачи
                         (нужно для случайной генерации с валидацией)
            difficulty: Уровень сложности от 1 до 10 (опционально)
            **kwargs: Дополнительные гиперпараметры для прямой настройки сложности
                     (например, min_resistors=5, max_resistors=10)
        
        Returns:
            Список объектов Data с сгенерированными задачами
            
        Raises:
            NotImplementedError: Если метод не реализован в подклассе
            
        Example:
            # Через difficulty:
            tasks = game.generate(num_of_questions=10, difficulty=5)
            
            # Через прямые гиперпараметры:
            tasks = game.generate(
                num_of_questions=10,
                min_resistors=5,
                max_resistors=10
            )
        """
        raise NotImplementedError("Game.generate() не реализован")
    
    def verify(self, data: Data, test_solution: str) -> bool:
        """Проверяет правильность решения через верификатор.
        
        Args:
            data: Объект Data с вопросом и правильным ответом
            test_solution: Решение для проверки (может содержать рассуждения)
        
        Returns:
            True если решение правильное, False иначе
        """
        return self.verifier.verify(data, test_solution)
    
    @abstractmethod
    def extract_answer(self, test_solution: str) -> str:
        """Извлекает финальный ответ из решения.
        
        Решение может содержать цепочку рассуждений, этот метод должен
        извлечь только финальный ответ для сравнения.
        
        Args:
            test_solution: Полное решение (может включать теги, рассуждения)
        
        Returns:
            Извлеченный ответ как строка
            
        Raises:
            NotImplementedError: Если метод не реализован в подклассе
        """
        raise NotImplementedError("Game.extract_answer() не реализован")