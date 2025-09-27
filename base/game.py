from abc import ABC, abstractmethod
from typing import Optional
from base.verifier import Verifier
from base.data import Data
    

class Game(ABC):
    """
    Базовый класс для игры
    @param name: название игры
    @param verifier: класс верификатора
    """
    def __init__(self, name: str, verifier: Verifier):
        self.name = name
        self.verifier = verifier()

    @abstractmethod
    def generate(self, num_of_questions: int = 100, max_attempts: int = 100, difficulty: Optional[int] = 1, **kwargs):
        """
        Генерирует вопросы и ответы для игры
        @param num_of_questions: int
        @param max_attempts: int
        @param difficulty: int
        @return: список Data
        """
        raise NotImplementedError("Game.generate() не реализован")
    
    def verify(self, data: Data, test_solution: str):
        """
        Проверяет, соответствует ли тестовое решение ответу игровых данных
        @param data: Data
        @param test_solution: str
        @return: bool
        """
        return self.verifier.verify(data, test_solution)
    
    @abstractmethod
    def extract_answer(self, test_solution: str):
        """
        Извлекает ответ из тестового решения
        @param test_solution: str
        @return: str
        """
        raise NotImplementedError("Game.extract_answer() не реализован")