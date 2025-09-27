from abc import ABC, abstractmethod
from base.data import Data


class Verifier(ABC):
    """
    Базовый класс для верификатора
    """
    def __init__(self):
        pass
    
    @abstractmethod
    def verify(self, data: Data, test_answer: str):
        """ 
        Проверяет, соответствует ли тестовый ответ золотому ответу
        @param data: Data
        @param test_answer: str
        @return: bool
        """
        raise NotImplementedError("Verifier.verify() не реализован")

    @abstractmethod
    def extract_answer(self, test_solution: str):
        """
        Извлекает ответ из тестового решения
        @param test_solution: str
        @return: str
        """
        raise NotImplementedError("Verifier.extract_answer() не реализован")