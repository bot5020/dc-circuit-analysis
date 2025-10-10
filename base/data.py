"""Модуль данных для игровых задач.

Содержит класс Data для представления задач с вопросами, ответами
и метаданными.
"""

import json
from typing import Optional, Dict, Any, List


class Data:
    """Класс данных для игры/корпуса.
    
    Представляет одну задачу с вопросом, ответом, уровнем сложности
    и опциональными метаданными.
    
    Attributes:
        question: Текст вопроса (промпт)
        answer: Правильный ответ
        difficulty: Уровень сложности от 1 до 10
        metadata: Дополнительные метаданные (параметры цепи и т.д.)
        gpt_response: Ответ модели (заполняется при инференсе)
    """
    
    def __init__(
        self, 
        question: str, 
        answer: str, 
        difficulty: int = 1, 
        metadata: Optional[Dict[str, Any]] = None, 
        **kwargs: Any
    ) -> None:
        self.question = question
        self.answer = answer
        self.difficulty = difficulty
        self.metadata = metadata
        self.gpt_response = ""
        
    def to_json(self) -> Dict[str, Any]:
        """Преобразует объект в словарь.
        
        Returns:
            Словарь с полями question, answer, difficulty, metadata, gpt_response
        """
        return {
            "question": self.question,
            "answer": self.answer,
            "difficulty": self.difficulty,
            "metadata": self.metadata,
            "gpt_response": self.gpt_response
        }
    
    def to_json_str(self) -> str:
        """Преобразует объект в JSON строку.
        
        Returns:
            JSON строка с данными объекта
        """
        return json.dumps(self.to_json(), ensure_ascii=False)
    
    @classmethod
    def from_json_str(cls, json_str: str) -> 'Data':
        """Создает объект Data из JSON строки.
        
        Args:
            json_str: JSON строка с данными
            
        Returns:
            Новый объект Data
        """
        json_data = json.loads(json_str)
        return cls(**json_data)
    
    @classmethod
    def from_json_dict(cls, json_dict: Dict[str, Any]) -> 'Data':
        """Создает объект Data из словаря.
        
        Args:
            json_dict: Словарь с данными
            
        Returns:
            Новый объект Data
        """
        instance = cls(**json_dict)
        if 'gpt_response' in json_dict:
            instance.gpt_response = json_dict['gpt_response']
        return instance
    
    @classmethod
    def from_jsonl_file(cls, file_path: str) -> List['Data']:
        """Загружает список объектов Data из JSONL файла.
        
        Args:
            file_path: Путь к JSONL файлу (каждая строка - JSON объект)
            
        Returns:
            Список объектов Data
        """
        data_list = []
        with open(file_path, "r") as f:
            for line in f:
                json_data = json.loads(line)
                instance = cls(**json_data)
                if 'gpt_response' in json_data:
                    instance.gpt_response = json_data['gpt_response']
                data_list.append(instance)
        return data_list