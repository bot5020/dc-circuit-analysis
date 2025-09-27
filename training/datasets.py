import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import Dataset, IterableDataset
import pickle
from typing import List
from base.data import Data
from dc_circuit.game import DCCircuitGame


class DCCircuitDataset(Dataset):
    """Статический датасет для тестирования модели"""
    
    def __init__(self, data_list: List[Data]):
        self.data = data_list
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "question": item.question,
            "answer": item.answer,
            "difficulty": item.difficulty
        }
    
    @classmethod
    def create_test_datasets(cls, difficulties: List[int], samples_per_difficulty: int = 500):
        """
        Создает тестовые датасеты для каждого уровня сложности
        @param difficulties: список уровней сложности
        @param samples_per_difficulty: количество образцов на уровень
        @return: словарь {difficulty: dataset}
        """
        game = DCCircuitGame()
        datasets = {}
        
        for difficulty in difficulties:
            print(f"Генерация тестового датасета сложности {difficulty}...")
            data_list = game.generate(
                num_of_questions=samples_per_difficulty,
                difficulty=difficulty,
                max_attempts=50
            )
            datasets[difficulty] = cls(data_list)
            
            # Сохраняем датасет
            with open(f"test_dataset_difficulty_{difficulty}.pkl", "wb") as f:
                pickle.dump(data_list, f)
        
        return datasets


class DCCircuitIterableDataset(IterableDataset):
    """Итерируемый датасет для обучения"""
    
    def __init__(self, difficulties: List[int] = [1, 2, 3, 4, 5], 
                 samples_per_difficulty: int = 1000):
        self.game = DCCircuitGame()
        self.difficulties = difficulties
        self.samples_per_difficulty = samples_per_difficulty
        
    def __iter__(self):
        """Генерирует образцы в бесконечном цикле"""
        while True:
            # Генерируем данные для всех уровней сложности
            for difficulty in self.difficulties:
                data_list = self.game.generate(
                    num_of_questions=self.samples_per_difficulty,
                    difficulty=difficulty,
                    max_attempts=30
                )
                
                for data in data_list:
                    yield {
                        "question": data.question,
                        "answer": data.answer,
                        "difficulty": data.difficulty
                    }


def create_training_dataset(total_samples: int = 10000, save_path: str = "training_dataset.pkl",
                           difficulties: List[int] = None, samples_per_difficulty: int = None):
    """
    Создает большой статический датасет для обучения
    @param total_samples: общее количество образцов (если не указаны difficulties)
    @param save_path: путь для сохранения
    @param difficulties: список сложностей (новый параметр)
    @param samples_per_difficulty: образцов на сложность (новый параметр)
    @return: список Data объектов
    """
    game = DCCircuitGame()
    all_data = []

    # Определяем параметры
    if difficulties is not None and samples_per_difficulty is not None:
        # Новый способ вызова
        target_difficulties = difficulties
        target_samples = samples_per_difficulty
    else:
        # Старый способ вызова
        target_difficulties = list(range(1, 11))
        target_samples = total_samples // len(target_difficulties)

    for difficulty in target_difficulties:
        print(f"Генерация обучающих данных сложности {difficulty}...")
        data_list = game.generate(
            num_of_questions=target_samples,
            difficulty=difficulty,
            max_attempts=50
        )
        all_data.extend(data_list)

    # Сохраняем датасет
    with open(save_path, "wb") as f:
        pickle.dump(all_data, f)

    print(f"Сохранено {len(all_data)} образцов в {save_path}")
    return all_data


def load_dataset(path: str) -> List[Data]:
    """Загружает сохраненный датасет"""
    with open(path, "rb") as f:
        return pickle.load(f)