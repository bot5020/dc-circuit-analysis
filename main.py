"""
Скрипт для демонстрации DC Circuit Analysis Environment

Основные функции:
- Генерация задач
- Показ промпта и ответа
- Простая демонстрация
"""

import os
import sys
from pathlib import Path

# Добавляем текущую папку в путь
sys.path.append(str(Path(__file__).parent))

from dc_circuit.game import DCCircuitGame
from config import CircuitConfig, VerifierConfig
from base.utils import get_system_prompt

# Константы
SAMPLES_PER_DIFFICULTY = 1
DIFFICULTIES = [1, 2]


def demo_generation():
    """Демонстрация генерации задач с промптом и ответом.
    
    Генерирует задачи для разных уровней сложности и выводит:
    - Промпт задачи
    - Правильный ответ
    - Метаданные (тип цепи, тип вопроса)
    """
    # Создаем игру с конфигурацией
    circuit_config = CircuitConfig()
    verifier_config = VerifierConfig()
    game = DCCircuitGame(circuit_config, verifier_config)

    # Генерируем примеры разной сложности
    for difficulty in DIFFICULTIES:
        print(f"\n--- Сложность {difficulty} ---")

        data_list = game.generate(num_of_questions=SAMPLES_PER_DIFFICULTY, difficulty=difficulty)

        if data_list:
            data = data_list[0]
            
            # Показываем полный промпт с системным сообщением
            print(f"📝 Полный промпт для модели:")
            print("=" * 60)
            print("SYSTEM MESSAGE:")
            print(get_system_prompt())
            print("\nUSER MESSAGE:")
            print(data.question)
            print("=" * 60)
            
            print(f"\n✅ Правильный ответ: {data.answer}")
            print(f"📊 Тип цепи: {data.metadata.get('circuit_type', 'unknown')}")
            print(f"📊 Тип вопроса: {data.metadata.get('question_type', 'unknown')}")
            print("-" * 50)
        else:
            print("❌ Ошибка генерации")



def main():
    """Главная функция для демонстрации DC Circuit Analysis.
    
    Запускает демонстрацию генерации задач для упрощенной системы
    с двумя уровнями сложности (1-2).
    """
    try:
        print("🚀 Запуск демонстрации DC Circuit Analysis")
        print("=" * 50)
        
        # Демонстрация генерации задач
        demo_generation()

        print("\n✅ Демонстрация завершена успешно!")
    except ImportError as e:
        print(f"\n❌ Ошибка импорта: {e}")
        print("   Проверьте, что все зависимости установлены")
    except Exception as e:
        print(f"\n❌ Неожиданная ошибка: {e}")
        print("   Проверьте конфигурацию системы")

if __name__ == "__main__":
    main()