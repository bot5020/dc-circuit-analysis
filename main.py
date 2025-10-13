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


def demo_generation():
    """Демонстрация генерации задач с промптом и ответом."""
    # Создаем игру с конфигурацией
    circuit_config = CircuitConfig()
    verifier_config = VerifierConfig()
    game = DCCircuitGame(circuit_config, verifier_config)

    # Генерируем примеры разной сложности
    for difficulty in [1, 3, 5]:
        print(f"\n--- Сложность {difficulty} ---")

        data_list = game.generate(num_of_questions=1, difficulty=difficulty)

        if data_list:
            data = data_list[0]
            print(f"📝 Промпт:\n{data.question}\n")
            print(f"✅ Правильный ответ: {data.answer}")
            print(f"📊 Тип цепи: {data.metadata.get('circuit_type', 'unknown')}")
            print(f"📊 Тип вопроса: {data.metadata.get('question_type', 'unknown')}")
            print("-" * 50)
        else:
            print("❌ Ошибка генерации")

def main():
    """Главная функция."""
    try:
        demo_generation()
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")

if __name__ == "__main__":
    main()