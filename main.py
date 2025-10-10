"""
Главный скрипт для демонстрации DC Circuit Analysis Environment

Основные функции:
- Демонстрация генерации задач
- Демонстрация верификации ответов
- Валидация системы
- Простые примеры для быстрого старта
"""

import os
import sys
from pathlib import Path

# Добавляем текущую папку в путь
sys.path.append(str(Path(__file__).parent))

from dc_circuit.game import DCCircuitGame


def demo_circuit_generation():
    """Демонстрация генерации и решения задач."""
    print("="*80)
    print("🔌 ДЕМОНСТРАЦИЯ ГЕНЕРАЦИИ ЗАДАЧ")
    print("="*80)

    game = DCCircuitGame()

    # Генерируем примеры разной сложности
    for difficulty in [1, 3, 5]:
        print(f"\n--- Сложность {difficulty} ---")

        data_list = game.generate(num_of_questions=1, difficulty=difficulty)

        if data_list:
            data = data_list[0]
            print(f"Задача: {data.question[:100]}...")
            print(f"Правильный ответ: {data.answer}")
            print(f"Метаданные: {data.metadata.get('circuit_type', 'unknown')}, "
                  f"{data.metadata.get('question_type', 'unknown')}")
        else:
            print("❌ Ошибка генерации")


def demo_verification():
    """Демонстрация верификации ответов."""
    print("\n" + "="*80)
    print("✅ ДЕМОНСТРАЦИЯ ВЕРИФИКАЦИИ")
    print("="*80)
    
    game = DCCircuitGame()
    data_list = game.generate(num_of_questions=1, difficulty=1)
    
    if data_list:
        data = data_list[0]
        print(f"\nЗадача: {data.question[:80]}...")
        print(f"Правильный ответ: {data.answer}")
        
        # Тестируем разные форматы ответов
        test_cases = [
            (f"<answer>{data.answer}</answer>", "Правильный в тегах"),
            (f"The answer is {data.answer}", "Правильный с префиксом"),
            ("999.999", "Неправильный"),
            ("<answer>not_a_number</answer>", "Нечисловой")
        ]
        
        print("\nТестирование верификации:")
        for test_answer, description in test_cases:
            is_correct = game.verify(data, test_answer)
            print(f"  {'✅' if is_correct else '❌'} {description}: '{test_answer}'")
    else:
        print("❌ Ошибка генерации задачи")


def validate_system():
    """Проверка корректности работы всей системы."""
    print("\n" + "="*80)
    print("🔍 ВАЛИДАЦИЯ СИСТЕМЫ")
    print("="*80)

    game = DCCircuitGame()

    # Тест 1: Генерация задач
    print("\n1️⃣ Тестируем генерацию задач...")
    for difficulty in [1, 3, 5]:
        tasks = game.generate(num_of_questions=3, difficulty=difficulty, max_attempts=20)
        if tasks:
            print(f"  ✅ Сложность {difficulty}: {len(tasks)} задач")
        else:
            print(f"  ❌ Сложность {difficulty}: не удалось сгенерировать")

    # Тест 2: Верификация
    print("\n2️⃣ Тестируем верификацию...")
    test_task = game.generate(num_of_questions=1, difficulty=1)[0]
    
    test_cases = [
        (f"<answer>{test_task.answer}</answer>", True, "Правильный"),
        ("999.999", False, "Неправильный"),
    ]
    
    for test_answer, expected, desc in test_cases:
        is_correct = game.verify(test_task, test_answer)
        status = "✅" if is_correct == expected else "❌"
        print(f"  {status} {desc}: {is_correct == expected}")

    # Тест 3: Калькуляторы
    print("\n3️⃣ Тестируем калькуляторы...")
    if hasattr(game, '_calculators') and game._calculators:
        print(f"  ✅ Калькуляторы инициализированы: {len(game._calculators)} типов")
        for calc_type in game._calculators.keys():
            print(f"     - {calc_type}")
    else:
        # Инициализируем калькуляторы
        _ = game.generate(1, difficulty=1)
        if hasattr(game, '_calculators'):
            print(f"  ✅ Калькуляторы инициализированы: {len(game._calculators)} типов")

    print("\n✅ Валидация завершена!")


def simple_demo():
    """Простая демонстрация"""
    print("\n" + "="*80)
    print("🚀 ПРОСТАЯ ДЕМОНСТРАЦИЯ")
    print("="*80)

    print("\n1. Создаём игру...")
    game = DCCircuitGame()

    print("2. Генерируем задачу...")
    tasks = game.generate(num_of_questions=1, difficulty=1)

    if tasks:
        task = tasks[0]
        print(f"\n📝 Вопрос:\n{task.question}\n")
        print(f"✅ Правильный ответ: {task.answer}")
        print(f"📊 Сложность: {task.difficulty}")

        # Проверяем верификатор
        print("\n3. Проверяем верификатор...")
        is_correct = game.verify(task, f"<answer>{task.answer}</answer>")
        print(f"   Проверка правильного ответа: {'✅ ВЕРНО' if is_correct else '❌ НЕВЕРНО'}")

        is_incorrect = game.verify(task, "<answer>999.999</answer>")
        print(f"   Проверка неправильного ответа: {'❌ НЕВЕРНО' if not is_incorrect else '✅ ВЕРНО'}")
    else:
        print("❌ Ошибка генерации задач")


def main():
    """Главная функция с меню."""
    
    while True:
        print("\n" + "="*80)
        print("="*80)
        print("\nВыберите действие:")
        print("  1. Простая демонстрация")
        print("  2. Демонстрация генерации задач")
        print("  3. Демонстрация верификации")
        print("  4. Валидация системы")
        print("  5. Выход")

        choice = input("\nВаш выбор (1-5): ").strip()

        if choice == "1":
            simple_demo()
        elif choice == "2":
            demo_circuit_generation()
        elif choice == "3":
            demo_verification()
        elif choice == "4":
            validate_system()
        elif choice == "5":
            print("\n👋 До свидания!")
            break
        else:
            print("❌ Неверный выбор. Попробуйте снова.")

if __name__ == "__main__":
    main()
