#!/usr/bin/env python3
"""
Главный скрипт для демонстрации работы DC Circuit Analysis Environment

Объединяет все основные функции:
- Демонстрация генерации задач
- Демонстрация верификации
- Полный пайплайн обучения
- Простое демо без зависимостей
- Тестирование модели
"""

import os
import sys
from pathlib import Path

# Добавляем текущую папку в путь
sys.path.append(str(Path(__file__).parent))

from dc_circuit.game import DCCircuitGame
from training.datasets import create_training_dataset, DCCircuitDataset
from training.evaluate import generate_full_report
from training.rl_trainer import DCCircuitRLTrainer, TrainingConfig
from config.models import get_model_name


def demo_circuit_generation():
    """Демонстрация генерации и решения задач"""
    print("=== ДЕМОНСТРАЦИЯ ГЕНЕРАЦИИ ЗАДАЧ ===\n")

    game = DCCircuitGame()

    # Генерируем примеры разной сложности
    for difficulty in [1, 3, 5]:
        print(f"--- Сложность {difficulty} ---")

        data_list = game.generate(num_of_questions=1, difficulty=difficulty)

        if data_list:
            data = data_list[0]
            print("Задача:")
            print(data.question)
            print(f"\nПравильный ответ: {data.answer}")
            print(f"Метаданные: {data.metadata['circuit_type']}, "
                  f"{data.metadata['question_type']}")
            print("\n" + "="*50 + "\n")


def demo_basic_generation():
    """Простая демонстрация генерации задач (из simple_demo.py)"""
    print("=== ПРОСТАЯ ДЕМОНСТРАЦИЯ ГЕНЕРАЦИИ ===\n")

    game = DCCircuitGame()

    # Генерируем одну простую задачу
    print("Генерируем задачу сложности 1...")
    tasks = game.generate(num_of_questions=1, difficulty=1)

    if tasks:
        task = tasks[0]
        print(f"Вопрос: {task.question}")
        print(f"Правильный ответ: {task.answer}")
        print(f"Сложность: {task.difficulty}")

        # Проверяем верификатор
        print("\nПроверим верификатор...")
        verifier = game.verifier

        # Тестируем с правильным ответом
        is_correct = verifier.verify(task, f"<answer>{task.answer}</answer>")
        print(f"Проверка правильного ответа: {'✅ ВЕРНО' if is_correct else '❌ НЕВЕРНО'}")

        # Тестируем с неправильным ответом
        is_correct = verifier.verify(task, "<answer>999.999</answer>")
        print(f"Проверка неправильного ответа: {'✅ ВЕРНО' if is_correct else '❌ НЕВЕРНО'}")

    else:
        print("❌ Ошибка генерации задач")


def demo_different_difficulties():
    """Демо задач разной сложности (из simple_demo.py)"""
    print("\n=== РАЗНЫЕ УРОВНИ СЛОЖНОСТИ ===\n")

    game = DCCircuitGame()

    for difficulty in [1, 3, 5]:
        print(f"--- Сложность {difficulty} ---")
        tasks = game.generate(num_of_questions=1, difficulty=difficulty)

        if tasks:
            task = tasks[0]
            print(f"Вопрос: {task.question[:100]}...")
            print(f"Правильный ответ: {task.answer}")
            print(f"Метаданные: {task.metadata.get('circuit_type', 'unknown')}")
        else:
            print("❌ Ошибка генерации")
        print()


def demo_verification():
    """Демонстрация верификации ответов"""
    print("=== ДЕМОНСТРАЦИЯ ВЕРИФИКАЦИИ ===\n")
    
    game = DCCircuitGame()
    data_list = game.generate(num_of_questions=1, difficulty=1)
    
    if data_list:
        data = data_list[0]
        print("Задача:")
        print(data.question)
        print(f"\nПравильный ответ: {data.answer}")
        
        # Тестируем разные форматы ответов
        test_cases = [
            f"The answer is {data.answer}",
            f"<answer>{data.answer}</answer>",
            f"After calculations: = {data.answer} A",
            "Wrong answer: 999.999",
            "<answer>not_a_number</answer>"
        ]
        
        print("\nТестирование верификации:")
        for i, test_answer in enumerate(test_cases, 1):
            is_correct = game.verify(data, test_answer)
            print(f"{i}. '{test_answer}' -> {'✓' if is_correct else '✗'}")


def test_model_simple():
    """Простой тест модели без vLLM (из simple_demo.py)"""
    print("=== ТЕСТИРОВАНИЕ МОДЕЛИ ===\n")

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_name = get_model_name("debug")
        print(f"Загружаем модель: {model_name}")

        # Простая загрузка без vLLM (используем CPU для стабильности)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="cpu"  # Принудительно CPU
        )

        # Генерируем задачу
        game = DCCircuitGame()
        tasks = game.generate(num_of_questions=1, difficulty=1)

        if tasks:
            task = tasks[0]
            prompt = f"Solve this circuit problem:\n{task.question}\n\nAnswer:"

            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(**inputs, max_length=inputs['input_ids'].shape[1] + 50, do_sample=True)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            print(f"Задача: {task.question}")
            print(f"Правильный ответ: {task.answer}")
            print(f"Ответ модели: {response[len(prompt):]}")

            # Проверяем
            verifier = game.verifier()
            is_correct = verifier.verify(task, response)
            print(f"Результат: {'✅ ВЕРНО' if is_correct else '❌ НЕВЕРНО'}")

        else:
            print("❌ Не удалось сгенерировать задачу")

    except Exception as e:
        print(f"❌ Ошибка тестирования модели: {e}")
        print("Попробуйте: pip install transformers torch")


def run_simple_training():
    """Простое обучение без лишних зависимостей (из simple_demo.py)"""
    print("=== ПРОСТОЕ ОБУЧЕНИЕ ===\n")

    try:
        # Создаем датасет
        print("1. Создаем тренировочный датасет...")
        train_data = create_training_dataset(
            difficulties=[1, 3],
            samples_per_difficulty=10
        )
        print(f"✅ Создано {len(train_data)} тренировочных примеров")

        # Сохраняем датасет
        import pickle
        os.makedirs("datasets", exist_ok=True)
        with open("datasets/simple_train.pkl", "wb") as f:
            pickle.dump(train_data, f)
        print("✅ Датасет сохранен в datasets/simple_train.pkl")

        print("\n2. Для обучения модели запустите:")
        print("   python training/grpo.py")

    except Exception as e:
        print(f"❌ Ошибка создания датасета: {e}")


def validate_system():
    """Проверка корректности работы всей системы"""
    print("=== ВАЛИДАЦИЯ СИСТЕМЫ ===\n")

    game = DCCircuitGame()

    # Тест 1: Генерация задач разной сложности
    print("1. Тестируем генерацию задач...")
    for difficulty in [1, 3, 5]:
        print(f"   Сложность {difficulty}:")
        tasks = game.generate(num_of_questions=3, difficulty=difficulty, max_attempts=20)

        if not tasks:
            print("      ❌ Не удалось сгенерировать задачи")
            continue

        print(f"      ✅ Сгенерировано {len(tasks)} задач")
        task = tasks[0]

        # Проверяем структуру задачи
        if not hasattr(task, 'question') or not task.question:
            print("      ❌ Задача без текста вопроса")
            continue
        if not hasattr(task, 'answer') or not task.answer:
            print("      ❌ Задача без ответа")
            continue
        if not hasattr(task, 'difficulty') or task.difficulty != difficulty:
            print("      ❌ Неправильная сложность")
            continue

        print(f"      📝 Вопрос: {task.question[:50]}...")
        print(f"      ✅ Ответ: {task.answer}")

    # Тест 2: Верификация ответов
    print("\n2. Тестируем верификацию...")
    test_task = game.generate(num_of_questions=1, difficulty=1, max_attempts=20)[0]

    test_cases = [
        (f"<answer>{test_task.answer}</answer>", "Правильный ответ в тегах"),
        (f"Answer: {test_task.answer}", "Правильный ответ с префиксом"),
        ("999.999", "Неправильный ответ"),
        ("не число", "Нечисловой ответ")
    ]

    for test_answer, description in test_cases:
        is_correct = game.verify(test_task, test_answer)
        status = "✅" if is_correct else "❌"
        print(f"   {status} {description}: '{test_answer}'")

    # Тест 3: Решение цепей
    print("\n3. Тестируем решение цепей...")
    try:
        # Создаем простую задачу
        simple_task = game.generate(num_of_questions=1, difficulty=1, max_attempts=20)[0]

        # Извлекаем метаданные
        metadata = simple_task.metadata
        if metadata and "resistors" in metadata:
            print(f"   ✅ Цепь содержит {len(metadata['resistors'])} резисторов")
            print(f"   📊 Тип цепи: {metadata.get('circuit_type', 'unknown')}")
            print(f"   ❓ Тип вопроса: {metadata.get('question_type', 'unknown')}")
        else:
            print("   ❌ Нет информации о резисторах")

    except Exception as e:
        print(f"   ❌ Ошибка при тестировании: {e}")

    # Тест 4: Создание датасета
    print("\n4. Тестируем создание датасета...")
    try:
        small_dataset = create_training_dataset(
            difficulties=[1, 2],
            samples_per_difficulty=5
        )
        print(f"   ✅ Создано {len(small_dataset)} примеров")

        # Проверяем структуру данных
        if small_dataset:
            sample = small_dataset[0]
            print(f"   📋 Пример: сложность {sample.difficulty}, ответ {sample.answer}")

    except Exception as e:
        print(f"   ❌ Ошибка создания датасета: {e}")

    print("\n✅ Валидация завершена!")


def run_full_pipeline():
    """Запускает полный пайплайн обучения и оценки"""
    print("=== ПОЛНЫЙ ПАЙПЛАЙН ===\n")

    # 1. Создание датасетов
    print("1. Создание обучающего датасета...")
    create_training_dataset(total_samples=5000, save_path="training_dataset.pkl")

    print("2. Создание тестовых датасетов...")
    DCCircuitDataset.create_test_datasets(
        difficulties=[1, 3, 5, 7, 9],
        samples_per_difficulty=100
    )

    # 2. RL обучение с GRPO
    print("\n3. RL обучение модели анализа цепей...")
    try:
        # Создаем конфигурацию для быстрого теста
        config = TrainingConfig(
            model_name=get_model_name('debug'),
            max_steps=10,  # Короткий тест
            per_device_train_batch_size=1,
            num_generations=4,  # Меньше генераций для теста
            lora_r=16,  # Меньше параметров для теста
            output_dir="./test_model"
        )

        # Создаем и запускаем тренер
        trainer = DCCircuitRLTrainer(config)
        trainer.setup_model_and_tokenizer()
        trainer.setup_trainer()
        trainer.train()

        print("✓ RL обучение завершено!")

    except Exception as e:
        print(f"⚠️ Ошибка RL обучения: {e}")
        print("Убедитесь, что установлены: pip install unsloth trl peft")

    # 3. Оценка моделей с LM Studio API
    print(f"\n4. Быстрая оценка с LM Studio API ({get_model_name('debug')})...")
    try:
        baseline_results, trained_results = generate_full_report(
            baseline_model=get_model_name('debug'),
            trained_model="./dc_circuit_model_grpo"
        )
        print("✓ Оценка завершена с LM Studio API!")
    except Exception as e:
        print(f"⚠️ Ошибка оценки: {e}")


def main():
    """Главная функция"""
    
    while True:
        print("\nВыберите действие:")
        print("1. Демонстрация генерации задач (детальная)")
        print("2. Демонстрация верификации ответов")
        print("3. Простая демонстрация генерации")
        print("4. Демонстрация разных уровней сложности")
        print("5. Тестирование модели (CPU)")
        print("6. Простое создание датасета")
        print("7. Валидация системы (проверка корректности)")
        print("8. Валидация системы")
        print("9. Полный пайплайн (обучение + оценка)")
        print("10. RL обучение модели анализа цепей")
        print("11. Оценка моделей")
        print("12. Выход")

        choice = input("\nВаш выбор (1-12): ").strip()

        if choice == "1":
            demo_circuit_generation()
        elif choice == "2":
            demo_verification()
        elif choice == "3":
            demo_basic_generation()
        elif choice == "4":
            demo_different_difficulties()
        elif choice == "5":
            test_model_simple()
        elif choice == "6":
            run_simple_training()
        elif choice == "7":
            validate_system()
        elif choice == "8":
            print("Валидация системы...")
            validate_system()
        elif choice == "9":
            run_full_pipeline()
        elif choice == "10":
            print("Запускаю RL обучение модели анализа цепей...")
            try:
                config = TrainingConfig(
                    model_name=get_model_name('debug'),
                    max_steps=50,
                    output_dir="./dc_circuit_model_rl"
                )
                trainer = DCCircuitRLTrainer(config)
                trainer.setup_model_and_tokenizer()
                trainer.setup_trainer()
                trainer.train()
                print("✅ RL обучение завершено!")
            except Exception as e:
                print(f"❌ Ошибка RL обучения: {e}")
                print("Убедитесь, что установлены: pip install unsloth trl peft")
        elif choice == "11":
            print("Запускаю оценку моделей...")
            try:
                baseline_results, trained_results = generate_full_report()
                print("✅ Оценка завершена!")
            except Exception as e:
                print(f"❌ Ошибка оценки: {e}")
        elif choice == "12":
            break
        else:
            print("Неверный выбор. Попробуйте снова.")


if __name__ == "__main__":
    main()