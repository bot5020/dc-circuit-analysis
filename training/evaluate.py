"""Оценка и визуализация результатов обучения модели.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np

from base.data import Data
from base.utils import get_system_prompt
from dc_circuit.game import DCCircuitGame


def evaluate_model(
    model_generate_func,
    test_data: List[Data],
    max_samples: int = 100
) -> Dict:
    """Оценивает качество модели на тестовых данных.
    
    Args:
        model_generate_func: Функция генерации ответов model(question) -> answer
        test_data: Список тестовых задач
        max_samples: Максимальное количество задач для оценки
    
    Returns:
        Словарь с метриками: {accuracy, correct, total}
    """
    game = DCCircuitGame()
    correct_count = 0
    total_count = min(len(test_data), max_samples)

    
    for i, data_item in enumerate(test_data[:total_count]):
        if i % 10 == 0 and i > 0:
            print(f"   Проверено {i}/{total_count}...")
        
        try:
            # Генерация ответа модели
            model_response = model_generate_func(data_item.question)
            
            # Проверка правильности через verifier
            if game.verify(data_item, model_response):
                correct_count += 1
                
        except Exception as e:
            print(f"⚠️  Ошибка на задаче {i}: {e}")
            continue
    
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "correct": correct_count,
        "total": total_count
    }


def plot_model_comparison(
    baseline_results: Dict[int, float],
    trained_results: Dict[int, float],
    save_path: str = "reports/model_comparison.png"
) -> None:
    """Создаёт парную барную диаграмму для сравнения моделей.
    
    Args:
        baseline_results: Результаты baseline модели {difficulty: accuracy}
        trained_results: Результаты обученной модели {difficulty: accuracy}
        save_path: Путь для сохранения графика
    """
    # Получение общих уровней сложности
    difficulties = sorted(set(baseline_results.keys()) & set(trained_results.keys()))
    
    if not difficulties:
        return
    
    # Подготовка данных
    baseline_accuracies = [baseline_results[d] for d in difficulties]
    trained_accuracies = [trained_results[d] for d in difficulties]
    
    # Создание графика с парными барами
    _, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(difficulties))
    width = 0.35  # Ширина бара
    
    # Создание парных баров
    bars1 = ax.bar(
        x - width/2, 
        baseline_accuracies, 
        width, 
        label='Baseline Model',
        color='skyblue',
        edgecolor='black'
    )
    bars2 = ax.bar(
        x + width/2, 
        trained_accuracies, 
        width, 
        label='Trained Model',
        color='lightcoral',
        edgecolor='black'
    )
    
    # Добавление меток значений на барах
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', 
                va='bottom', 
                fontsize=9
            )
    
    # Настройка осей и заголовка
    ax.set_xlabel('Уровень сложности', fontsize=12, fontweight='bold')
    ax.set_ylabel('Точность', fontsize=12, fontweight='bold')
    ax.set_title(
        'Сравнение моделей: Baseline vs Trained\n(DC Circuit Analysis Tasks)', 
        fontsize=14, 
        fontweight='bold', 
        pad=20
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f'Уровень {d}' for d in difficulties])
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(max(baseline_accuracies), max(trained_accuracies)) * 1.15)
    
    # Сохранение графика
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"График сохранён: {save_path}")


def generate_evaluation_report(
    baseline_results: Dict[int, float],
    trained_results: Dict[int, float],
    baseline_model: str = "baseline",
    trained_model: str = "trained",
    save_dir: str = "reports"
) -> None:
    """Генерирует полный отчёт оценки с визуализацией и JSON.
    
    Args:
        baseline_results: Результаты baseline модели
        trained_results: Результаты обученной модели
        baseline_model: Название baseline модели
        trained_model: Название обученной модели
        save_dir: Директория для сохранения отчётов
    """
    # Вычисление улучшений
    difficulties = sorted(set(baseline_results.keys()) & set(trained_results.keys()))
    improvements = {}
    
    for difficulty in difficulties:
        baseline_acc = baseline_results[difficulty]
        trained_acc = trained_results[difficulty]
        improvement = trained_acc - baseline_acc
        improvements[difficulty] = improvement
    
        print(f"Сложность {difficulty}: {baseline_acc:.3f} → {trained_acc:.3f} ({improvement:+.3f})")
    
    # Среднее улучшение
    if improvements:
        avg_improvement = sum(improvements.values()) / len(improvements)
        print(f"\nСреднее улучшение: {avg_improvement:+.3f}")
        
        if avg_improvement > 0:
            print("УСПЕХ: Модель улучшилась после обучения!")
    else:
        avg_improvement = 0.0
    
    # Создание визуализации
    plot_model_comparison(
        baseline_results, 
        trained_results, 
        save_path=f"{save_dir}/model_comparison.png"
    )
    
    # Сохранение результатов в JSON
    results_data = {
        "baseline_model": baseline_model,
        "trained_model": trained_model,
        "baseline_results": baseline_results,
        "trained_results": trained_results,
        "improvements": improvements,
        "avg_improvement": avg_improvement
    }
    
    os.makedirs(save_dir, exist_ok=True)
    json_path = f"{save_dir}/evaluation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)
    
    print(f"JSON результаты: {json_path}")


def load_model(model_path: str, max_seq_length: int = 3072):
    """Загружает обученную модель.
    
    Args:
        model_path: Путь к папке с моделью
        max_seq_length: Максимальная длина последовательности
    
    Returns:
        (model, tokenizer) - загруженная модель и токенизатор
    """
    print(f"📥 Загрузка модели из: {model_path}")
    
    from unsloth import FastLanguageModel
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        dtype=None,
    )
    
    # Включаем inference mode
    FastLanguageModel.for_inference(model)
    
    print("✅ Модель загружена!")
    return model, tokenizer


def create_model_generator(model, tokenizer, max_new_tokens: int = 512):
    """Создаёт функцию генерации для модели.
    
    Args:
        model: Загруженная модель
        tokenizer: Токенизатор
        max_new_tokens: Максимальное количество новых токенов
    
    Returns:
        Функция генерации: question -> answer
    """
    def generate(question: str) -> str:
        """Генерирует ответ на вопрос."""
        # Форматируем промпт
        messages = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": question}
        ]
        
        # Применяем chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Токенизация
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
        
        # Генерация
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,  # Более детерминированный для оценки
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Декодирование
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Убираем промпт из ответа
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        return response
    
    return generate


def main():
    """Основная функция оценки модели."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Оценка обученной модели на DC Circuit Analysis")
    parser.add_argument("--model_path", type=str, default="./dc_circuit_model_rl", help="Путь к обученной модели")
    parser.add_argument("--baseline_model", type=str, default="Qwen/Qwen3-4B-Instruct-2507", help="Базовая модель для сравнения")
    parser.add_argument("--samples_per_difficulty", type=int, default=20, help="Количество задач на уровень сложности")
    parser.add_argument("--difficulties", type=str, default="1,2,3,4,5", help="Уровни сложности через запятую")
    parser.add_argument("--output_dir", type=str, default="./evaluation_reports", help="Папка для отчётов")
    
    args = parser.parse_args()
    
    difficulties = [int(d) for d in args.difficulties.split(",")]
    
    print("=" * 80)
    print("🔬 ОЦЕНКА МОДЕЛИ DC CIRCUIT ANALYSIS")
    print("=" * 80)
    
    # Генерация тестовых данных
    print("\n📊 Генерация тестовых данных...")
    game = DCCircuitGame()
    test_data_by_difficulty = {}
    
    for difficulty in difficulties:
        print(f"  Сложность {difficulty}: генерация {args.samples_per_difficulty} задач...")
        data = game.generate(
            num_of_questions=args.samples_per_difficulty,
            difficulty=difficulty,
            max_attempts=50
        )
        test_data_by_difficulty[difficulty] = data
        print(f"    ✅ Сгенерировано {len(data)} задач")
    
    total_tasks = sum(len(data) for data in test_data_by_difficulty.values())
    print(f"📊 Всего тестовых задач: {total_tasks}")
    
    # Оценка обученной модели
    print(f"\n🎯 ОЦЕНКА ОБУЧЕННОЙ МОДЕЛИ: {args.model_path}")
    print("-" * 80)
    
    if os.path.exists(args.model_path):
        trained_model, trained_tokenizer = load_model(args.model_path)
        trained_generator = create_model_generator(trained_model, trained_tokenizer)
        
        trained_results = {}
        for difficulty, data in test_data_by_difficulty.items():
            print(f"\n  📝 Сложность {difficulty}:")
            result = evaluate_model(trained_generator, data, max_samples=len(data))
            trained_results[difficulty] = result["accuracy"]
            print(f"    ✅ Точность: {result['accuracy']:.3f} ({result['correct']}/{result['total']})")
        
        overall_accuracy = sum(trained_results.values()) / len(trained_results)
        print(f"\n  🎯 ОБЩАЯ ТОЧНОСТЬ: {overall_accuracy:.3f}")
    else:
        print(f"❌ Модель не найдена: {args.model_path}")
        trained_results = {d: 0.0 for d in difficulties}
    
    # Оценка baseline модели (опционально)
    print(f"\n📊 ОЦЕНКА BASELINE МОДЕЛИ: {args.baseline_model}")
    print("-" * 80)
    print("(Загрузка baseline модели...)")
    
    try:
        baseline_model, baseline_tokenizer = load_model(args.baseline_model)
        baseline_generator = create_model_generator(baseline_model, baseline_tokenizer)
        
        baseline_results = {}
        for difficulty, data in test_data_by_difficulty.items():
            print(f"\n  📝 Сложность {difficulty}:")
            result = evaluate_model(baseline_generator, data, max_samples=len(data))
            baseline_results[difficulty] = result["accuracy"]
            print(f"    ✅ Точность: {result['accuracy']:.3f} ({result['correct']}/{result['total']})")
        
        overall_baseline = sum(baseline_results.values()) / len(baseline_results)
        print(f"\n  🎯 ОБЩАЯ ТОЧНОСТЬ: {overall_baseline:.3f}")
        
    except Exception as e:
        print(f"⚠️  Ошибка оценки baseline: {e}")
        print("Пропускаем baseline оценку...")
        baseline_results = {d: 0.0 for d in difficulties}
    
    # Генерация отчёта
    print("\n" + "=" * 80)
    print("📈 СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 80)
    
    generate_evaluation_report(
        baseline_results=baseline_results,
        trained_results=trained_results,
        baseline_model=args.baseline_model,
        trained_model=args.model_path,
        save_dir=args.output_dir
    )
    
    print("\n" + "=" * 80)
    print("✅ ОЦЕНКА ЗАВЕРШЕНА!")
    print("=" * 80)
    print(f"📁 Отчёты сохранены в: {args.output_dir}")


if __name__ == "__main__":
    main()

