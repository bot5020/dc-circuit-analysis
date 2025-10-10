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