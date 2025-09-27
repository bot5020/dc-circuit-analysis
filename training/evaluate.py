"""
Модуль для оценки и сравнения моделей анализа электрических цепей

Предоставляет функции для:
- Оценки качества модели на тестовых данных
- Сравнения baseline и обученной модели
- Генерации отчетов с визуализацией результатов

"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import json
from typing import Dict, List
from training.datasets import DCCircuitDataset
from training.utils import LMStudioClient
from base.data import Data
from dc_circuit.game import DCCircuitGame

# matplotlib импортируем только когда нужно для визуализации
matplotlib = None


def evaluate_model_on_datasets(model_path: str,
                              test_datasets: Dict[int, DCCircuitDataset],
                              batch_size: int = 50) -> Dict[int, float]:
    """
    Оценивает модель на тестовых датасетах используя API

    Args:
        model_path: Путь к модели или название модели
        test_datasets: Словарь с датасетами по уровням сложности
        batch_size: Количество образцов для оценки на каждый уровень сложности

    Returns:
        Словарь {сложность: точность}
    """
    print(f"🔥 Оцениваем модель: {model_path}")

    # Инициализируем LM Studio клиент
    lm_studio_client = LMStudioClient()

    # Проверяем доступность LM Studio сервера
    if not lm_studio_client.health_check():
        print("❌ LM Studio сервер недоступен. Убедитесь, что сервер запущен на http://localhost:1234")
        return {}

    # Создаем оценщик модели
    evaluator = ModelEvaluator(lm_studio_client)

    results = {}

    for difficulty, dataset in test_datasets.items():
        print(f"📊 Оцениваем сложность {difficulty}...")

        # Конвертируем датасет в список Data объектов для оценки
        test_data = []
        for i in range(min(len(dataset), batch_size)):
            item = dataset[i]
            data_obj = Data(
                question=item["question"],
                answer=item["answer"],
                difficulty=difficulty
            )
            test_data.append(data_obj)

        # Запускаем оценку на датасете
        eval_results = evaluator.evaluate_on_dataset(test_data, batch_size)
        accuracy = eval_results["accuracy"]
        correct = eval_results["correct"]
        total = eval_results["total"]

        results[difficulty] = accuracy
        print(f"✅ Сложность {difficulty}: {accuracy:.3f} ({correct}/{total})")

    return results


class ModelEvaluator:
    """
    Класс для оценки качества модели анализа электрических цепей

    Использует LM Studio API для генерации ответов и верификации их правильности
    """

    def __init__(self, lm_studio_client: LMStudioClient):
        """
        Инициализация оценщика модели

        Args:
            lm_studio_client: Клиент для работы с LM Studio API
        """
        self.lm_studio_client = lm_studio_client
        self.system_prompt = self._get_system_prompt()

    def _get_system_prompt(self) -> str:
        """
        Возвращает системный промпт для модели с инструкциями по формату ответа
        """
        return (
            "You are an expert circuit analysis engineer. Solve electrical circuit problems using physics laws.\n\n"
            "Respond in the following format:\n"
            "<think>Reason step by step briefly.</think>\n"
            "<answer>Return ONLY the final number with exactly 3 decimal places (e.g., 1.234), no units.</answer>"
        )

    def generate_response(self, question: str) -> str:
        """
        Генерирует ответ модели на вопрос через LM Studio API

        Args:
            question: Текст вопроса

        Returns:
            Ответ модели или пустая строка при ошибке
        """
        # Формируем полный промпт с системным сообщением
        full_prompt = f"{self.system_prompt}\n\nQuestion: {question}\nAnswer:"

        try:
            # Отправляем запрос к LM Studio API
            response = self.lm_studio_client.generate(
                prompt=full_prompt,
                max_tokens=128,
                temperature=0.0,  # Детерминированные ответы
                stop_sequences=["\n", " "]  # Останавливаемся на переносах строк и пробелах
            )
            return response
        except Exception as e:
            print(f"❌ Ошибка генерации ответа: {e}")
            return ""

    def evaluate_on_dataset(self, test_data: List[Data], max_samples: int = 100) -> Dict:
        """
        Оценивает модель на наборе данных

        Args:
            test_data: Список тестовых данных
            max_samples: Максимальное количество образцов для оценки

        Returns:
            Словарь с результатами: {"accuracy": float, "correct": int, "total": int}
        """
        correct_count = 0
        total_count = min(len(test_data), max_samples)

        print(f"📊 Оцениваем {total_count} образцов...")

        for i, data_item in enumerate(test_data[:total_count]):
            # Выводим прогресс каждые 10 образцов
            if i % 10 == 0:
                print(f"   Проверено {i}/{total_count}...")

            # Генерируем ответ модели
            model_response = self.generate_response(data_item.question)

            if not model_response:
                continue

            # Проверяем правильность ответа используя верификатор
            game = DCCircuitGame()
            test_data_obj = Data(
                question=data_item.question,
                answer=data_item.answer,
                difficulty=data_item.difficulty
            )

            if game.verify(test_data_obj, model_response):
                correct_count += 1

        # Вычисляем точность
        accuracy = correct_count / total_count if total_count > 0 else 0.0

        return {
            "accuracy": accuracy,
            "correct": correct_count,
            "total": total_count
        }


def load_test_datasets() -> Dict[int, DCCircuitDataset]:
    """
    Загружает сохраненные тестовые датасеты по уровням сложности

    Проходит по всем уровням сложности (1-10) и пытается загрузить
    соответствующие pickle файлы с тестовыми данными

    Returns:
        Словарь {сложность: датасет}
    """
    datasets = {}

    # Проходим по всем уровням сложности от 1 до 10
    for difficulty in range(1, 11):
        try:
            # Пытаемся загрузить датасет для текущего уровня сложности
            with open(f"test_dataset_difficulty_{difficulty}.pkl", "rb") as f:
                data_list = pickle.load(f)
                datasets[difficulty] = DCCircuitDataset(data_list)
                print(f"✅ Загружен датасет сложности {difficulty}: {len(data_list)} образцов")
        except FileNotFoundError:
            print(f"⚠️ Датасет сложности {difficulty} не найден")

    return datasets


def compare_models(baseline_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
                   trained_model: str = "./dc_circuit_model") -> tuple[Dict[int, float], Dict[int, float]]:
    """
    Сравнивает baseline и обученную модель

    Загружает тестовые датасеты и оценивает обе модели на них,
    возвращая результаты для дальнейшего анализа

    Args:
        baseline_model: Название baseline модели
        trained_model: Путь к обученной модели

    Returns:
        Кортеж (результаты_baseline, результаты_обученной_модели)
    """
    print("🔍 Сравниваем модели...")

    # Загружаем тестовые датасеты по уровням сложности
    test_datasets = load_test_datasets()

    if not test_datasets:
        print("❌ Тестовые датасеты не найдены. Сначала создайте их.")
        return {}, {}

    # Оцениваем baseline модель
    print(f"📊 Оцениваем baseline модель: {baseline_model}")
    baseline_results = evaluate_model_on_datasets(baseline_model, test_datasets)

    # Оцениваем обученную модель
    print(f"📊 Оцениваем обученную модель: {trained_model}")
    trained_results = evaluate_model_on_datasets(trained_model, test_datasets)

    return baseline_results, trained_results


def generate_full_report(baseline_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
                        trained_model: str = "./dc_circuit_model") -> tuple[Dict[int, float], Dict[int, float]]:
    """
    Генерирует полный отчет оценки с визуализацией

    Выполняет сравнение моделей, рассчитывает улучшения,
    создает графики и сохраняет результаты в JSON

    Args:
        baseline_model: Baseline модель для сравнения
        trained_model: Обученная модель

    Returns:
        Кортеж с результатами оценки
    """
    print("📈 Генерируем полный отчет оценки...")

    # Получаем результаты сравнения моделей
    baseline_results, trained_results = compare_models(baseline_model, trained_model)

    if not baseline_results or not trained_results:
        print("❌ Невозможно сгенерировать отчет: отсутствуют результаты оценки")
        return {}, {}

    # Вычисляем улучшения для каждого уровня сложности
    difficulties = sorted(set(baseline_results.keys()) & set(trained_results.keys()))
    improvements = {}

    for difficulty in difficulties:
        baseline_acc = baseline_results[difficulty]
        trained_acc = trained_results[difficulty]
        improvement = trained_acc - baseline_acc
        improvements[difficulty] = improvement

        # Выводим результаты с эмодзи для наглядности
        status = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
        print(f"{status} Сложность {difficulty}: {baseline_acc:.3f} → {trained_acc:.3f} ({improvement:.3f})")

    # Вычисляем среднее улучшение по всем уровням сложности
    if improvements:
        avg_improvement = sum(improvements.values()) / len(improvements)
        print(f"\n🎯 Среднее улучшение: {avg_improvement:.3f}")

        if avg_improvement > 0:
            print("🎉 УСПЕХ: Модель улучшилась после обучения!")
        else:
            print("⚠️ Модель не улучшилась. Нужно доработать обучение.")

    # Создаем визуализацию результатов
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))

        # График точности моделей
        plt.subplot(1, 2, 1)
        plt.plot(difficulties, [baseline_results[d] for d in difficulties],
                'o-', label='Baseline', color='blue')
        plt.plot(difficulties, [trained_results[d] for d in difficulties],
                'o-', label='Обученная', color='green')
        plt.xlabel('Сложность')
        plt.ylabel('Точность')
        plt.title('Сравнение точности моделей')
        plt.legend()
        plt.grid(True)

        # График улучшений
        plt.subplot(1, 2, 2)
        plt.bar(difficulties, [improvements[d] for d in difficulties],
               color=['green' if x > 0 else 'red' for x in improvements.values()])
        plt.xlabel('Сложность')
        plt.ylabel('Улучшение')
        plt.title('Улучшение после обучения')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('reports/model_comparison.png', dpi=300, bbox_inches='tight')
        print("📊 График сохранен: reports/model_comparison.png")

    except Exception as e:
        print(f"⚠️ Не удалось создать график: {e}")

    # Сохраняем детальные результаты в JSON файл
    results_data = {
        "baseline_model": baseline_model,
        "trained_model": trained_model,
        "baseline_results": baseline_results,
        "trained_results": trained_results,
        "improvements": improvements,
        "avg_improvement": avg_improvement if improvements else 0
    }

    # Создаем директорию reports если её нет
    os.makedirs("reports", exist_ok=True)
    with open("reports/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)

    print("💾 Результаты сохранены: reports/evaluation_results.json")

    return baseline_results, trained_results