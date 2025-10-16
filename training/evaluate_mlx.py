import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict
import re
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
import matplotlib.pyplot as plt
import numpy as np

from base.data import Data
from dc_circuit.game import DCCircuitGame
from config import CircuitConfig, VerifierConfig, TrainingConfig
from base.utils import get_system_prompt

# Глобальная переменная для папки результатов
results_dir = "results"



class Evaluator:
    """Оценщик для тестирования моделей (MLX версия)."""


    def __init__(
        self,
        baseline_model: str = "unsloth/Qwen2.5-1.5B-instruct",  # Можно использовать любую HF модель!
        trained_model_path: str = "./dc_circuit_model_rl",
        samples_per_difficulty: int = 5
    ):
        """Инициализация оценщика.

        Args:
            baseline_model: Название любой HF модели (mlx-lm конвертирует автоматически)
            trained_model_path: Путь к обученной модели
            samples_per_difficulty: Количество задач на уровень сложности
        """
        self.baseline_model_name = baseline_model
        self.trained_model_path = trained_model_path
        self.samples_per_difficulty = samples_per_difficulty

        # Конфигурации
        self.circuit_config = CircuitConfig()
        self.verifier_config = VerifierConfig()
        self.training_config = TrainingConfig()

        # Game для генерации и верификации
        self.game = DCCircuitGame(self.circuit_config, self.verifier_config)

    def _has_strict_answer_format(self, response: str) -> bool:
        """Проверяет строгий формат ответа в <answer>: ровно число с 3 знаками.

        Условия:
        - В тексте есть теги <answer>...</answer>
        - Внутри ровно одно число формата X.XXX (3 десятичных знака)
        - Нет единиц измерения и лишнего текста
        """
        if not response:
            return False
        # Найти последний <answer>...</answer>
        tag_matches = re.findall(r"<answer>([\s\S]*?)</answer>", response, flags=re.IGNORECASE)
        if not tag_matches:
            return False
        content = tag_matches[-1].strip()
        # Должно быть строго число с 3 десятичными, без единиц и текста
        return bool(re.fullmatch(r"[-+]?\d+\.\d{3}", content))

    def generate_test_data(self) -> Dict[int, List[Data]]:
        """Генерирует тестовые данные для всех уровней сложности.

        Returns:
            Словарь {difficulty: list_of_data}
        """
        print("\n📝 Генерация тестовых данных...")
        test_data = {}

        for difficulty in [1, 2, 5, 6]:
            print(f"  Сложность {difficulty}: генерация {self.samples_per_difficulty} задач...")
            data_list = self.game.generate(
                num_of_questions=self.samples_per_difficulty,
                difficulty=difficulty
            )
            test_data[difficulty] = data_list
            print(f"    ✓ Сгенерировано {len(data_list)} задач")

        total = sum(len(data) for data in test_data.values())
        print(f"  Всего тестовых задач: {total}\n")

        return test_data

    def load_model(self, model_path: str, is_trained: bool = False):
        """Загружает модель через MLX или transformers (для совместимости с LoRA).

        Args:
            model_path: Путь к модели или Hugging Face ID (любая HF модель)
            is_trained: True если это обученная модель с LoRA

        Returns:
            (model, tokenizer) - модель и токенизатор
        """
        # Определяем тип модели: Hugging Face или локальная
        is_huggingface = "/" in model_path and not model_path.startswith("./") and not model_path.startswith("../") and not model_path.startswith("/")

        if is_huggingface:
            # Hugging Face модель - используем MLX
            print(f"🚀 Загрузка HF модели через MLX: {model_path}")
            try:
                model, tokenizer = load(model_path)
                print("  ✓ Модель загружена через MLX\n")
            except Exception as e:
                print(f"⚠️  MLX не доступен ({e}), используем transformers")
                from transformers import AutoModelForCausalLM, AutoTokenizer
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype="auto",
                    device_map="auto",
                    trust_remote_code=True
                )
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                print("  ✓ Модель загружена через transformers\n")
        else:
            # Локальная модель (LoRA) - используем transformers для совместимости
            print(f"📦 Загрузка локальной LoRA модели через transformers: {model_path}")

            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Проверяем существование пути
            if not os.path.exists(model_path):
                possible_paths = [
                    model_path,
                    f"./{model_path}",
                    f"../{model_path}",
                    os.path.join(os.getcwd(), model_path),
                    os.path.join(os.path.dirname(os.getcwd()), model_path)
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        model_path = path
                        break
                else:
                    raise FileNotFoundError(f"Модель не найдена по путям: {possible_paths}")

            # Загружаем базовую модель + LoRA адаптер
            try:
                # Определяем базовую модель из адаптера
                import json
                with open(os.path.join(model_path, "adapter_config.json"), "r") as f:
                    adapter_config = json.load(f)
                base_model_name = adapter_config.get("base_model_name_or_path", "unsloth/Qwen2.5-1.5B-instruct")

                print(f"  Базовая модель: {base_model_name}")
                print(f"  LoRA адаптер: {model_path}")

                # Загружаем базовую модель
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype="auto",
                    device_map="auto",
                    trust_remote_code=True
                )

                # Загружаем LoRA адаптер
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, model_path)

                tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
                print("  ✓ LoRA модель загружена через transformers\n")

            except Exception as e:
                print(f"❌ Ошибка загрузки LoRA модели: {e}")
                raise

        # Устанавливаем chat template для Qwen (если его нет)
        if tokenizer.chat_template is None:
            tokenizer.chat_template = """{% for message in messages %}
{% if message['role'] == 'system' %}
<|im_start|>system
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'user' %}
<|im_start|>user
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'assistant' %}
<|im_start|>assistant
{{ message['content'] }}<|im_end|>
{% endif %}
{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"""

        return model, tokenizer

    def generate_answer(
        self,
        model,
        tokenizer,
        question: str,
        use_mlx: bool = True
    ) -> str:
        """Генерирует ответ модели через MLX или transformers.

        Args:
            model: Модель (MLX или transformers)
            tokenizer: Токенизатор
            question: Вопрос
            use_mlx: True для MLX генерации, False для transformers

        Returns:
            Ответ модели
        """
        # Формируем промпт
        messages = []

        # Всегда добавляем системный промпт RL среды
        messages.append({"role": "system", "content": get_system_prompt()})

        # Добавляем основной вопрос (уже содержит описание цепи из среды)
        messages.append({"role": "user", "content": question})

        # Формируем промпт для MLX
        prompt_parts = []
        for message in messages:
            if message["role"] == "system":
                prompt_parts.append(f"<|im_start|>system\n{message['content']}<|im_end|>")
            elif message["role"] == "user":
                prompt_parts.append(f"<|im_start|>user\n{message['content']}<|im_end|>")
            elif message["role"] == "assistant":
                prompt_parts.append(f"<|im_start|>assistant\n{message['content']}<|im_end|>")

        # Добавляем начало ответа для текущего вопроса
        prompt_parts.append("<|im_start|>assistant\n")
        prompt = "\n".join(prompt_parts)

        # 🔍 ОТЛАДОЧНЫЙ ВЫВОД ГЕНЕРАЦИИ
        print(f"\n🔧 ОТЛАДКА ГЕНЕРАЦИИ:")
        print(f"📝 ПРОМПТ (первые 200 символов): {prompt[:200]}...")

        if use_mlx:
            # Генерируем ответ через MLX
            response = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=self.training_config.max_completion_length,
                verbose=False
            )
        else:
            # Генерируем ответ через transformers (для LoRA моделей)
            import torch
            inputs = tokenizer(prompt, return_tensors="pt")

            # Перемещаем на устройство модели
            if hasattr(model, 'device'):
                inputs = inputs.to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.training_config.max_completion_length,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        print(f"🤖 ПОЛНЫЙ ОТВЕТ МОДЕЛИ:")
        print(f"{response}")
        print(f"📏 Длина ответа: {len(response)} символов")


        return response

    def evaluate_model_on_data(
        self,
        model,
        tokenizer,
        test_data: Dict[int, List[Data]],
        method_name: str,
        use_mlx: bool = True
    ) -> Dict[int, Dict[str, float]]:
        """Оценивает модель на тестовых данных.

        Args:
            model: Модель (MLX или transformers)
            tokenizer: Токенизатор
            test_data: Тестовые данные
            method_name: Название метода для вывода
            use_mlx: True для MLX генерации, False для transformers

        Returns:
            Словарь {difficulty: {"accuracy": float, "format_score": float, "strict_format_score": float}}
        """
        print(f"🧪 Тестирование: {method_name}")

        # 🔍 ОТЛАДОЧНЫЙ ВЫВОД СИСТЕМНОГО ПРОМПТА
        from base.utils import get_system_prompt
        system_prompt = get_system_prompt()
        print(f"\n📋 СИСТЕМНЫЙ ПРОМПТ RL СРЕДЫ:")
        print("=" * 80)
        print(f"{system_prompt}")
        print("=" * 80)

        # Группируем задачи по типам цепей
        circuit_type_results = {"series": {}, "parallel": {}}

        results = {}

        for difficulty, data_list in sorted(test_data.items()):
            # Группируем по типам цепей
            series_correct = 0
            series_format_correct = 0
            series_strict_format_correct = 0
            series_total = 0

            parallel_correct = 0
            parallel_format_correct = 0
            parallel_strict_format_correct = 0
            parallel_total = 0

            for i, data in enumerate(data_list):
                # Получаем тип цепи из метаданных
                circuit_type = getattr(data, 'metadata', {}).get('circuit_type', 'unknown')
                question_type = getattr(data, 'metadata', {}).get('question_type', 'unknown')
                # Генерируем ответ
                response = self.generate_answer(
                    model, tokenizer, data.question, use_mlx
                )

    
                print("=" * 80)
                print(f"📋 ПОЛНАЯ ЗАДАЧА:")
                print(f"{data.question}")
                print(f"\n✅ ОЖИДАЕМЫЙ ОТВЕТ: {data.answer}")
                print(f"\n🤖 ОТВЕТ МОДЕЛИ:")
                print(f"{response}")

                # Извлекаем ответ из ответа модели
                from base.utils import extract_answer
                extracted_answer = extract_answer(response)
                print(f"\n🔍 ИЗВЛЕЧЕННЫЙ ОТВЕТ: '{extracted_answer}'")

                # Проверяем правильность
                accuracy_score = self.game.verifier.get_accuracy_score(data, response)
                print(f"\n📊 РЕЗУЛЬТАТ ВЕРИФИКАЦИИ:")

                print(f"  Accuracy Score: {accuracy_score:.3f}")

                # Исправлено: переменные has_think, has_answer, strict_format_ok должны быть определены
                has_think = "<think>" in response.lower()
                has_answer = "<answer>" in response.lower()
                strict_format_ok = self._has_strict_answer_format(response)

                # Суммируем accuracy score по типам цепей
                if circuit_type == "series":
                    series_correct += accuracy_score
                    series_total += 1
                    # Теперь format_ok учитывает и теги И строгий формат
                    if has_think and has_answer and strict_format_ok:
                        series_format_correct += 1
                    if strict_format_ok:
                        series_strict_format_correct += 1
                elif circuit_type == "parallel":
                    parallel_correct += accuracy_score
                    parallel_total += 1
                    # Теперь format_ok учитывает и теги И строгий формат
                    if has_think and has_answer and strict_format_ok:
                        parallel_format_correct += 1
                    if strict_format_ok:
                        parallel_strict_format_correct += 1

                print("=" * 80)

                # Прогресс
                if (i + 1) % 5 == 0 or (i + 1) == len(data_list):
                    print(f"  Сложность {difficulty}: {i+1}/{len(data_list)} задач...", end='\r')

            # Сохраняем результаты для каждого типа цепи
            if series_total > 0:
                series_accuracy = round(series_correct) / series_total
                series_format_score = series_format_correct / series_total
                series_strict_format_score = series_strict_format_correct / series_total
                circuit_type_results["series"][difficulty] = {
                    "accuracy": series_accuracy,
                    "format_score": series_format_score,
                    "strict_format_score": series_strict_format_score,
                    "total_tasks": series_total
                }
                print(f"  Сложность {difficulty} (Series): {round(series_correct)}/{series_total} = {series_accuracy:.1%} | Формат (строгий): {series_format_correct}/{series_total} = {series_format_score:.1%}")

            if parallel_total > 0:
                parallel_accuracy = round(parallel_correct) / parallel_total
                parallel_format_score = parallel_format_correct / parallel_total
                parallel_strict_format_score = parallel_strict_format_correct / parallel_total
                circuit_type_results["parallel"][difficulty] = {
                    "accuracy": parallel_accuracy,
                    "format_score": parallel_format_score,
                    "strict_format_score": parallel_strict_format_score,
                    "total_tasks": parallel_total
                }
                print(f"  Сложность {difficulty} (Parallel): {round(parallel_correct)}/{parallel_total} = {parallel_accuracy:.1%} | Формат (строгий): {parallel_format_correct}/{parallel_total} = {parallel_format_score:.1%}")

        # Возвращаем результаты по типам цепей
        return circuit_type_results

    def run_evaluation(self):
        """Запускает полную оценку всех трех методов."""
        print("================================================")
        print("                ОЦЕНКА МОДЕЛЕЙ DC CIRCUIT ANALYSIS (MLX)")
        print("================================================")

        # 🔍 ОТЛАДОЧНАЯ ИНФОРМАЦИЯ
        print(f"\n🔧 ОТЛАДОЧНАЯ ИНФОРМАЦИЯ:")
        print(f"📊 Образцов на сложность: {self.samples_per_difficulty}")
        print(f"🎯 Сложности: {self.circuit_config.difficulties}")
        print(f"📏 Максимальная длина ответа: {self.training_config.max_completion_length}")
        print("=" * 80)

        # 1. Генерация тестовых данных
        test_data = self.generate_test_data()

        # 2. Загрузка baseline модели
        baseline_model, baseline_tokenizer = self.load_model(
            self.baseline_model_name,
            is_trained=False
        )

        # 3. Baseline Model оценка (с системным промптом RL среды)
        print("-"*70)
        baseline_results = self.evaluate_model_on_data(
            baseline_model,
            baseline_tokenizer,
            test_data,
            "Baseline Model (with RL system prompt)",
            use_mlx=True  # HF модель через MLX
        )

        # Очистка памяти
        del baseline_model, baseline_tokenizer

        # 5. GRPO Trained оценка (если модель существует)
        print("-"*70)
        grpo_results = {"series": {}, "parallel": {}}
        if os.path.exists(self.trained_model_path):
            trained_model, trained_tokenizer = self.load_model(
                self.trained_model_path,
                is_trained=True
            )

            grpo_results = self.evaluate_model_on_data(
                trained_model,
                trained_tokenizer,
                test_data,
                "GRPO Trained (with LoRA)",
                use_mlx=False  # LoRA модель через transformers
            )

            del trained_model, trained_tokenizer
        else:
            print(f"⚠️  Обученная модель не найдена: {self.trained_model_path}")
            print(f"   Пропускаем оценку GRPO Trained\n")
            # Создаем пустые результаты для отсутствующей модели
            grpo_results = {"series": {}, "parallel": {}}

        # 6. Вывод итоговых результатов по типам цепей
        self.print_summary(baseline_results, grpo_results)

        return {
            "baseline_model": baseline_results,
            "grpo_trained": grpo_results
        }

    def print_summary(
        self,
        baseline: Dict[str, Dict[int, Dict[str, float]]],
        grpo: Dict[str, Dict[int, Dict[str, float]]]
    ):
        """Выводит итоговую таблицу результатов по типам цепей с красивой диаграммой.

        Args:
            baseline: Результаты Baseline Model по типам цепей
            grpo: Результаты GRPO Trained по типам цепей
        """
        print("="*80)
        print(" 📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ ПО ТИПАМ ЦЕПЕЙ")
        print("="*80)
        print()

        for circuit_type in ["series", "parallel"]:
            if not baseline.get(circuit_type) and not grpo.get(circuit_type):
                continue

            print(f"🔌 ТИП ЦЕПИ: {circuit_type.upper()}")
            print("-" * 60)

            # Получаем все сложности для этого типа цепи
            all_difficulties = set()
            if circuit_type in baseline:
                all_difficulties.update(baseline[circuit_type].keys())
            if circuit_type in grpo:
                all_difficulties.update(grpo[circuit_type].keys())
            difficulties = sorted(all_difficulties)

            if not difficulties:
                continue

            header = "| Метод                  |" + "".join(f" Сложность {d} |" for d in difficulties) + " Среднее |"
            separator = "|" + "-" * 24 + "|" + "".join("-" * 13 + "|" for _ in difficulties) + "-" * 9 + "|"

            print("🎯 ТОЧНОСТЬ ОТВЕТОВ:")
            print(header)
            print(separator)

            # Baseline Model
            if circuit_type in baseline:
                baseline_values = [baseline[circuit_type].get(d, {}).get('accuracy', 0.0) for d in difficulties]
                avg_baseline_acc = sum(baseline_values) / len(baseline_values) if baseline_values else 0.0
                print(f"| Baseline Model         |" + "".join(f" {v:>10.1%} |" for v in baseline_values) + f" {avg_baseline_acc:>6.1%} |")
            else:
                print(f"| Baseline Model         |" + "".join("       0.0% |" for _ in difficulties) + "    0.0% |")

            # GRPO Trained
            if circuit_type in grpo:
                grpo_values = [grpo[circuit_type].get(d, {}).get('accuracy', 0.0) for d in difficulties]
                avg_grpo_acc = sum(grpo_values) / len(grpo_values) if grpo_values else 0.0
                print(f"| GRPO Trained           |" + "".join(f" {v:>10.1%} |" for v in grpo_values) + f" {avg_grpo_acc:>6.1%} |")
            else:
                print(f"| GRPO Trained           |" + "".join("       0.0% |" for _ in difficulties) + "    0.0% |")

            print()
            print(f"📊 Всего задач для {circuit_type}:")
            if circuit_type in baseline:
                total_baseline = sum(baseline[circuit_type].get(d, {}).get('total_tasks', 0) for d in difficulties)
                print(f"   Baseline: {total_baseline} задач")
            if circuit_type in grpo:
                total_grpo = sum(grpo[circuit_type].get(d, {}).get('total_tasks', 0) for d in difficulties)
                print(f"   GRPO: {total_grpo} задач")
            print("-" * 60)
            print()

        # Создаем диаграммы
        self.print_visual_chart(baseline, grpo)

    def print_visual_chart(self, baseline, grpo):
        """Создает красивые matplotlib диаграммы по типам цепей."""
        # Создаем диаграммы для каждого типа цепи отдельно
        for circuit_type in ["series", "parallel"]:
            baseline_circuit = baseline.get(circuit_type, {})
            grpo_circuit = grpo.get(circuit_type, {})

            if not baseline_circuit and not grpo_circuit:
                continue

            # Получаем сложности для этого типа цепи
            all_difficulties = set(list(baseline_circuit.keys()) + list(grpo_circuit.keys()))
            difficulties = sorted(all_difficulties)

            if not difficulties:
                continue

            # Подготавливаем данные для диаграммы
            methods = ["Baseline Model", "GRPO Trained"]
            accuracy_data = {
                'Baseline Model': [baseline_circuit.get(d, {}).get('accuracy', 0.0) * 100 for d in difficulties],
                'GRPO Trained': [grpo_circuit.get(d, {}).get('accuracy', 0.0) * 100 for d in difficulties]
            }

            format_data = {
                'Baseline Model': [baseline_circuit.get(d, {}).get('format_score', 0.0) * 100 for d in difficulties],
                'GRPO Trained': [grpo_circuit.get(d, {}).get('format_score', 0.0) * 100 for d in difficulties]
            }

            # Создаем диаграммы для этого типа цепи
            colors = ['#FF6B6B', '#4ECDC4']

            # Диаграмма: Точность по сложностям для типа цепи
            plt.figure(figsize=(12, 8))
            x = np.arange(len(difficulties))
            width = 0.35

            for i, (method, values) in enumerate(accuracy_data.items()):
                offset = width * i
                bars = plt.bar(x + offset, values, width, label=method, color=colors[i], alpha=0.8)
                plt.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=10)

            plt.ylabel('Accuracy (%)', fontsize=12)
            plt.title(f'DC Circuit Analysis - {circuit_type.title()} Circuits - Accuracy by Difficulty (MLX)', fontsize=14, fontweight='bold')
            plt.xticks(x + width/2, [f'Difficulty {d}' for d in difficulties])
            plt.legend(loc='upper left')
            plt.ylim(0, 100)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{results_dir}/{circuit_type}_accuracy_by_difficulty.png', dpi=300, bbox_inches='tight')
            print(f"📊 Диаграмма {circuit_type} точность сохранена в {results_dir}/{circuit_type}_accuracy_by_difficulty.png")
            plt.close()


        # Дополнительные графики: средние показатели

        # График 3: Средняя точность по всем уровням сложности
        plt.figure(figsize=(10, 6))
        methods = ["Baseline Model", "GRPO Trained"]

        # Собираем данные по всем типам цепей
        all_accuracy_data = {"Baseline Model": [], "GRPO Trained": []}
        all_format_data = {"Baseline Model": [], "GRPO Trained": []}

        for circuit_type in ["series", "parallel"]:
            baseline_circuit = baseline.get(circuit_type, {})
            grpo_circuit = grpo.get(circuit_type, {})

            # Получаем все сложности для этого типа цепи
            all_difficulties = set(list(baseline_circuit.keys()) + list(grpo_circuit.keys()))
            difficulties = sorted(all_difficulties)

            if difficulties:
                # Добавляем данные точности
                for method in methods:
                    if method == "Baseline Model":
                        circuit_data = baseline_circuit
                    else:
                        circuit_data = grpo_circuit

                    accuracy_values = [circuit_data.get(d, {}).get('accuracy', 0.0) * 100 for d in difficulties]
                    all_accuracy_data[method].extend(accuracy_values)

                    format_values = [circuit_data.get(d, {}).get('format_score', 0.0) * 100 for d in difficulties]
                    all_format_data[method].extend(format_values)

        # Средняя точность
        avg_accuracy = [sum(all_accuracy_data[m]) / len(all_accuracy_data[m]) if all_accuracy_data[m] else 0 for m in methods]
        bars = plt.bar(methods, avg_accuracy, color=colors, alpha=0.8)
        plt.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=12)

        plt.ylabel('Average Accuracy (%)', fontsize=12)
        plt.title('DC Circuit Analysis - Average Accuracy Across All Difficulties (MLX)', fontsize=14, fontweight='bold')
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{results_dir}/overall_average_accuracy.png', dpi=300, bbox_inches='tight')
        print(f"📊 График общей точности сохранен в {results_dir}/overall_average_accuracy.png")
        plt.close()

        # График 4: Средний формат (общий)
        plt.figure(figsize=(10, 6))
        avg_format = [sum(all_format_data[m]) / len(all_format_data[m]) if all_format_data[m] else 0 for m in methods]
        bars = plt.bar(methods, avg_format, color=colors, alpha=0.8)
        plt.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=12)

        plt.ylabel('Average Format Score (%)', fontsize=12)
        plt.title('DC Circuit Analysis - Average Format Score Across All Tasks (MLX)', fontsize=14, fontweight='bold')
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{results_dir}/overall_average_format.png', dpi=300, bbox_inches='tight')
        print(f"📊 График общего формата сохранен в {results_dir}/overall_average_format.png")
        plt.close()

        print(f"📊 Все диаграммы сохранены в папке {results_dir}/")


def main():
    """Главная функция."""
    # Создаем папку results если не существует
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    evaluator = Evaluator(
        baseline_model="unsloth/Qwen2.5-1.5B-instruct",
        trained_model_path="/Users/stepprog/Downloads/content 2/dc_circuit_model_rl",
        samples_per_difficulty=5
    )

    results = evaluator.run_evaluation()

    # Сохраняем результаты в файл
    import json
    results_file = f"{results_dir}/evaluation_results_mlx.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"💾 Результаты сохранены в {results_file}")


if __name__ == "__main__":
    main()
