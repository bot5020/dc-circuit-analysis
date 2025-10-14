"""Cкрипт оценки моделей для DC Circuit Analysis.

Тестирует три подхода:
1. Zero-shot - базовая модель без специального промпта
2. Prompt Engineering - модель с детальным системным промптом
3. GRPO Trained - обученная модель с LoRA
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict
import re
import torch
from unsloth import FastLanguageModel
from vllm import LLM, SamplingParams

from base.data import Data
from dc_circuit.game import DCCircuitGame
from config import CircuitConfig, VerifierConfig, TrainingConfig
from base.utils import get_system_prompt



class Evaluator:
    """Оценщик для тестирования моделей."""

    
    def __init__(
        self,
        baseline_model: str = "unsloth/Qwen2.5-0.5B",
        trained_model_path: str = "./dc_circuit_model_rl",
        samples_per_difficulty: int = 5
    ):
        """Инициализация оценщика.
        
        Args:
            baseline_model: Название базовой модели
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
        
        for difficulty in [1, 2]:  
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
        """Загружает модель через vLLM.
        
        Args:
            model_path: Путь к модели
            is_trained: True если это обученная модель с LoRA
        
        Returns:
            (llm, sampling_params)
        """
        print(f"📥 Загрузка модели через vLLM: {model_path}")
        
        # Настройки vLLM
        llm = LLM(
            model=model_path,
            max_model_len=self.training_config.max_seq_length,
            dtype='bfloat16',
            gpu_memory_utilization=self.DEFAULT_GPU_MEMORY_UTILIZATION,
            trust_remote_code=True,
            enforce_eager=False
        )
        
        # Параметры семплирования
        sampling_params = SamplingParams(
            temperature=self.DEFAULT_TEMPERATURE,
            max_tokens=self.training_config.max_completion_length,
            stop=["<|im_end|>", "</s>"]
        )
        
        print(f"  ✓ Модель загружена через vLLM\n")
        return llm, sampling_params
    
    def generate_answer(
        self, 
        llm, 
        sampling_params, 
        question: str, 
        use_system_prompt: bool = True
    ) -> str:
        """Генерирует ответ модели на вопрос через vLLM.
        
        Args:
            llm: vLLM модель
            sampling_params: Параметры семплирования
            question: Вопрос
            use_system_prompt: Использовать ли системный промпт
        
        Returns:
            Ответ модели
        """
        # Формируем промпт
        if use_system_prompt:
            messages = [
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": question}
            ]
        else:
            # Zero-shot: только вопрос, минимальный промпт
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question}
            ]
        
        # Формируем промпт для vLLM (простой формат)
        if use_system_prompt:
            prompt = f"<|im_start|>system\n{get_system_prompt()}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        else:
            prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        
        # 🔍 ОТЛАДОЧНЫЙ ВЫВОД ГЕНЕРАЦИИ
        print(f"\n🔧 ОТЛАДКА ГЕНЕРАЦИИ:")
        print(f"📝 ПРОМПТ (первые 200 символов): {prompt[:200]}...")
        
        # Генерируем ответ через vLLM
        outputs = llm.generate([prompt], sampling_params)
        response = outputs[0].outputs[0].text
        
        print(f"🤖 ПОЛНЫЙ ОТВЕТ МОДЕЛИ:")
        print(f"{response}")
        print(f"📏 Длина ответа: {len(response)} символов")
        
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Если модель генерирует весь системный промпт,
        # извлекаем только часть после "assistant"
        if "assistant" in response and len(response) > 1000:
            # Ищем последний "assistant" в ответе
            assistant_parts = response.split("assistant")
            if len(assistant_parts) > 1:
                # Берем только часть после последнего "assistant"
                response = assistant_parts[-1].strip()
                print(f"🔧 ИСПРАВЛЕНИЕ: Извлечен только ответ ассистента")
                print(f"✂️ ФИНАЛЬНЫЙ ОТВЕТ:")
                print(f"{response}")
        
        return response
    
    def evaluate_model_on_data(
        self,
        llm,
        sampling_params,
        test_data: Dict[int, List[Data]],
        method_name: str,
        use_system_prompt: bool = True
    ) -> Dict[int, Dict[str, float]]:
        """Оценивает модель на тестовых данных.
        
        Args:
            llm: vLLM модель
            sampling_params: Параметры семплирования
            test_data: Тестовые данные
            method_name: Название метода для вывода
            use_system_prompt: Использовать ли системный промпт
        
        Returns:
            Словарь {difficulty: {"accuracy": float, "format_score": float, "strict_format_score": float}}
        """
        print(f"🧪 Тестирование: {method_name}")
        
        # 🔍 ОТЛАДОЧНЫЙ ВЫВОД СИСТЕМНОГО ПРОМПТА
        if use_system_prompt:
            from base.utils import get_system_prompt
            system_prompt = get_system_prompt()
            print(f"\n📋 СИСТЕМНЫЙ ПРОМПТ:")
            print("=" * 80)
            print(f"{system_prompt}")
            print("=" * 80)
        else:
            print(f"\n📋 РЕЖИМ: Zero-shot (без системного промпта)")
        
        results = {}
        
        for difficulty, data_list in sorted(test_data.items()):
            correct = 0
            format_correct = 0
            strict_format_correct = 0
            total = len(data_list)
            
            for i, data in enumerate(data_list):
                # Генерируем ответ
                response = self.generate_answer(
                    llm, sampling_params, data.question, use_system_prompt
                )
                
                # 🔍 ОТЛАДОЧНЫЙ ВЫВОД
                print(f"\n🔍 ОТЛАДКА ЗАДАЧИ {i+1}/{total} (Сложность {difficulty}):")
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
                is_correct = self.game.verify(data, response)
                accuracy_score = self.game.verifier.get_accuracy_score(data, response)
                print(f"\n📊 РЕЗУЛЬТАТ ВЕРИФИКАЦИИ:")
                print(f"  Правильный: {is_correct}")
                print(f"  Accuracy Score: {accuracy_score:.3f}")
                
                if is_correct:
                    correct += 1
                
                # Проверяем формат ответа
                has_think = "<think>" in response
                has_answer = "<answer>" in response
                format_ok = has_think and has_answer
                strict_format_ok = self._has_strict_answer_format(response)
                print(f"\n📝 ФОРМАТ ОТВЕТА:")
                print(f"  Есть <think>: {has_think}")
                print(f"  Есть <answer>: {has_answer}")
                print(f"  Формат правильный: {format_ok}")
                print(f"  Строгий формат <answer>X.XXX</answer>: {strict_format_ok}")
                
                if format_ok:
                    format_correct += 1
                if strict_format_ok:
                    strict_format_correct += 1
                
                print("=" * 80)
                
                # Прогресс
                if (i + 1) % 5 == 0 or (i + 1) == total:
                    print(f"  Сложность {difficulty}: {i+1}/{total} задач...", end='\r')
            
            accuracy = correct / total if total > 0 else 0.0
            format_score = format_correct / total if total > 0 else 0.0
            strict_format_score = strict_format_correct / total if total > 0 else 0.0
            results[difficulty] = {
                "accuracy": accuracy,
                "format_score": format_score,
                "strict_format_score": strict_format_score
            }
            print(f"  Сложность {difficulty}: {correct}/{total} = {accuracy:.1%} | Формат: {format_correct}/{total} = {format_score:.1%} | Строгий формат: {strict_format_correct}/{total} = {strict_format_score:.1%}    ")
        
        # Общие показатели
        avg_accuracy = sum(r["accuracy"] for r in results.values()) / len(results) if results else 0.0
        avg_format = sum(r["format_score"] for r in results.values()) / len(results) if results else 0.0
        avg_strict_format = sum(r["strict_format_score"] for r in results.values()) / len(results) if results else 0.0
        print(f"  📊 Средняя точность: {avg_accuracy:.1%} | Средний формат: {avg_format:.1%} | Средний строгий формат: {avg_strict_format:.1%}\n")
        
        return results
    
    def run_evaluation(self):
        """Запускает полную оценку всех трех методов."""
        print("================================================")
        print("                ОЦЕНКА МОДЕЛЕЙ DC CIRCUIT ANALYSIS")
        print("================================================")
        
        # 🔍 ОТЛАДОЧНАЯ ИНФОРМАЦИЯ
        print(f"\n🔧 ОТЛАДОЧНАЯ ИНФОРМАЦИЯ:")
        print(f"📊 Образцов на сложность: {self.samples_per_difficulty}")
        print(f"🎯 Сложности: {self.circuit_config.difficulties}")
        print(f"🌡️ Температура: {self.DEFAULT_TEMPERATURE}")
        print(f"💾 GPU память: {self.DEFAULT_GPU_MEMORY_UTILIZATION}")
        print(f"📏 Максимальная длина ответа: {self.training_config.max_completion_length}")
        print("=" * 80)
        
        # 1. Генерация тестовых данных
        test_data = self.generate_test_data()
        
        # 2. Загрузка baseline модели
        baseline_llm, baseline_sampling_params = self.load_model(
            self.baseline_model_name, 
            is_trained=False
        )
        
        # 3. Zero-shot оценка (без специального промпта)
        print("-"*70)
        zero_shot_results = self.evaluate_model_on_data(
            baseline_llm,
            baseline_sampling_params,
            test_data,
            "Zero-shot (no system prompt)",
            use_system_prompt=False
        )
        
        # 4. Prompt Engineering оценка (с системным промптом)
        print("-"*70)
        prompt_eng_results = self.evaluate_model_on_data(
            baseline_llm,
            baseline_sampling_params,
            test_data,
            "Prompt Engineering (with system prompt)",
            use_system_prompt=True
        )
        
        # Очистка памяти
        del baseline_llm, baseline_sampling_params
        torch.cuda.empty_cache()
        
        # 5. GRPO Trained оценка (если модель существует)
        print("-"*70)
        grpo_results = {}
        if os.path.exists(self.trained_model_path):
            trained_llm, trained_sampling_params = self.load_model(
                self.trained_model_path,
                is_trained=True
            )
            
            grpo_results = self.evaluate_model_on_data(
                trained_llm,
                trained_sampling_params,
                test_data,
                "GRPO Trained (with LoRA)",
                use_system_prompt=True
            )
            
            del trained_llm, trained_sampling_params
            torch.cuda.empty_cache()
        else:
            print(f"⚠️  Обученная модель не найдена: {self.trained_model_path}")
            print(f"   Пропускаем оценку GRPO Trained\n")
            grpo_results = {1: {"accuracy": 0.0, "format_score": 0.0, "strict_format_score": 0.0}, 
                           2: {"accuracy": 0.0, "format_score": 0.0, "strict_format_score": 0.0}}
        
        # 6. Вывод итоговых результатов
        self.print_summary(zero_shot_results, prompt_eng_results, grpo_results)
        
        return {
            "zero_shot": zero_shot_results,
            "prompt_engineering": prompt_eng_results,
            "grpo_trained": grpo_results
        }
    
    def print_summary(
        self,
        zero_shot: Dict[int, Dict[str, float]],
        prompt_eng: Dict[int, Dict[str, float]],
        grpo: Dict[int, Dict[str, float]]
    ):
        """Выводит итоговую таблицу результатов с красивой диаграммой.
        
        Args:
            zero_shot: Результаты Zero-shot
            prompt_eng: Результаты Prompt Engineering
            grpo: Результаты GRPO
        """
        print("="*80)
        print(" 📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
        print("="*80)
        print()
        
        # Таблица точности
        print("🎯 ТОЧНОСТЬ ОТВЕТОВ:")
        print("| Метод                  | Сложность 1 | Сложность 2 | Среднее |")
        print("|------------------------|-------------|-------------|---------|")
        
        # Zero-shot
        avg_zero_acc = sum(zero_shot[d]["accuracy"] for d in zero_shot) / len(zero_shot) if zero_shot else 0.0
        print(f"| Zero-shot              | {zero_shot.get(1, {}).get('accuracy', 0.0):>10.1%} | "
              f"{zero_shot.get(2, {}).get('accuracy', 0.0):>10.1%} | {avg_zero_acc:>6.1%} |")
        
        # Prompt Engineering
        avg_pe_acc = sum(prompt_eng[d]["accuracy"] for d in prompt_eng) / len(prompt_eng) if prompt_eng else 0.0
        print(f"| Prompt Engineering     | {prompt_eng.get(1, {}).get('accuracy', 0.0):>10.1%} | "
              f"{prompt_eng.get(2, {}).get('accuracy', 0.0):>10.1%} | {avg_pe_acc:>6.1%} |")
        
        # GRPO Trained
        avg_grpo_acc = sum(grpo[d]["accuracy"] for d in grpo) / len(grpo) if grpo else 0.0
        print(f"| GRPO Trained           | {grpo.get(1, {}).get('accuracy', 0.0):>10.1%} | "
              f"{grpo.get(2, {}).get('accuracy', 0.0):>10.1%} | {avg_grpo_acc:>6.1%} |")
        
        print()
        
        # Таблица формата
        print("📝 ПРАВИЛЬНЫЙ ФОРМАТ ОТВЕТОВ:")
        print("| Метод                  | Сложность 1 | Сложность 2 | Среднее |")
        print("|------------------------|-------------|-------------|---------|")
        
        # Zero-shot format
        avg_zero_fmt = sum(zero_shot[d]["format_score"] for d in zero_shot) / len(zero_shot) if zero_shot else 0.0
        print(f"| Zero-shot              | {zero_shot.get(1, {}).get('format_score', 0.0):>10.1%} | "
              f"{zero_shot.get(2, {}).get('format_score', 0.0):>10.1%} | {avg_zero_fmt:>6.1%} |")
        
        # Prompt Engineering format
        avg_pe_fmt = sum(prompt_eng[d]["format_score"] for d in prompt_eng) / len(prompt_eng) if prompt_eng else 0.0
        print(f"| Prompt Engineering     | {prompt_eng.get(1, {}).get('format_score', 0.0):>10.1%} | "
              f"{prompt_eng.get(2, {}).get('format_score', 0.0):>10.1%} | {avg_pe_fmt:>6.1%} |")
        
        # GRPO Trained format
        avg_grpo_fmt = sum(grpo[d]["format_score"] for d in grpo) / len(grpo) if grpo else 0.0
        print(f"| GRPO Trained           | {grpo.get(1, {}).get('format_score', 0.0):>10.1%} | "
              f"{grpo.get(2, {}).get('format_score', 0.0):>10.1%} | {avg_grpo_fmt:>6.1%} |")
        
        print()
        # Таблица строгого формата
        print("🔒 СТРОГИЙ ФОРМАТ <answer>X.XXX</answer>:")
        print("| Метод                  | Сложность 1 | Сложность 2 | Среднее |")
        print("|------------------------|-------------|-------------|---------|")
        avg_zero_strict = sum(zero_shot[d]["strict_format_score"] for d in zero_shot) / len(zero_shot) if zero_shot else 0.0
        print(f"| Zero-shot              | {zero_shot.get(1, {}).get('strict_format_score', 0.0):>10.1%} | "
              f"{zero_shot.get(2, {}).get('strict_format_score', 0.0):>10.1%} | {avg_zero_strict:>6.1%} |")
        avg_pe_strict = sum(prompt_eng[d]["strict_format_score"] for d in prompt_eng) / len(prompt_eng) if prompt_eng else 0.0
        print(f"| Prompt Engineering     | {prompt_eng.get(1, {}).get('strict_format_score', 0.0):>10.1%} | "
              f"{prompt_eng.get(2, {}).get('strict_format_score', 0.0):>10.1%} | {avg_pe_strict:>6.1%} |")
        avg_grpo_strict = sum(grpo[d]["strict_format_score"] for d in grpo) / len(grpo) if grpo else 0.0
        print(f"| GRPO Trained           | {grpo.get(1, {}).get('strict_format_score', 0.0):>10.1%} | "
              f"{grpo.get(2, {}).get('strict_format_score', 0.0):>10.1%} | {avg_grpo_strict:>6.1%} |")
        print()
        
        # Красивая диаграмма
        self.print_visual_chart(avg_zero_acc, avg_pe_acc, avg_grpo_acc, avg_zero_fmt, avg_pe_fmt, avg_grpo_fmt)
    
    def print_visual_chart(self, acc_zero, acc_pe, acc_grpo, fmt_zero, fmt_pe, fmt_grpo):
        """Создает красивую ASCII диаграмму результатов."""
        print("📈 ВИЗУАЛЬНАЯ ДИАГРАММА РЕЗУЛЬТАТОВ:")
        print("="*60)
        print()
        
        # Диаграмма точности
        print("🎯 ТОЧНОСТЬ ОТВЕТОВ:")
        self._print_bar_chart([
            ("Zero-shot", acc_zero),
            ("Prompt Eng", acc_pe), 
            ("GRPO Trained", acc_grpo)
        ])
        
        print()
        
        # Диаграмма формата
        print("📝 ПРАВИЛЬНЫЙ ФОРМАТ:")
        self._print_bar_chart([
            ("Zero-shot", fmt_zero),
            ("Prompt Eng", fmt_pe),
            ("GRPO Trained", fmt_grpo)
        ])
        
        print()
        # Диаграмма строгого формата
        print("🔒 СТРОГИЙ ФОРМАТ:")

    
    def _print_bar_chart(self, data):
        """Создает ASCII bar chart."""
        max_val = max(item[1] for item in data) if data else 0
        if max_val == 0:
            max_val = 1
        
        for name, value in data:
            bar_length = int(value * 30 / max_val) if max_val > 0 else 0
            bar = "█" * bar_length + "░" * (30 - bar_length)
            print(f"  {name:<12} │{bar}│ {value:.1%}")
    


def main():
    """Главная функция."""
    evaluator = Evaluator(
        baseline_model="unsloth/Qwen2.5-0.5B",
        trained_model_path="./dc_circuit_model_rl",
        samples_per_difficulty=5
    )
    
    results = evaluator.run_evaluation()
    
    # Сохраняем результаты в файл
    import json
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("💾 Результаты сохранены в evaluation_results.json")


if __name__ == "__main__":
    main()