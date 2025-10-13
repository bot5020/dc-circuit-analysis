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
import torch
from unsloth import FastLanguageModel

from base.data import Data
from dc_circuit.game import DCCircuitGame
from config import CircuitConfig, VerifierConfig, TrainingConfig
from base.utils import get_system_prompt


class Evaluator:
    """Оценщик для тестирования моделей."""
    
    # Константы
    DEFAULT_GPU_MEMORY_UTILIZATION = 0.55
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_SAMPLES_PER_DIFFICULTY = 20
    
    def __init__(
        self,
        baseline_model: str = "unsloth/qwen3-4b-instruct-2507-unsloth-bnb-4bit",
        trained_model_path: str = "./dc_circuit_model_rl",
        samples_per_difficulty: int = DEFAULT_SAMPLES_PER_DIFFICULTY
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
        
    def generate_test_data(self) -> Dict[int, List[Data]]:
        """Генерирует тестовые данные для всех уровней сложности.
        
        Returns:
            Словарь {difficulty: list_of_data}
        """
        print("\n📝 Генерация тестовых данных...")
        test_data = {}
        
        for difficulty in [1, 2]:  # Упрощенная система: только 2 уровня сложности
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
        """Загружает модель.
        
        Args:
            model_path: Путь к модели
            is_trained: True если это обученная модель с LoRA
        
        Returns:
            (model, tokenizer)
        """
        print(f"📥 Загрузка модели: {model_path}")
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=self.training_config.max_seq_length,
            load_in_4bit=True,
            dtype=None,
            fast_inference=True,
            gpu_memory_utilization=self.DEFAULT_GPU_MEMORY_UTILIZATION
        )
        
        # Режим инференса
        FastLanguageModel.for_inference(model)
        
        print(f"  ✓ Модель загружена\n")
        return model, tokenizer
    
    def generate_answer(
        self, 
        model, 
        tokenizer, 
        question: str, 
        use_system_prompt: bool = True
    ) -> str:
        """Генерирует ответ модели на вопрос.
        
        Args:
            model: Модель
            tokenizer: Токенизатор
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
        
        # Применяем chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Генерируем ответ
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=self.training_config.max_completion_length,
            temperature=self.DEFAULT_TEMPERATURE,  
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Убираем промпт из ответа
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        return response
    
    def evaluate_model_on_data(
        self,
        model,
        tokenizer,
        test_data: Dict[int, List[Data]],
        method_name: str,
        use_system_prompt: bool = True
    ) -> Dict[int, Dict[str, float]]:
        """Оценивает модель на тестовых данных.
        
        Args:
            model: Модель
            tokenizer: Токенизатор
            test_data: Тестовые данные
            method_name: Название метода для вывода
            use_system_prompt: Использовать ли системный промпт
        
        Returns:
            Словарь {difficulty: {"accuracy": float, "format_score": float}}
        """
        print(f"🧪 Тестирование: {method_name}")
        results = {}
        
        for difficulty, data_list in sorted(test_data.items()):
            correct = 0
            format_correct = 0
            total = len(data_list)
            
            for i, data in enumerate(data_list):
                # Генерируем ответ
                response = self.generate_answer(
                    model, tokenizer, data.question, use_system_prompt
                )
                
                # Проверяем правильность
                is_correct = self.game.verify(data, response)
                if is_correct:
                    correct += 1
                
                # Проверяем формат ответа
                has_think = "<think>" in response
                has_answer = "<answer>" in response
                if has_think and has_answer:
                    format_correct += 1
                
                # Прогресс
                if (i + 1) % 5 == 0 or (i + 1) == total:
                    print(f"  Сложность {difficulty}: {i+1}/{total} задач...", end='\r')
            
            accuracy = correct / total if total > 0 else 0.0
            format_score = format_correct / total if total > 0 else 0.0
            results[difficulty] = {
                "accuracy": accuracy,
                "format_score": format_score
            }
            print(f"  Сложность {difficulty}: {correct}/{total} = {accuracy:.1%} | Формат: {format_correct}/{total} = {format_score:.1%}    ")
        
        # Общие показатели
        avg_accuracy = sum(r["accuracy"] for r in results.values()) / len(results) if results else 0.0
        avg_format = sum(r["format_score"] for r in results.values()) / len(results) if results else 0.0
        print(f"  📊 Средняя точность: {avg_accuracy:.1%} | Средний формат: {avg_format:.1%}\n")
        
        return results
    
    def run_evaluation(self):
        """Запускает полную оценку всех трех методов."""
        print("================================================")
        print("                ОЦЕНКА МОДЕЛЕЙ DC CIRCUIT ANALYSIS")
        print("================================================")
        
        # 1. Генерация тестовых данных
        test_data = self.generate_test_data()
        
        # 2. Загрузка baseline модели
        baseline_model, baseline_tokenizer = self.load_model(
            self.baseline_model_name, 
            is_trained=False
        )
        
        # 3. Zero-shot оценка (без специального промпта)
        print("-"*70)
        zero_shot_results = self.evaluate_model_on_data(
            baseline_model,
            baseline_tokenizer,
            test_data,
            "Zero-shot (no system prompt)",
            use_system_prompt=False
        )
        
        # 4. Prompt Engineering оценка (с системным промптом)
        print("-"*70)
        prompt_eng_results = self.evaluate_model_on_data(
            baseline_model,
            baseline_tokenizer,
            test_data,
            "Prompt Engineering (with system prompt)",
            use_system_prompt=True
        )
        
        # Очистка памяти
        del baseline_model, baseline_tokenizer
        torch.cuda.empty_cache()
        
        # 5. GRPO Trained оценка (если модель существует)
        print("-"*70)
        grpo_results = {}
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
                use_system_prompt=True
            )
            
            del trained_model, trained_tokenizer
            torch.cuda.empty_cache()
        else:
            print(f"⚠️  Обученная модель не найдена: {self.trained_model_path}")
            print(f"   Пропускаем оценку GRPO Trained\n")
            grpo_results = {1: 0.0, 2: 0.0}
        
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
        
        # Сравнение улучшений
        print("📊 УЛУЧШЕНИЯ:")
        if acc_pe > acc_zero:
            print(f"✅ Prompt Engineering улучшил точность на {acc_pe - acc_zero:.1%}")
        else:
            print(f"❌ Prompt Engineering снизил точность на {acc_zero - acc_pe:.1%}")
            
        if acc_grpo > acc_pe:
            print(f"🚀 GRPO обучение улучшил точность на {acc_grpo - acc_pe:.1%}")
        else:
            print(f"⚠️  GRPO обучение снизил точность на {acc_pe - acc_grpo:.1%}")
            
        if fmt_pe > fmt_zero:
            print(f"✅ Prompt Engineering улучшил формат на {fmt_pe - fmt_zero:.1%}")
        else:
            print(f"❌ Prompt Engineering снизил формат на {fmt_zero - fmt_pe:.1%}")
            
        if fmt_grpo > fmt_pe:
            print(f"🚀 GRPO обучение улучшил формат на {fmt_grpo - fmt_pe:.1%}")
        else:
            print(f"⚠️  GRPO обучение снизил формат на {fmt_pe - fmt_grpo:.1%}")
    
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
        baseline_model="unsloth/qwen3-4b-instruct-2507-unsloth-bnb-4bit",
        trained_model_path="./dc_circuit_model_rl",
        samples_per_difficulty=20 
    )
    
    results = evaluator.run_evaluation()
    
    # Сохраняем результаты в файл
    import json
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("💾 Результаты сохранены в evaluation_results.json")


if __name__ == "__main__":
    main()