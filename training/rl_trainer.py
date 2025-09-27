"""
ПОЛНАЯ СИСТЕМА RL ОБУЧЕНИЯ ДЛЯ АНАЛИЗА ЭЛЕКТРИЧЕСКИХ ЦЕПЕЙ

Реализует GRPO (Generative Reward Policy Optimization) обучение
с использованием unsloth, TRL и множественных reward functions.
"""

import os
import sys
import json
import argparse
import re
from typing import Dict, List
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
from accelerate import Accelerator

from training.datasets import DCCircuitIterableDataset
from dc_circuit.game import DCCircuitGame


@dataclass
class TrainingConfig:
    """Конфигурация обучения"""
    model_name: str = "unsloth/Qwen2.5-3B-Instruct"
    output_dir: str = "./dc_circuit_model_rl"
    max_seq_length: int = 1024
    load_in_4bit: bool = True
    fast_inference: bool = True
    gpu_memory_utilization: float = 0.9

    # GRPO параметры
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_generations: int = 8
    learning_rate: float = 5e-6
    max_steps: int = 250
    logging_steps: int = 1
    save_steps: int = 250
    max_prompt_length: int = 256
    max_completion_length: int = 200
    temperature: float = 0.7
    beta: float = 0.04

    # Multi-GPU параметры
    num_gpus: int = 2  # Количество GPU (1 или 2 для T4)
    gpu_memory_utilization: float = 0.9

    # LoRA параметры (расширенные)
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None
    use_gradient_checkpointing: str = "unsloth"

    # Оптимизация
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    weight_decay: float = 0.1
    warmup_ratio: float = 0.1
    max_grad_norm: float = 0.1
    optim: str = "adamw_8bit"

    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]


class DCCircuitRLTrainer:
    """
    Полная система RL обучения для анализа электрических цепей

    Использует GRPO алгоритм с reward function на основе точности ответов
    """

    def __init__(self, config: TrainingConfig):
        """
        Инициализация RL тренера

        Args:
            config: Конфигурация обучения
        """
        self.config = config
        self.accelerator = Accelerator()

        print("🔧 Инициализация RL тренера...")
        print(f"📋 Модель: {config.model_name}")
        print(f"🎯 Выходная директория: {config.output_dir}")

        # Инициализируем компоненты
        self.game = DCCircuitGame()
        self.tokenizer = None
        self.model = None
        self.trainer = None

    def setup_model_and_tokenizer(self):
        """
        Настраивает модель и токенизатор

        🔧 ШАГИ:
        1. Загружаем базовую модель и токенизатор (unsloth или стандартно)
        2. Добавляем LoRA адаптер
        3. Настраиваем для обучения
        """
        print("🔄 Загрузка модели и токенизатора...")

        if UNSLOTH_AVAILABLE:
            print("🚀 Используем unsloth для ускоренной загрузки...")

            # Загружаем модель с unsloth
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.model_name,
                max_seq_length=self.config.max_seq_length,
                load_in_4bit=self.config.load_in_4bit,
                fast_inference=self.config.fast_inference,
                max_lora_rank=self.config.lora_r,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
            )

            # Добавляем LoRA адаптер
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=self.config.lora_r,
                target_modules=self.config.lora_target_modules,
                lora_alpha=self.config.lora_alpha,
                use_gradient_checkpointing=self.config.use_gradient_checkpointing,
                random_state=3407,
            )

        else:
            print("⚠️ unsloth недоступен, используем стандартный подход...")
            print("💡 Для лучшей производительности установите: pip install unsloth")

            # Загружаем токенизатор
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )

            # Добавляем pad token если его нет
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Загружаем модель с multi-GPU поддержкой
            if self.config.num_gpus > 1:
                print(f"🔧 Настройка для {self.config.num_gpus} GPU...")

                # Для multi-GPU используем device_map
                device_map = "auto" if self.config.num_gpus == 1 else "balanced"

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
                    device_map=device_map,
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
                    max_memory={i: f"{int(self.config.gpu_memory_utilization * 16)}GiB" for i in range(self.config.num_gpus)}
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2" if torch.cuda.is_available() else None
                )

            # Настраиваем LoRA
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                task_type="CAUSAL_LM"
            )

            self.model = get_peft_model(self.model, lora_config)

    def get_system_prompt(self) -> str:
        """
        Возвращает системный промпт для модели

        Returns:
            Системный промпт с инструкциями
        """
        return (
            "You are an expert circuit analysis engineer. Solve electrical circuit problems using physics laws.\n\n"
            "Respond in the following format:\n"
            "<think>Reason step by step briefly.</think>\n"
            "<answer>Return ONLY the final number with exactly 3 decimal places (e.g., 1.234), no units.</answer>"
        )

    def extract_answer(self, text: str) -> str:
        """Извлекает ответ из <answer> тегов"""
        answer_start = text.find("<answer>")
        answer_end = text.find("</answer>")

        if answer_start != -1 and answer_end != -1:
            return text[answer_start + 8:answer_end].strip()
        return ""

    def correctness_reward_func(self, completions, prompts, answer, **kwargs) -> List[float]:
        """
        Основная reward function - точность ответа

        Args:
            completions: Сгенерированные ответы модели
            prompts: Промпты с правильными ответами
            answer: Правильные ответы
            **kwargs: Дополнительные параметры

        Returns:
            Список вознаграждений (2.0 за правильный ответ, 0.0 за неправильный)
        """
        responses = [completion[0]['content'] for completion in completions]
        extracted_responses = [self.extract_answer(r) for r in responses]

        print('-' * 20, f"Question:\n{prompts[0][-1]['content']}",
              f"\nAnswer:\n{answer[0]}",
              f"\nResponse:\n{responses[0]}",
              f"\nExtracted:\n{extracted_responses[0]}")

        return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

    def numeric_reward_func(self, completions, **kwargs) -> List[float]:
        """Reward за численный ответ"""
        responses = [completion[0]['content'] for completion in completions]
        extracted_responses = [self.extract_answer(r) for r in responses]
        return [0.5 if r.replace('.', '').replace('-', '').isdigit() else 0.0 for r in extracted_responses]

    def strict_format_reward_func(self, completions, **kwargs) -> List[float]:
        """Reward за правильный формат ответа"""
        pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, r, re.DOTALL) for r in responses]
        return [0.5 if match else 0.0 for match in matches]

    def soft_format_reward_func(self, completions, **kwargs) -> List[float]:
        """Reward за мягкий формат (теги могут быть в любом порядке)"""
        pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, r, re.DOTALL) for r in responses]
        return [0.5 if match else 0.0 for match in matches]

    def xml_count_reward_func(self, completions, **kwargs) -> List[float]:
        """Reward за правильное количество XML тегов"""
        contents = [completion[0]["content"] for completion in completions]

        def count_xml(text) -> float:
            count = 0.0
            if text.count("<reasoning>\n") == 1:
                count += 0.125
            if text.count("\n</reasoning>\n") == 1:
                count += 0.125
            if text.count("\n<answer>\n") == 1:
                count += 0.125
            if text.count("\n</answer>") == 1:
                count += 0.125
            return count

        return [count_xml(c) for c in contents]

    def get_reward_functions(self) -> List:
        """Возвращает список всех reward functions"""
        return [
            self.xml_count_reward_func,
            self.soft_format_reward_func,
            self.strict_format_reward_func,
            self.numeric_reward_func,
            self.correctness_reward_func,
        ]

    def create_training_dataset(self):
        """
        Создает датасет для обучения

        Returns:
            Итерируемый датасет с обучающими данными
        """
        print("📚 Создание обучающего датасета...")

        return DCCircuitIterableDataset(
            difficulties=[1, 3, 5, 7],  # Разные уровни сложности
            samples_per_difficulty=10  # Образцов на каждый уровень
        )

    def format_prompt(self, example: Dict[str, str]) -> str:
        """
        Форматирует пример в промпт для обучения

        Args:
            example: Пример с вопросом и ответом

        Returns:
            Отформатированный промпт
        """
        question = example["question"]
        answer = example["answer"]

        system_prompt = self.get_system_prompt()

        return (
            f"{system_prompt}\n\n"
            f"Question: {question}\n"
            f"Answer: <gold>{answer}</gold>"
        )

    def setup_trainer(self):
        """
        Настраивает GRPO тренер

        🔧 ШАГИ:
        1. Создает датасет
        2. Настраивает training arguments
        3. Создает GRPOTrainer с множественными reward functions
        """
        print("🎯 Настройка GRPO тренера...")

        # Создаем датасет
        train_dataset = self.create_training_dataset()

        # Настраиваем training arguments (как в примере пользователя)
        training_args = GRPOConfig(
            use_vllm=True,  # Используем vLLM для быстрого inference
            learning_rate=self.config.learning_rate,
            adam_beta1=self.config.adam_beta1,
            adam_beta2=self.config.adam_beta2,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type="cosine",
            optim=self.config.optim,
            logging_steps=self.config.logging_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_generations=self.config.num_generations,
            max_prompt_length=self.config.max_prompt_length,
            max_completion_length=self.config.max_completion_length,
            max_steps=self.config.max_steps,
            save_steps=self.config.save_steps,
            max_grad_norm=self.config.max_grad_norm,
            report_to="none",  # Можно использовать Weights & Biases
            output_dir=self.config.output_dir,
        )

        # Получаем все reward functions
        reward_functions = self.get_reward_functions()

        # Создаем GRPO тренер
        self.trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=reward_functions,
            args=training_args,
            train_dataset=train_dataset,
        )


    def train(self):
        """
        Запускает обучение

        🔄 ПРОЦЕСС ОБУЧЕНИЯ:
        1. Подготовка модели и данных
        2. Цикл GRPO: generate → reward → optimize
        3. Сохранение прогресса
        4. Финальное сохранение модели
        """
        try:
            # Запускаем обучение
            self.trainer.train()

            # Сохраняем финальную модель
            self.trainer.save_model(self.config.output_dir)
            self.tokenizer.save_pretrained(self.config.output_dir)

            print("\n✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
            print(f"💾 Модель сохранена в: {self.config.output_dir}")

            # Сохраняем статистику
            self.save_training_stats()

        except KeyboardInterrupt:
            print("\n⏹️ Обучение прервано пользователем")
            self.save_checkpoint()
        except Exception as e:
            print(f"\n❌ Ошибка обучения: {e}")
            self.save_checkpoint()

    def save_training_stats(self):
        """
        Сохраняет статистику обучения
        """
        try:
            stats = {
                "config": {
                    "model_name": self.config.model_name,
                    "max_steps": self.config.max_steps,
                    "learning_rate": self.config.learning_rate,
                    "lora_r": self.config.lora_r,
                    "lora_alpha": self.config.lora_alpha,
                },
                "training_completed": True,
                "output_dir": self.config.output_dir
            }

            os.makedirs(self.config.output_dir, exist_ok=True)
            with open(f"{self.config.output_dir}/training_stats.json", "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)

            print(f"📊 Статистика сохранена: {self.config.output_dir}/training_stats.json")

        except Exception as e:
            print(f"❌ Ошибка сохранения статистики: {e}")

    def save_checkpoint(self):
        """
        Сохраняет checkpoint при прерывании
        """
        try:
            checkpoint_dir = f"{self.config.output_dir}/checkpoint"
            os.makedirs(checkpoint_dir, exist_ok=True)

            self.trainer.save_model(checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)

            print(f"💾 Checkpoint сохранен: {checkpoint_dir}")

        except Exception as e:
            print(f"❌ Ошибка сохранения checkpoint: {e}")


def main():
    """
    Основная функция для запуска GRPO обучения с unsloth
    """
    parser = argparse.ArgumentParser()

    # Модель и базовая конфигурация
    parser.add_argument("--model", default="unsloth/Qwen2.5-3B-Instruct",
                       help="Название базовой модели (рекомендуется unsloth)")
    parser.add_argument("--output-dir", default="./dc_circuit_model_rl",
                       help="Директория для сохранения модели")
    parser.add_argument("--max-seq-length", type=int, default=1024,
                       help="Максимальная длина последовательности")

    # Обучение
    parser.add_argument("--max-steps", type=int, default=250,
                       help="Максимальное количество шагов обучения")
    parser.add_argument("--learning-rate", type=float, default=5e-6,
                       help="Скорость обучения")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Размер батча")
    parser.add_argument("--gradient-accumulation", type=int, default=1,
                       help="Gradient accumulation steps")
    parser.add_argument("--num-generations", type=int, default=8,
                       help="Количество генераций на шаг")

    # LoRA параметры
    parser.add_argument("--lora-r", type=int, default=64,
                       help="LoRA rank (чем больше - тем умнее, но медленнее)")
    parser.add_argument("--lora-alpha", type=int, default=64,
                       help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                       help="LoRA dropout")

    # vLLM параметры
    parser.add_argument("--load-in-4bit", action="store_true", default=True,
                       help="Загружать модель в 4-bit для экономии памяти")
    parser.add_argument("--fast-inference", action="store_true", default=True,
                       help="Использовать vLLM для быстрого inference")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                       help="Использование GPU памяти")
    parser.add_argument("--num-gpus", type=int, default=1,
                       help="Количество GPU (1 или 2 для T4)")

    # GRPO параметры
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--beta", type=float, default=0.04,
                       help="KL divergence coefficient")

    args = parser.parse_args()


    # Создаем конфигурацию
    config = TrainingConfig(
        model_name=args.model,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        fast_inference=args.fast_inference,
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_gpus=args.num_gpus,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_generations=args.num_generations,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        temperature=args.temperature,
        beta=args.beta
    )

    # Создаем и запускаем тренер
    trainer = DCCircuitRLTrainer(config)

    try:
        # Настраиваем модель
        trainer.setup_model_and_tokenizer()

        # Настраиваем тренер
        trainer.setup_trainer()

        # Запускаем обучение
        trainer.train()

    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
