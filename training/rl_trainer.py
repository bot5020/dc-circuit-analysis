"""GRPO обучение для анализа электрических цепей.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from typing import List

from torch.utils.data import Dataset
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig

from base.utils import extract_answer
from base.data import Data
from dc_circuit.game import DCCircuitGame
from dc_circuit.verifier import DCCircuitVerifier


# ============================================================================
# КОНФИГУРАЦИЯ ОБУЧЕНИЯ
# ============================================================================

@dataclass
class TrainingConfig:
    """Настройки обучения"""
    
    # Модель
    model_name: str = "Qwen/Qwen3-0.6B"
    output_dir: str = "./dc_circuit_model_rl"
    max_seq_length: int = 8192
    
    # LoRA
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    
    # Обучение
    learning_rate: float = 1e-5
    max_steps: int = 500
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    num_generations: int = 2
    save_steps: int = 100
    
    # Dataset
    difficulties: List[int] = None
    samples_per_difficulty: int = 500
    
    def __post_init__(self):
        if self.difficulties is None:
            self.difficulties = [1, 2, 3, 4, 5]


# Глобальный конфиг
CONFIG = TrainingConfig()


# ============================================================================
# DATASET
# ============================================================================

class DCCircuitDataset(Dataset):
    """Dataset для GRPO обучения"""

    def __init__(self, config: TrainingConfig):
        self.game = DCCircuitGame()
        self.config = config
        self._data_cache = None

    def _generate_data(self) -> List[dict]:
        """Генерирует данные через game.generate()"""
        if self._data_cache is None:
            all_data = []
            for difficulty in self.config.difficulties:
                print(f"Генерируем данные сложности {difficulty}...")
                data_list = self.game.generate(
                    num_of_questions=self.config.samples_per_difficulty,
                    difficulty=difficulty,
                    max_attempts=30
                )
                print(f"✅ Сгенерировано {len(data_list)} элементов")
                
                all_data.extend([{
                    "prompt": [
                        {"role": "user", "content": f"{data.question}\n<gold>{data.answer}</gold>"}
                    ],
                    "question": data.question,
                    "answer": data.answer,
                    "difficulty": data.difficulty
                } for data in data_list])
            
            self._data_cache = all_data
            print(f"📊 Всего в датасете: {len(all_data)} элементов")
        
        return self._data_cache

    def __len__(self) -> int:
        return len(self._generate_data())

    def __getitem__(self, idx: int) -> dict:
        data = self._generate_data()
        return data[idx]


# ============================================================================
# GRPO ТРЕНЕР
# ============================================================================

class DCCircuitRLTrainer:
    """Тренер для GRPO обучения на задачах анализа DC цепей"""

    def __init__(self, config: TrainingConfig = None):
        """Инициализирует тренер с конфигурацией"""
        self.config = config or CONFIG
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self._verifier = None
        
        print(f"🚀 Инициализация GRPO тренера")
        print(f"   Модель: {self.config.model_name}")
        print(f"   Выход: {self.config.output_dir}")
        print(f"   Шаги: {self.config.max_steps}")

    def setup_model(self):
        """Загружает и настраивает модель с LoRA."""
        print("\n📦 Загрузка модели с unsloth...")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=True,
            fast_inference=False
        )
        
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        
        self.model.train()
        print("✅ Модель загружена")
        self.model.print_trainable_parameters()

    def reward_function(self, prompts, completions, **kwargs) -> List[float]:
        """Reward на основе verifier.
        
        Использует DCCircuitVerifier.get_accuracy_score():
        - 1.0 за ошибку <= 0.1% -> reward = 2.0
        - 0.75 за ошибку <= 0.2% -> reward = 1.5
        - 0.5 за ошибку <= 0.3% -> reward = 1.0
        - 0.25 за ошибку <= 0.5% -> reward = 0.5
        - 0.0 за ошибку > 0.5% -> reward = 0.0
        """
        if self._verifier is None:
            self._verifier = DCCircuitVerifier()
        
        # Нормализуем completions
        responses = []
        for item in completions:
            if isinstance(item, str):
                responses.append(item)
            elif isinstance(item, dict):
                responses.append(item.get("content", ""))
            elif isinstance(item, list) and item:
                candidate = item[0]
                responses.append(candidate.get("content", "") if isinstance(candidate, dict) else str(candidate))
            else:
                responses.append(str(item))
        
        # Детальное логирование на ключевых шагах
        trainer_state = kwargs.get('trainer_state')
        step = getattr(trainer_state, 'global_step', 0) if trainer_state else 0
        
        # Логируем на шагах: 1, 2, 3, 5, 10, 20, 40, 60, ...
        should_log = (step in [1, 2, 3, 5, 10]) or (step > 10 and step % 20 == 0)
        
        if should_log and prompts and responses:
            print("\n" + "="*80)
            print(f"📊 ДЕТАЛЬНОЕ ЛОГИРОВАНИЕ | Step {step}")
            print("="*80)
            
            prompt_content = prompts[0] if isinstance(prompts[0], str) else \
                           (prompts[0][-1]['content'] if prompts[0] else "")
            gold_start = prompt_content.find("<gold>")
            gold_end = prompt_content.find("</gold>")
            
            if gold_start != -1 and gold_end != -1:
                correct_answer = prompt_content[gold_start + 6:gold_end].strip()
                
                # Показываем первый ответ модели
                raw_response = responses[0] if responses else ""
                extracted = extract_answer(raw_response) if responses else ""
                
                print(f"\n✅ Правильный ответ: {correct_answer}")
                print(f"\n🤖 Ответ модели (raw, первые 200 символов):")
                print(f"   {raw_response[:200]}...")
                print(f"\n🔍 Извлечённый ответ: '{extracted}'")
                
                # Вычисляем reward для первого ответа
                if extracted:
                    data = Data(question="", answer=correct_answer, difficulty=1, metadata={})
                    try:
                        score = self._verifier.get_accuracy_score(data, raw_response)
                        reward = score * 2.0
                        print(f"\n💰 Accuracy Score: {score:.2f} → Reward: {reward:.2f}")
                    except:
                        print(f"\n💰 Reward: 0.0 (ошибка вычисления)")
                else:
                    print(f"\n💰 Reward: 0.0 (не удалось извлечь ответ)")
                
                print("="*80 + "\n")
        
        if not prompts:
            return [0.0] * len(responses)
        
        # Извлекаем правильный ответ
        prompt_content = prompts[0] if isinstance(prompts[0], str) else \
                        (prompts[0][-1]['content'] if prompts and prompts[0] else "")
        
        gold_start = prompt_content.find("<gold>")
        gold_end = prompt_content.find("</gold>")
        
        if gold_start == -1 or gold_end == -1:
            return [0.0] * len(responses)
        
        correct_answer = prompt_content[gold_start + 6:gold_end].strip()
        
        # Создаём Data для verifier
        data = Data(question="", answer=correct_answer, difficulty=1, metadata={})
        
        # Вычисляем reward через verifier
        rewards = []
        for response in responses:
            try:
                accuracy_score = self._verifier.get_accuracy_score(data, response)
                reward = accuracy_score * 2.0  # Масштабируем для GRPO
            except Exception:
                reward = 0.0
            rewards.append(reward)
        
        return rewards

    def setup_trainer(self):
        """Настраивает GRPO тренер."""
        print("\n  Настройка GRPO тренера...")
        
        # Создаём датасет
        train_dataset = DCCircuitDataset(self.config)
        
        # GRPO конфигурация
        training_args = GRPOConfig(
            use_vllm=False,
            learning_rate=self.config.learning_rate,
            adam_beta1=0.9,
            adam_beta2=0.99,
            weight_decay=0.1,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            logging_steps=1,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_generations=self.config.num_generations,
            max_prompt_length=4096,
            max_completion_length=8192,
            max_steps=self.config.max_steps,
            save_steps=self.config.save_steps,
            max_grad_norm=0.1,
            report_to="none",
            output_dir=self.config.output_dir,
            temperature=0.7,
        )
        
        # Создаём тренер
        self.model.train()
        self.trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=[self.reward_function],
            args=training_args,
            train_dataset=train_dataset,
        )
        
        print("✅ GRPO тренер настроен")

    def train(self):
        """Запускает обучение."""
        print("\n🚀 Начинаем GRPO обучение...")
        
        try:
            self.model.train()
            self.trainer.train()
            
            # Сохраняем
            self.trainer.save_model(self.config.output_dir)
            self.tokenizer.save_pretrained(self.config.output_dir)
            
            print(f"\n✅ Обучение завершено!")
            print(f"💾 Модель: {self.config.output_dir}")
            
        except KeyboardInterrupt:
            print("\n⚠️  Прервано пользователем")
            checkpoint_dir = f"{self.config.output_dir}/checkpoint"
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.trainer.save_model(checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)
            print(f"💾 Checkpoint: {checkpoint_dir}")
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")
            raise

    def run(self):
        """Полный цикл обучения."""
        self.setup_model()
        self.setup_trainer()
        self.train()


# ============================================================================
# ЗАПУСК
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("🎯 DC CIRCUIT GRPO ОБУЧЕНИЕ")
    print("="*80)
    print("\nТекущий конфиг:")
    print(f"  Модель: {CONFIG.model_name}")
    print(f"  Шаги: {CONFIG.max_steps}")
    print(f"  Batch size: {CONFIG.batch_size}")
    print(f"  Learning rate: {CONFIG.learning_rate}")
    print(f"  Сложности: {CONFIG.difficulties}")
    print(f"  Датасет: {CONFIG.samples_per_difficulty} на сложность")
    print("="*80)
    
    trainer = DCCircuitRLTrainer(CONFIG)
    trainer.run()
