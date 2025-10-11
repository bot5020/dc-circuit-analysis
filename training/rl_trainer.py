"""GRPO обучение для анализа электрических цепей.
"""

import os
import sys

# ============================================================================
# vLLM ОТКЛЮЧЕН ДЛЯ T4 15GB - СЛИШКОМ МНОГО ПАМЯТИ!
# Unsloth игнорирует переменные окружения, поэтому без vLLM стабильнее!
# ============================================================================

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
    model_name: str = "unsloth/Qwen2.5-3B-Instruct"
    output_dir: str = "./dc_circuit_model_rl"
    max_seq_length: int = 2048  # Уменьшили для KV cache!
    
    # LoRA
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    
    # Обучение
    learning_rate: float = 1e-5
    max_steps: int = 50  # Больше шагов для T4
    batch_size: int = 1  
    gradient_accumulation_steps: int = 24  # Компенсируем (эфф=24)
    num_generations: int = 1  # Для vLLM только 1! 
    save_steps: int = 25 
    
    # Dataset (УВЕЛИЧИЛИ ДЛЯ ЛУЧШЕГО ОБУЧЕНИЯ)
    difficulties: List[int] = None
    samples_per_difficulty: int = 25  # Уменьшили с 50 для скорости!
    
    def __post_init__(self):
        if self.difficulties is None:
            self.difficulties = [1, 3, 5]  # Простой, средний, сложный - весь диапазон!


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
                        {"role": "system", "content": "You are an expert in DC circuit analysis. IMPORTANT: Show ALL your calculations and reasoning steps, but at the very end, provide the FINAL ANSWER in this exact format: 'The answer is: [numerical_value]'. Do not include units in the final answer format, only the number."},
                        {"role": "user", "content": f"{data.question}\n<gold>{float(data.answer):.3f}</gold>"}
                    ],
                    "question": data.question,
                    "answer": f"{float(data.answer):.3f}",
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

    def setup_model(self):
        """Загружает модель с LoRA"""        
        print(f"📦 Загрузка модели {self.config.model_name}...")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=True,  
            dtype=None,  # Auto
            fast_inference=True,  # vLLM ВКЛЮЧЕН!
            max_lora_rank=self.config.lora_r, 
            gpu_memory_utilization=0.5  
        )
        
        # Установка базового chat_template если его нет
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ message['content'] }}{% endif %}{% endfor %}"
        
        # LoRA обучение
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
        self.model.print_trainable_parameters()

    def _extract_prompt_content(self, prompts) -> str:
        """Извлекает содержимое промпта из разных форматов.
        
        Args:
            prompts: Список промптов (может быть str или list of dicts)
        
        Returns:
            Содержимое промпта как строка
        """
        if not prompts:
            return ""
        if isinstance(prompts[0], str):
            return prompts[0]
        return prompts[0][-1]['content'] if prompts[0] else ""
    
    def _normalize_completions(self, completions) -> List[str]:
        """Нормализует completions в список строк.
        
        Args:
            completions: Ответы модели в разных форматах
        
        Returns:
            Список строковых ответов
        """
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
        return responses
    
    def _should_log_step(self, step: int) -> bool:
        """Определяет нужно ли логировать текущий шаг.
        
        Args:
            step: Номер шага
        
        Returns:
            True если нужно логировать
        """
        return (step in [1, 2, 3, 5, 10]) or (step > 10 and step % 20 == 0)
    
    def _extract_gold_answer(self, prompt_content: str) -> str:
        """Извлекает правильный ответ из промпта.
        
        Args:
            prompt_content: Содержимое промпта
        
        Returns:
            Правильный ответ или пустая строка
        """
        gold_start = prompt_content.find("<gold>")
        gold_end = prompt_content.find("</gold>")
        
        if gold_start != -1 and gold_end != -1:
            return prompt_content[gold_start + 6:gold_end].strip()
        return ""
    
    def _log_detailed_metrics(self, step: int, correct_answer: str, 
                             raw_response: str, extracted: str):
        """Выводит детальное логирование метрик.
        
        Args:
            step: Номер шага
            correct_answer: Правильный ответ
            raw_response: Сырой ответ модели
            extracted: Извлечённый ответ
        """
        print("\n" + "="*80)
        print(f"📊 ДЕТАЛЬНОЕ ЛОГИРОВАНИЕ | Step {step}")
        print("="*80)
        print(f"\n✅ Правильный ответ: {correct_answer}")
        print(f"\n🤖 Ответ модели (raw, первые 200 символов):")
        print(f"   {raw_response[:200]}...")
        print(f"\n🔍 Извлечённый ответ: '{extracted}'")
        
        # Вычисляем reward для логирования
        if extracted:
            data = Data(question="", answer=correct_answer, difficulty=1, metadata={})
            try:
                score = self._verifier.get_accuracy_score(data, raw_response)
                reward = score * 2.0
                
                # Детальный расчёт погрешности
                try:
                    model_val = float(extracted)
                    correct_val = float(correct_answer)
                    rounded_correct = round(correct_val, 3)
                    rounded_model = round(model_val, 3)
                    
                    if abs(rounded_correct) < 1e-12:
                        rel_error = abs(rounded_model - rounded_correct)
                        rel_error_percent = rel_error * 100
                    else:
                        rel_error = abs(rounded_model - rounded_correct) / abs(rounded_correct)
                        rel_error_percent = rel_error * 100
                    
                    print(f"\n🔬 Детальный расчёт:")
                    print(f"   Правильно (raw):       {correct_val}")
                    print(f"   Правильно (округл.):   {rounded_correct}")
                    print(f"   Модель (raw):          {model_val}")
                    print(f"   Модель (округл.):      {rounded_model}")
                    print(f"   Абс. погрешность:      {abs(rounded_model - rounded_correct):.6f}")
                    print(f"   Отн. погрешность:      {rel_error:.6f} ({rel_error_percent:.2f}%)")
                    print(f"   Пороги: 0.1%={0.001}, 0.2%={0.002}, 0.3%={0.003}, 0.5%={0.005}")
                except:
                    pass
                
                print(f"\n💰 Accuracy Score: {score:.2f} → Reward: {reward:.2f}")
            except Exception as e:
                print(f"\n💰 Reward: 0.0 (ошибка: {e})")
        else:
            print(f"\n💰 Reward: 0.0 (не удалось извлечь ответ)")
    
    
    def _calculate_rewards(self, correct_answer: str, responses: List[str]) -> List[float]:
        """Вычисляет rewards для всех ответов.
        
        Args:
            correct_answer: Правильный ответ
            responses: Список ответов модели
        
        Returns:
            Список rewards
        """
        data = Data(question="", answer=correct_answer, difficulty=1, metadata={})
        rewards = []
        
        for response in responses:
            try:
                accuracy_score = self._verifier.get_accuracy_score(data, response)
                reward = accuracy_score * 2.0
            except Exception as e:
                print(f"⚠️  Ошибка в _calculate_rewards: {e}")
                reward = 0.0
            rewards.append(reward)
        
        return rewards
    
    def reward_function(self, prompts, completions, **kwargs) -> List[float]:
        """Reward на основе verifier.

        Args:
            prompts: Список промптов
            completions: Список ответов модели
            **kwargs: Дополнительные параметры (trainer_state)
        
        Returns:
            Список rewards для каждого ответа
        """
        # Инициализация verifier
        if self._verifier is None:
            self._verifier = DCCircuitVerifier()
        
        # Нормализация ответов
        responses = self._normalize_completions(completions)
        
        # Детальное логирование на ключевых шагах
        trainer_state = kwargs.get('trainer_state')
        step = getattr(trainer_state, 'global_step', 0) if trainer_state else 0
        
        if self._should_log_step(step) and prompts and responses:
            prompt_content = self._extract_prompt_content(prompts)
            correct_answer = self._extract_gold_answer(prompt_content)
            
            if correct_answer:
                raw_response = responses[0] if responses else ""
                extracted = extract_answer(raw_response) if responses else ""
                self._log_detailed_metrics(step, correct_answer, raw_response, extracted)
        
        # Извлекаем правильный ответ и вычисляем rewards
        prompt_content = self._extract_prompt_content(prompts)
        correct_answer = self._extract_gold_answer(prompt_content)
        
        if not correct_answer:
            return [0.0] * len(responses)
        
        return self._calculate_rewards(correct_answer, responses)

    def setup_trainer(self):
        """Настраивает GRPO тренер."""
        train_dataset = DCCircuitDataset(self.config)
        
        training_args = GRPOConfig(
            use_vllm=True,  # ВКЛЮЧИЛИ! 
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
            max_prompt_length=1024,  # Уменьшили для памяти!
            max_completion_length=1024,  # Уменьшили для памяти!
            max_steps=self.config.max_steps,
            save_steps=self.config.save_steps,
            max_grad_norm=0.1,
            report_to="none",
            output_dir=self.config.output_dir,
            temperature=0.7,
            repetition_penalty=1.1,
        )
        
        # Создание тренера
        self.model.train()
        self.trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=[self.reward_function],
            args=training_args,
            train_dataset=train_dataset,
        )
        

    def train(self):
        """Запускает обучение."""
        try:
            self.model.train()
            self.trainer.train()
            
            # Сохранение модели
            self.trainer.save_model(self.config.output_dir)
            self.tokenizer.save_pretrained(self.config.output_dir)
            
        except KeyboardInterrupt:
            checkpoint_dir = f"{self.config.output_dir}/checkpoint"
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.trainer.save_model(checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")
            raise

    def run(self):
        """Полный цикл обучения."""
        self.setup_model()
        self.setup_trainer()
        self.train()

if __name__ == "__main__":    
    trainer = DCCircuitRLTrainer(CONFIG)
    trainer.run()