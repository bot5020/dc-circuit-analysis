"""GRPO обучение для анализа электрических цепей.
"""

import os
import sys
import torch


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List

from torch.utils.data import Dataset
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig

from base.utils import get_system_prompt
from base.data import Data
from dc_circuit.game import DCCircuitGame
from dc_circuit.verifier import DCCircuitVerifier
from config import TrainingConfig, CircuitConfig, VerifierConfig 

# Глобальный конфиг
CONFIG = TrainingConfig()


class DCCircuitDataset(Dataset):
    """Dataset для GRPO обучения"""

    def __init__(self, config: TrainingConfig, circuit_config: CircuitConfig = None, verifier_config: VerifierConfig = None):
        circuit_config = circuit_config or CircuitConfig()
        verifier_config = verifier_config or VerifierConfig()
        self.game = DCCircuitGame(circuit_config, verifier_config)
        self.config = config
        self.data = None  # Кэшируем данные
        self._generate_data()

    def _generate_data(self) -> None:
        """Генерация данных через game.generate() (один раз при инициализации)"""
        if self.data is not None:
            return  # Уже сгенерировано
            
        all_data = []
        for difficulty in self.config.difficulties:
            data_list = self.game.generate(
                num_of_questions=self.config.samples_per_difficulty,
                difficulty=difficulty
            )
            
            all_data.extend([{
                "prompt": [
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": data.question}
                ],
                "question": data.question,
                "answer": f"{float(data.answer):.3f}",
                "difficulty": data.difficulty
            } for data in data_list])
        
        self.data = all_data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]


class DCCircuitRLTrainer:
    """Тренер для GRPO обучения на задачах анализа DC цепей"""

    def __init__(self, config: TrainingConfig = None, circuit_config: CircuitConfig = None, verifier_config: VerifierConfig = None):
        """Инициализирует тренер с конфигурацией"""
        self.config = config or CONFIG
        self.circuit_config = circuit_config or CircuitConfig()
        self.verifier_config = verifier_config or VerifierConfig()
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self._verifier = None
        self.dataset = None  # Сохраняем dataset для reward функции

    def setup_model(self):
        """Загружает модель с LoRA"""        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=self.config.use_4bit,  
            dtype=self.config.dtype,
            fast_inference=True,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            use_flash_attention=self.config.use_flash_attention,
            device_map="auto"  # Автоматическое размещение на GPU
        )
        
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


    
    def reward_function(self, prompts, completions, **kwargs) -> List[float]:
        """Reward на основе verifier.
        
        Используем промпты для поиска правильных ответов в dataset.
        """
        if self._verifier is None:
            self._verifier = DCCircuitVerifier(self.verifier_config)
        
        if self.dataset is None:
            raise ValueError("dataset not initialized")
        
        # Вычисляем rewards для каждого completion
        rewards = []
        for idx, completion in enumerate(completions):
            # Ищем соответствующий элемент в dataset по промпту
            prompt_text = prompts[idx] if isinstance(prompts[idx], str) else str(prompts[idx])
            
            # Ищем в dataset по вопросу из промпта
            correct_answer = None
            for data_item in self.dataset:
                if data_item["question"] in prompt_text:
                    correct_answer = data_item["answer"]
                    break
            
            if correct_answer is None:
                # Если не нашли, используем случайный reward
                reward = 0.0
            else:
                # Создаем минимальный Data объект для верификатора
                data = Data(question="", answer=correct_answer, difficulty=1, metadata={})
                accuracy_score = self._verifier.get_accuracy_score(data, completion)
                reward = accuracy_score * 2.0  # Масштабируем reward [0, 2]
            
            rewards.append(reward)

        return rewards

    def setup_trainer(self):
        """Настройка GRPO тренера."""
        train_dataset = DCCircuitDataset(self.config, self.circuit_config, self.verifier_config)
        self.dataset = train_dataset  # Сохраняем для reward функции
        
        training_args = GRPOConfig(
            use_vllm=True, 
            learning_rate=self.config.learning_rate,
            adam_beta1=self.config.adam_beta1,
            adam_beta2=self.config.adam_beta2,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            logging_steps=1,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_generations=self.config.num_generations,
            max_prompt_length=4096, 
            max_completion_length=self.config.max_completion_length,  
            max_steps=self.config.max_steps,
            save_steps=self.config.save_steps,
            max_grad_norm=self.config.max_grad_norm,
            report_to="none",
            output_dir=self.config.output_dir,
            temperature=self.config.temperature,
            repetition_penalty=self.config.repetition_penalty,
        )
        
        # Создание тренера GRPO
        self.model.train()
        self.trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=[self.reward_function],
            args=training_args,
            train_dataset=train_dataset,
        )
        

    def train(self):
        """Запуск обучения."""
        try:        
            self.model.train()
            self.trainer.train()
            
            # Сохранение модели GRPO
            self.trainer.save_model(self.config.output_dir)
            self.tokenizer.save_pretrained(self.config.output_dir)
            
        except KeyboardInterrupt:
            checkpoint_dir = f"{self.config.output_dir}/checkpoint"
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.trainer.save_model(checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)
        except Exception as e:
            raise

    def run(self):
        """Цикл обучения GRPO."""
        self.setup_model()
        print("✅ Модель загружена")

        self.setup_trainer()
        print("✅ Тренер настроен")

        self.train()
        print("✅ Обучение завершено")

if __name__ == "__main__":    
    trainer = DCCircuitRLTrainer(CONFIG)
    trainer.run()