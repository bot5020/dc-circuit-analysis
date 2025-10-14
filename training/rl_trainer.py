"""GRPO обучение для анализа электрических цепей.
"""

import os
import sys
import torch
import json
import datetime
import re


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
    
    # Константы
    REWARD_SCALE_FACTOR = 2.0
    FORMAT_BONUS = 0.2  # Бонус за правильный формат ответа
    STRICT_FORMAT_BONUS = 0.5  # Бонус за строгий формат <answer>X.XXX</answer>
    RANDOM_STATE = 3407

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
        
        # Создаем файл для логирования LLM
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.llm_log_file = f"llm_logs_{timestamp}.jsonl"
        self.log_entries = []

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
            # Правильный chat template для Qwen3
            self.tokenizer.chat_template = """{% for message in messages %}
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
{% endfor %}"""
        
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
            random_state=self.RANDOM_STATE,
        )
        
        self.model.train()
        self.model.print_trainable_parameters()

    def log_llm_interaction(self, prompt, completion, reward=None, metadata=None):
        """Логирует взаимодействие с LLM"""
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "prompt": prompt,
            "completion": completion,
            "reward": reward,
            "metadata": metadata or {}
        }
        self.log_entries.append(entry)
        
        # Записываем в файл
        with open(self.llm_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    def save_llm_logs(self):
        """Сохраняет все логи LLM"""
        with open(f"llm_logs_complete_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w', encoding='utf-8') as f:
            json.dump(self.log_entries, f, ensure_ascii=False, indent=2)
    
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
            
            # Извлекаем вопрос из промпта (последнее сообщение user)
            correct_answer = None
            question_from_prompt = None
            
            # Если промпт - это список сообщений
            if isinstance(prompts[idx], list):
                for msg in reversed(prompts[idx]):
                    if isinstance(msg, dict) and msg.get('role') == 'user':
                        question_from_prompt = msg.get('content', '')
                        break
            else:
                # Если промпт - строка, используем её целиком
                question_from_prompt = prompt_text
            
            # Ищем соответствующую задачу в датасете ПО СОДЕРЖИМОМУ
            if question_from_prompt:
                for data_item in self.dataset:
                    if data_item["question"] == question_from_prompt:
                        correct_answer = data_item["answer"]
                        print(f"✅ Найден ответ по точному совпадению вопроса: {correct_answer}")
                        break
                
                # Если не нашли точного совпадения, ищем частичное
                if correct_answer is None:
                    for data_item in self.dataset:
                        if data_item["question"] in question_from_prompt or question_from_prompt in data_item["question"]:
                            correct_answer = data_item["answer"]
                            print(f"✅ Найден ответ по частичному совпадению: {correct_answer}")
                            break
            
            accuracy_score = None
            reward = 0.0
            
            if correct_answer is None:
                print(f"❌ Не найден правильный ответ для индекса {idx}")
            else:
                data = Data(question="", answer=correct_answer, difficulty=1, metadata={})
                completion_str_for_verifier = str(completion) if not isinstance(completion, str) else completion
                accuracy_score = self._verifier.get_accuracy_score(data, completion_str_for_verifier)
                
                # Базовый reward за правильность
                base_reward = accuracy_score * self.REWARD_SCALE_FACTOR
                
                # Бонус за правильный формат
                format_bonus = 0.0
                strict_format_bonus = 0.0
                has_think_tag = "<think>" in completion_str_for_verifier
                has_answer_tag = "<answer>" in completion_str_for_verifier
                
                # Строгая проверка: внутри <answer> ровно число с 3 знаками после точки
                strict_format_ok = False
                try:
                    answer_tags = re.findall(r"<answer>([\s\S]*?)</answer>", completion_str_for_verifier, flags=re.IGNORECASE)
                    if answer_tags:
                        last_answer = answer_tags[-1].strip()
                        strict_format_ok = bool(re.fullmatch(r"[-+]?\d+\.\d{3}", last_answer))
                except Exception:
                    strict_format_ok = False
                
                if has_think_tag and has_answer_tag:
                    format_bonus = self.FORMAT_BONUS
                    print(f"🎯 Бонус за формат: +{format_bonus:.1f}")
                if strict_format_ok:
                    strict_format_bonus = self.STRICT_FORMAT_BONUS
                    print(f"🔒 Бонус за строгий формат: +{strict_format_bonus:.1f}")
                
                reward = base_reward + format_bonus + strict_format_bonus
                print(f"✅ Accuracy: {accuracy_score:.3f}, base_reward: {base_reward:.3f}, total_reward: {reward:.3f}")
            
            self.log_llm_interaction(
                prompt=prompt_text,
                completion=completion,
                reward=reward,
                metadata={
                    "correct_answer": correct_answer,
                    "accuracy_score": accuracy_score if correct_answer else None,
                    "base_reward": base_reward if correct_answer else 0.0,
                    "format_bonus": format_bonus,
                        "strict_format_bonus": strict_format_bonus,
                    "total_reward": reward,
                    "batch_idx": idx,
                    "completion_has_think": has_think_tag,
                    "completion_has_answer": has_answer_tag,
                        "has_correct_format": has_think_tag and has_answer_tag,
                        "has_strict_answer_format": strict_format_ok
                }
            )
            
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
            
            # Сохраняем логи LLM
            self.save_llm_logs()
            print(f"✅ Логи LLM сохранены в {self.llm_log_file}")
            
        except KeyboardInterrupt:
            checkpoint_dir = f"{self.config.output_dir}/checkpoint"
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.trainer.save_model(checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)
            
            # Сохраняем логи при прерывании
            self.save_llm_logs()
            print(f"✅ Логи LLM сохранены в {self.llm_log_file}")
        except Exception as e:
            # Сохраняем логи при ошибке
            self.save_llm_logs()
            print(f"✅ Логи LLM сохранены в {self.llm_log_file}")
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