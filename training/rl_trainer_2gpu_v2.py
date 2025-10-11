"""GRPO обучение для 2x GPU - УПРОЩЕННАЯ ВЕРСИЯ БЕЗ DDP"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from typing import List

import torch
from torch.utils.data import Dataset
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig

from base.utils import extract_answer
from base.data import Data
from dc_circuit.game import DCCircuitGame
from dc_circuit.verifier import DCCircuitVerifier


print("="*80)
print("🔍 ПРОВЕРКА GPU")
print("="*80)
print(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_properties(i).name}")
print("="*80)


# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

@dataclass
class TrainingConfig:
    """Настройки для 2 GPU через DataParallel (ПРОЩЕ ЧЕМ DDP)"""
    
    # Модель
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    output_dir: str = "./dc_circuit_model_rl_2gpu_v2"
    max_seq_length: int = 2048
    
    # LoRA
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.0  # Ставим 0 для скорости
    
    # Обучение (увеличено для 2 GPU)
    learning_rate: float = 1e-5
    max_steps: int = 500
    batch_size: int = 8  # На каждой GPU
    gradient_accumulation_steps: int = 2
    num_generations: int = 2
    save_steps: int = 50
    
    # Dataset
    difficulties: List[int] = None
    samples_per_difficulty: int = 100
    
    def __post_init__(self):
        if self.difficulties is None:
            self.difficulties = [1, 2, 3, 4, 5]


CONFIG = TrainingConfig()


# ============================================================================
# DATASET (БЕЗ ИЗМЕНЕНИЙ)
# ============================================================================

class DCCircuitDataset(Dataset):
    def __init__(self, config: TrainingConfig):
        self.game = DCCircuitGame()
        self.config = config
        self._data_cache = None

    def _generate_data(self) -> List[dict]:
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
                        {"role": "user", "content": f"{data.question}\n<gold>{float(data.answer):.3f}</gold>"}
                    ],
                    "question": data.question,
                    "answer": f"{float(data.answer):.3f}",
                    "difficulty": data.difficulty
                } for data in data_list])
            
            self._data_cache = all_data
            print(f"📊 Всего: {len(all_data)} элементов")
        
        return self._data_cache

    def __len__(self) -> int:
        return len(self._generate_data())

    def __getitem__(self, idx: int) -> dict:
        return self._generate_data()[idx]


# ============================================================================
# TRAINER
# ============================================================================

class DCCircuitRLTrainer:
    def __init__(self, config: TrainingConfig = None):
        self.config = config or CONFIG
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self._verifier = None
        
        print(f"\n🎮 GRPO Trainer для 2x GPU (DataParallel)")
        print(f"Batch size: {self.config.batch_size} per GPU")
        print(f"GPU count: {torch.cuda.device_count()}")

    def setup_model(self):
        """Загружает модель БЕЗ DDP - используем DataParallel"""
        print(f"📦 Загрузка {self.config.model_name}...")
        
        # Загружаем модель на GPU 0
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=True,
            fast_inference=False,
            device_map="cuda:0"  # Явно на GPU 0
        )
        
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ message['content'] }}{% endif %}{% endfor %}"
        
        # LoRA
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
        
        # ВАРИАНТ 1: Если есть 2+ GPU - используем DataParallel
        if torch.cuda.device_count() > 1:
            print(f"🔗 Используем {torch.cuda.device_count()} GPU через DataParallel")
            self.model = torch.nn.DataParallel(self.model)
        
        self.model.train()
        print("✅ Модель загружена")
        
        # Print trainable parameters
        if hasattr(self.model, 'module'):
            self.model.module.print_trainable_parameters()
        else:
            self.model.print_trainable_parameters()

    def _extract_prompt_content(self, prompts) -> str:
        if not prompts:
            return ""
        if isinstance(prompts[0], str):
            return prompts[0]
        return prompts[0][-1]['content'] if prompts[0] else ""
    
    def _normalize_completions(self, completions) -> List[str]:
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
        return (step in [1, 2, 3, 5, 10]) or (step > 10 and step % 20 == 0)
    
    def _extract_gold_answer(self, prompt_content: str) -> str:
        gold_start = prompt_content.find("<gold>")
        gold_end = prompt_content.find("</gold>")
        
        if gold_start != -1 and gold_end != -1:
            return prompt_content[gold_start + 6:gold_end].strip()
        return ""
    
    def _log_detailed_metrics(self, step: int, correct_answer: str, 
                             raw_response: str, extracted: str):
        print("\n" + "="*80)
        print(f"📊 Step {step}")
        print("="*80)
        print(f"\n✅ Правильный: {correct_answer}")
        print(f"\n🤖 Модель: {raw_response[:150]}...")
        print(f"\n🔍 Извлечён: '{extracted}'")
        
        if extracted:
            data = Data(question="", answer=correct_answer, difficulty=1, metadata={})
            try:
                score = self._verifier.get_accuracy_score(data, raw_response)
                reward = score * 2.0
                
                # Детальный расчёт
                try:
                    model_val = float(extracted)
                    correct_val = float(correct_answer)
                    if abs(correct_val) > 1e-12:
                        rel_error = abs(model_val - correct_val) / abs(correct_val)
                        rel_error_percent = rel_error * 100
                        print(f"\n🔬 Погрешность: {rel_error_percent:.2f}%")
                except:
                    pass
                
                print(f"\n💰 Score: {score:.2f} → Reward: {reward:.2f}")
            except Exception as e:
                print(f"\n💰 Reward: 0.0 ({e})")
        
        print("="*80 + "\n")
    
    def _calculate_rewards(self, correct_answer: str, responses: List[str]) -> List[float]:
        data = Data(question="", answer=correct_answer, difficulty=1, metadata={})
        rewards = []
        
        for response in responses:
            try:
                accuracy_score = self._verifier.get_accuracy_score(data, response)
                reward = accuracy_score * 2.0
            except Exception:
                reward = 0.0
            rewards.append(reward)
        
        return rewards
    
    def reward_function(self, prompts, completions, **kwargs) -> List[float]:
        if self._verifier is None:
            self._verifier = DCCircuitVerifier()
        
        responses = self._normalize_completions(completions)
        
        trainer_state = kwargs.get('trainer_state')
        step = getattr(trainer_state, 'global_step', 0) if trainer_state else 0
        
        if self._should_log_step(step) and prompts and responses:
            prompt_content = self._extract_prompt_content(prompts)
            correct_answer = self._extract_gold_answer(prompt_content)
            
            if correct_answer:
                raw_response = responses[0] if responses else ""
                extracted = extract_answer(raw_response) if responses else ""
                self._log_detailed_metrics(step, correct_answer, raw_response, extracted)
        
        prompt_content = self._extract_prompt_content(prompts)
        correct_answer = self._extract_gold_answer(prompt_content)
        
        if not correct_answer:
            return [0.0] * len(responses)
        
        return self._calculate_rewards(correct_answer, responses)

    def setup_trainer(self):
        train_dataset = DCCircuitDataset(self.config)
    
        training_args = GRPOConfig(
            use_vllm=True,
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
            max_completion_length=2048,
            max_steps=self.config.max_steps,
            save_steps=self.config.save_steps,
            max_grad_norm=0.1,
            report_to="none",
            output_dir=self.config.output_dir,
            temperature=0.7,
            repetition_penalty=1.1,
        )
        
        # ВАЖНО: Передаем модель С DataParallel оберткой!
        # GRPOTrainer должен видеть DataParallel чтобы использовать обе GPU
        self.trainer = GRPOTrainer(
            model=self.model,  # ← Передаём как есть, с DataParallel!
            processing_class=self.tokenizer,
            reward_funcs=[self.reward_function],
            args=training_args,
            train_dataset=train_dataset,
        )
        
        print("✅ GRPO тренер настроен")

    def train(self):
        try:
            self.model.train()
            self.trainer.train()
            
            self.trainer.save_model(self.config.output_dir)
            self.tokenizer.save_pretrained(self.config.output_dir)
            print(f"\n✅ Модель сохранена в {self.config.output_dir}")
            
        except KeyboardInterrupt:
            print("\n⚠️  Прервано")
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()
            raise

    def run(self):
        self.setup_model()
        self.setup_trainer()
        self.train()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 ЗАПУСК ОБУЧЕНИЯ (DataParallel)")
    print("="*80)
    print(f"GPU доступно: {torch.cuda.device_count()}")
    print(f"Effective batch: {CONFIG.batch_size * CONFIG.gradient_accumulation_steps * torch.cuda.device_count()}")
    print("="*80 + "\n")
    
    trainer = DCCircuitRLTrainer(CONFIG)
    trainer.run()
    
    print("\n✅ Обучение завершено!")
