"""GRPO обучение для анализа электрических цепей - 2 GPU VERSION"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from typing import List

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig

from base.utils import extract_answer
from base.data import Data
from dc_circuit.game import DCCircuitGame
from dc_circuit.verifier import DCCircuitVerifier


# ============================================================================
# DISTRIBUTED SETUP
# ============================================================================

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", -1))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
IS_DISTRIBUTED = LOCAL_RANK != -1

if IS_DISTRIBUTED:
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(LOCAL_RANK)
    print(f"🚀 Rank {LOCAL_RANK}/{WORLD_SIZE} initialized")


# ============================================================================
# КОНФИГУРАЦИЯ ОБУЧЕНИЯ
# ============================================================================

@dataclass
class TrainingConfig:
    """Настройки обучения для 2x L4"""
    
    # Модель
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    output_dir: str = "./dc_circuit_model_rl_2gpu"
    max_seq_length: int = 2048
    
    # LoRA
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    
    # Обучение (УВЕЛИЧЕНО для 2 GPU)
    learning_rate: float = 1e-5
    max_steps: int = 500
    batch_size: int = 8  # Увеличено с 4 до 8!
    gradient_accumulation_steps: int = 2
    num_generations: int = 2
    save_steps: int = 50
    
    # Dataset
    difficulties: List[int] = None
    samples_per_difficulty: int = 100
    
    # Distributed
    distributed: bool = IS_DISTRIBUTED
    local_rank: int = LOCAL_RANK
    world_size: int = WORLD_SIZE
    
    def __post_init__(self):
        if self.difficulties is None:
            self.difficulties = [1, 2, 3, 4, 5]


CONFIG = TrainingConfig()


# ============================================================================
# DATASET (БЕЗ ИЗМЕНЕНИЙ)
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
            
            # Только главный процесс генерирует данные
            if not IS_DISTRIBUTED or LOCAL_RANK == 0:
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
                
                print(f"📊 Всего в датасете: {len(all_data)} элементов")
            
            # Синхронизация между процессами
            if IS_DISTRIBUTED:
                # Broadcast размера данных
                if LOCAL_RANK == 0:
                    data_size = torch.tensor([len(all_data)], device='cuda')
                else:
                    data_size = torch.tensor([0], device='cuda')
                dist.broadcast(data_size, src=0)
                
                # Broadcast сами данные
                if LOCAL_RANK != 0:
                    all_data = [None] * data_size.item()
                
                import pickle
                if LOCAL_RANK == 0:
                    data_bytes = pickle.dumps(all_data)
                    data_tensor = torch.ByteTensor(list(data_bytes)).cuda()
                    size_tensor = torch.tensor([len(data_bytes)], device='cuda')
                else:
                    size_tensor = torch.tensor([0], device='cuda')
                
                dist.broadcast(size_tensor, src=0)
                
                if LOCAL_RANK != 0:
                    data_tensor = torch.ByteTensor(size_tensor.item()).cuda()
                
                dist.broadcast(data_tensor, src=0)
                
                if LOCAL_RANK != 0:
                    all_data = pickle.loads(bytes(data_tensor.cpu().numpy()))
            
            self._data_cache = all_data
        
        return self._data_cache

    def __len__(self) -> int:
        return len(self._generate_data())

    def __getitem__(self, idx: int) -> dict:
        data = self._generate_data()
        return data[idx]


# ============================================================================
# GRPO ТРЕНЕР (С DDP)
# ============================================================================

class DCCircuitRLTrainer:
    """Тренер для GRPO обучения на 2x L4"""

    def __init__(self, config: TrainingConfig = None):
        self.config = config or CONFIG
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self._verifier = None
        
        if LOCAL_RANK in [-1, 0]:
            print(f"\n🎮 GRPO Trainer для 2x L4")
            print(f"Distributed: {self.config.distributed}")
            print(f"World size: {self.config.world_size}")
            print(f"Batch size: {self.config.batch_size} (per GPU)")
            print(f"Effective batch: {self.config.batch_size * self.config.gradient_accumulation_steps * self.config.world_size}")

    def setup_model(self):
        """Загружает модель с DDP поддержкой"""
        if LOCAL_RANK in [-1, 0]:
            print(f"📦 Загрузка модели {self.config.model_name}...")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=True,
            fast_inference=False
        )
        
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ message['content'] }}{% endif %}{% endfor %}"
        
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
        
        # ============== ВАЖНО: DDP WRAPPER ==============
        if IS_DISTRIBUTED:
            if LOCAL_RANK == 0:
                print("🔗 Wrapping model with DDP...")
            
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[LOCAL_RANK],
                output_device=LOCAL_RANK,
                find_unused_parameters=False
            )
        # ================================================
        
        self.model.train()
        if LOCAL_RANK in [-1, 0]:
            print("✅ Модель загружена")
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
        """Логирование только на главном процессе"""
        if LOCAL_RANK not in [-1, 0]:
            return
            
        print("\n" + "="*80)
        print(f"📊 GPU {LOCAL_RANK} | Step {step}")
        print("="*80)
        print(f"\n✅ Правильный: {correct_answer}")
        print(f"\n🤖 Модель: {raw_response[:150]}...")
        print(f"\n🔍 Извлечён: '{extracted}'")
        
        if extracted:
            data = Data(question="", answer=correct_answer, difficulty=1, metadata={})
            try:
                score = self._verifier.get_accuracy_score(data, raw_response)
                reward = score * 2.0
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
        """Reward function (БЕЗ ИЗМЕНЕНИЙ)"""
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
        """Настройка GRPO тренера с DDP"""
        train_dataset = DCCircuitDataset(self.config)
    
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
            max_completion_length=2048,
            max_steps=self.config.max_steps,
            save_steps=self.config.save_steps,
            max_grad_norm=0.1,
            report_to="none",
            output_dir=self.config.output_dir,
            temperature=0.7,
            repetition_penalty=1.1,
            
            # ============== ВАЖНО: DDP SETTINGS ==============
            local_rank=self.config.local_rank,
            ddp_find_unused_parameters=False,
            # =================================================
        )
        
        # Получаем базовую модель для trainer
        model_for_trainer = self.model.module if IS_DISTRIBUTED else self.model
        
        self.trainer = GRPOTrainer(
            model=model_for_trainer,
            processing_class=self.tokenizer,
            reward_funcs=[self.reward_function],
            args=training_args,
            train_dataset=train_dataset,
        )
        
        if LOCAL_RANK in [-1, 0]:
            print("✅ GRPO тренер настроен")

    def train(self):
        """Запуск обучения"""
        try:
            self.model.train()
            self.trainer.train()
            
            # Сохранение только на главном процессе
            if LOCAL_RANK in [-1, 0]:
                self.trainer.save_model(self.config.output_dir)
                self.tokenizer.save_pretrained(self.config.output_dir)
                print(f"\n✅ Модель сохранена в {self.config.output_dir}")
            
        except KeyboardInterrupt:
            if LOCAL_RANK in [-1, 0]:
                print("\n⚠️  Прервано пользователем")
        except Exception as e:
            if LOCAL_RANK in [-1, 0]:
                print(f"\n❌ Ошибка: {e}")
            raise

    def run(self):
        """Полный цикл обучения"""
        self.setup_model()
        self.setup_trainer()
        self.train()


if __name__ == "__main__":
    if LOCAL_RANK in [-1, 0]:
        print("\n" + "="*80)
        print("🚀 ЗАПУСК ОБУЧЕНИЯ НА 2x L4")
        print("="*80)
        print(f"Процессов: {WORLD_SIZE}")
        print(f"Эффективный batch: {CONFIG.batch_size * CONFIG.gradient_accumulation_steps * WORLD_SIZE}")
        print("="*80 + "\n")
    
    trainer = DCCircuitRLTrainer(CONFIG)
    trainer.run()
    
    if IS_DISTRIBUTED:
        dist.destroy_process_group()
    
    if LOCAL_RANK in [-1, 0]:
        print("\n✅ Обучение завершено!")
