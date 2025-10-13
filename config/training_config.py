"""Конфигурация для обучения модели."""

from dataclasses import dataclass
from typing import List


@dataclass
class TrainingConfig:
    """Конфигурация для GRPO обучения модели."""
    
    # Модель
    model_name: str = "unsloth/Qwen2.5-3B-Instruct" 
    output_dir: str = "./dc_circuit_model_rl"
    
    # Формат модели
    use_4bit: bool = False  # True для 4-bit, False для BF16
    dtype: str = "bfloat16"  # "bfloat16" или "float16"
    use_flash_attention: bool = True  # Flash Attention 2
    
    # LoRA параметры
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None
    
    # Обучение
    learning_rate: float = 2e-5
    max_steps: int = 200 
    save_steps: int = 50 
    batch_size: int = 16 
    gradient_accumulation_steps: int = 1  
    
    # Оптимизатор
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    weight_decay: float = 0.1
    warmup_ratio: float = 0.1
    max_grad_norm: float = 0.1
    
    # Генерация
    max_seq_length: int = 11000
    max_completion_length: int = 11000
    num_generations: int = 8
    temperature: float = 0.7
    top_p: float = 0.95
    repetition_penalty: float = 1.1
    do_sample: bool = False
    gpu_memory_utilization: float = 0.35
    
    # Dataset
    difficulties: List[int] = None
    samples_per_difficulty: int = 150

    def __post_init__(self):
        if self.difficulties is None:
            self.difficulties = [1, 2]  # 2 уровня: series, parallel
        
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
