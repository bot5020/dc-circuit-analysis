"""Конфигурация для обучения модели."""

from dataclasses import dataclass
from typing import List


@dataclass
class TrainingConfig:
    """Конфигурация для GRPO обучения модели."""
    
    # Модель
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    output_dir: str = "./dc_circuit_model_rl"
    
    # LoRA параметры
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None
    
    # Обучение
    learning_rate: float = 1e-5
    max_steps: int = 500
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    
    # Генерация
    max_seq_length: int = 4096
    max_completion_length: int = 8196
    num_generations: int = 3
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = False
    gpu_memory_utilization: float = 0.25
    
    # Dataset
    difficulties: List[int] = None
    samples_per_difficulty: int = 100

    def __post_init__(self):
        if self.difficulties is None:
            self.difficulties = [1, 2, 3]  # 3 уровня: series, parallel, mixed
        
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
