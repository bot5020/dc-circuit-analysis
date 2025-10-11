#!/usr/bin/env python3
"""Проверка доступных GPU"""

import torch

print("="*80)
print("🔍 ПРОВЕРКА GPU")
print("="*80)

# Количество GPU
num_gpus = torch.cuda.device_count()
print(f"\nКоличество GPU: {num_gpus}")

if num_gpus == 0:
    print("❌ GPU не найдено!")
elif num_gpus == 1:
    print("⚠️  Найдена ОДНА GPU - нельзя использовать DDP для 2 GPU")
    print("   Используйте обычный rl_trainer.py")
else:
    print(f"✅ Найдено {num_gpus} GPU - можно использовать DDP")

# Информация о каждой GPU
print("\n" + "-"*80)
for i in range(num_gpus):
    props = torch.cuda.get_device_properties(i)
    print(f"\nGPU {i}:")
    print(f"  Название: {props.name}")
    print(f"  Память: {props.total_memory / 1024**3:.1f} GB")
    print(f"  Compute: {props.major}.{props.minor}")

# CUDA_VISIBLE_DEVICES
import os
cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'не задано')
print(f"\nCUDA_VISIBLE_DEVICES: {cuda_visible}")

print("\n" + "="*80)
if num_gpus == 1:
    print("💡 РЕКОМЕНДАЦИЯ: Используйте training/rl_trainer.py (без DDP)")
    print("="*80)
elif num_gpus >= 2:
    print("💡 РЕКОМЕНДАЦИЯ: Можно использовать training/rl_trainer_2gpu.py")
    print("="*80)
else:
    print("❌ GPU не найдено")
    print("="*80)
