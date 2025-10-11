#!/bin/bash

echo "════════════════════════════════════════════════════════════"
echo "🚀 Запуск обучения на 2x L4"
echo "════════════════════════════════════════════════════════════"

# Проверка наличия 2 GPU
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "Найдено GPU: $GPU_COUNT"

if [ "$GPU_COUNT" -lt 2 ]; then
    echo "❌ Нужно минимум 2 GPU!"
    exit 1
fi

echo "✅ Запуск на 2 GPU..."
echo ""

# Запуск distributed training
torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    training/rl_trainer_2gpu.py

echo ""
echo "════════════════════════════════════════════════════════════"
echo "✅ Готово!"
echo "════════════════════════════════════════════════════════════"
