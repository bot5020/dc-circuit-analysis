#!/bin/bash

echo "════════════════════════════════════════════════════════════"
echo "🚀 Запуск на 2 GPU (ПРОСТОЙ СПОСОБ - DataParallel)"
echo "════════════════════════════════════════════════════════════"

# Явно указываем обе GPU
export CUDA_VISIBLE_DEVICES=0,1

echo ""
echo "Проверка GPU..."
python3 check_gpu_kaggle.sh 2>/dev/null || nvidia-smi -L

echo ""
echo "Запуск обучения..."
echo ""

# ПРОСТОЙ ЗАПУСК - без torchrun!
python3 training/rl_trainer_2gpu_v2.py

echo ""
echo "════════════════════════════════════════════════════════════"
echo "✅ Готово!"
echo "════════════════════════════════════════════════════════════"
