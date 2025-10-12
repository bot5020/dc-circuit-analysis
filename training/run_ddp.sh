#!/bin/bash
# Скрипт для запуска обучения с DDP на 2x T4 GPU

echo "════════════════════════════════════════════════════════════════"
echo "🚀 Запуск обучения с DDP на 2 GPU"
echo "════════════════════════════════════════════════════════════════"

# Проверка количества GPU
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Обнаружено GPU: $NUM_GPUS"

if [ "$NUM_GPUS" -lt 2 ]; then
    echo "❌ Найдено меньше 2 GPU! Запускаем на 1 GPU..."
    python training/rl_trainer.py
else
    echo "✅ Запускаем DDP на $NUM_GPUS GPU..."
    
    # Вариант 1: torchrun (рекомендуется)
    if command -v torchrun &> /dev/null; then
        echo "Используем torchrun..."
        torchrun --nproc_per_node=$NUM_GPUS training/rl_trainer.py
    # Вариант 2: accelerate
    elif command -v accelerate &> /dev/null; then
        echo "Используем accelerate..."
        accelerate launch --num_processes=$NUM_GPUS training/rl_trainer.py
    # Вариант 3: python -m torch.distributed.launch
    else
        echo "Используем torch.distributed.launch..."
        python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS training/rl_trainer.py
    fi
fi

echo "════════════════════════════════════════════════════════════════"
echo "✅ Обучение завершено!"
echo "════════════════════════════════════════════════════════════════"
