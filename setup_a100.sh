#!/bin/bash
# Скрипт для установки зависимостей на A100

echo "🚀 Установка зависимостей для A100..."

# Обновляем pip
pip install --upgrade pip

# Устанавливаем PyTorch с CUDA поддержкой
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Устанавливаем Flash Attention
pip install flash-attn --no-build-isolation

# Устанавливаем Unsloth
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Устанавливаем TRL
pip install trl

# Устанавливаем другие зависимости
pip install -r requirements.txt

echo "✅ Установка завершена!"
echo "Теперь можно запускать обучение на A100 с Flash Attention поддержкой"
