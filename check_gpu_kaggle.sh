#!/bin/bash

echo "════════════════════════════════════════════════════════════"
echo "🔍 ПРОВЕРКА GPU НА KAGGLE"
echo "════════════════════════════════════════════════════════════"

echo ""
echo "1️⃣ nvidia-smi:"
nvidia-smi -L

echo ""
echo "2️⃣ Количество GPU:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

echo ""
echo "3️⃣ CUDA_VISIBLE_DEVICES:"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-не установлено}"

echo ""
echo "4️⃣ PyTorch видит:"
python3 << 'PYEOF'
import torch
print(f"Количество GPU: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
PYEOF

echo ""
echo "════════════════════════════════════════════════════════════"
