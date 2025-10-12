#!/usr/bin/env python3
"""
Простой скрипт для запуска обучения с DDP на 2 GPU
Использование: python training/run_2gpu.py
"""

import subprocess
import sys
import torch

def main():
    num_gpus = torch.cuda.device_count()
    
    print("=" * 80)
    print(f"🚀 Запуск обучения на {num_gpus} GPU")
    print("=" * 80)
    
    if num_gpus < 2:
        print("❌ Найдено меньше 2 GPU! Запускаем на 1 GPU...")
        cmd = [sys.executable, "training/rl_trainer.py"]
    else:
        print(f"✅ Запускаем DDP на {num_gpus} GPU...")
        # Используем torchrun для DDP
        cmd = [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            "--standalone",
            "training/rl_trainer.py"
        ]
    
    print(f"\n📝 Команда: {' '.join(cmd)}\n")
    
    try:
        subprocess.run(cmd, check=True)
        print("\n" + "=" * 80)
        print("✅ Обучение завершено успешно!")
        print("=" * 80)
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 80)
        print(f"❌ Ошибка при обучении: {e}")
        print("=" * 80)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("⚠️  Обучение прервано пользователем")
        print("=" * 80)
        sys.exit(1)

if __name__ == "__main__":
    main()
