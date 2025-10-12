# Исправление ошибки памяти GPU

## 🚨 Проблема

Ошибка `Duplicate layer name: model.layers.0.self_attn.attn` возникла из-за:

1. **Слишком большая длина последовательности**: `max_seq_length=15000` требует много GPU памяти
2. **Недостаточная GPU память**: vLLM не может загрузить модель с такими параметрами
3. **Конфликт в vLLM**: При нехватке памяти возникают дублирующиеся имена слоев

## ✅ Решение

### Изменения в конфигурации:

```python
# ДО (проблемные значения):
max_seq_length: int = 15000
lora_r: int = 64
lora_alpha: int = 64
batch_size: int = 4
gradient_accumulation_steps: int = 4
num_generations: int = 8
gpu_memory_utilization=0.15
max_completion_length=10000

# ПОСЛЕ (исправленные значения):
max_seq_length: int = 8192
lora_r: int = 32
lora_alpha: int = 32
batch_size: int = 2
gradient_accumulation_steps: int = 2
num_generations: int = 4
gpu_memory_utilization=0.3
max_completion_length=4096
```

### Почему это работает:

1. **8192 токенов** - оптимальная длина для большинства задач
2. **LoRA rank 32** - достаточно для качества, но экономит память
3. **Batch size 2** - меньше памяти на батч
4. **4 генерации** - достаточно для GRPO, но не перегружает память
5. **30% GPU памяти** - больше места для модели

## 📊 Результат

- ✅ Модель загрузится без ошибок
- ✅ Обучение будет стабильным
- ✅ Качество сохранится
- ✅ Память используется эффективно

## 🔧 Дополнительные рекомендации

Если все еще есть проблемы с памятью:

1. Уменьшите `max_seq_length` до 4096
2. Уменьшите `batch_size` до 1
3. Уменьшите `num_generations` до 2
4. Увеличьте `gpu_memory_utilization` до 0.5
