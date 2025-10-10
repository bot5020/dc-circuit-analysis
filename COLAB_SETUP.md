# 🎓 Запуск в Google Colab

**Полная инструкция по запуску DC Circuit Analysis в Google Colab**

---

## 🚀 Быстрый старт (3 минуты)

### Шаг 1: Откройте новый Colab notebook

1. Перейдите на [colab.research.google.com](https://colab.research.google.com/)
2. **File** → **New notebook**
3. Убедитесь, что выбран **GPU runtime**:
   - **Runtime** → **Change runtime type** → **T4 GPU**

### Шаг 2: Скопируйте код в ячейку

```python
# Клонируем репозиторий
!git clone https://github.com/your-username/tbank2.git
%cd tbank2

# Устанавливаем зависимости (займёт ~2-3 минуты)
!pip install -q -r requirements.txt

print("✅ Установка завершена!")
```

**Запустите ячейку** (Shift + Enter)

### Шаг 3: Проверьте установку

```python
# Проверяем что всё работает
from dc_circuit.game import DCCircuitGame

game = DCCircuitGame()
data = game.generate(1, difficulty=1)

print(f"✅ Генерация работает!")
print(f"📊 Промпт: {len(data[0].question)} символов")
print(f"🎯 Ответ: {data[0].answer}")
```

**Если видите ✅ - всё готово к работе!**

---

## 📚 Примеры использования

### 1. Демонстрация системы

```python
# Запуск интерактивного меню
!python main.py
```

Или прямо в коде:

```python
from dc_circuit.game import DCCircuitGame

game = DCCircuitGame()

# Генерация задач разной сложности
for difficulty in [1, 3, 5]:
    data = game.generate(1, difficulty=difficulty)
    task = data[0]
    
    print(f"\n{'='*60}")
    print(f"Сложность {difficulty}")
    print(f"{'='*60}")
    print(f"Вопрос: {task.question[:100]}...")
    print(f"Ответ: {task.answer}")
    print(f"Метаданные: {task.metadata.get('circuit_type')}")
```

### 2. GRPO обучение

**Вариант A: Стандартная конфигурация**

```python
from training.rl_trainer import DCCircuitRLTrainer, TrainingConfig

# Создаём конфигурацию для Colab (T4 GPU)
config = TrainingConfig(
    model_name="Qwen/Qwen2.5-1.5B",      # Маленькая модель для T4
    output_dir="./trained_model",
    max_steps=100,                        # Быстрая демонстрация
    batch_size=2,                         # Для T4 GPU (16GB)
    learning_rate=1e-5,
    lora_r=32,                            # Меньше параметров для быстроты
    lora_alpha=32,
    difficulties=[1, 2, 3],               # Простые задачи
    samples_per_difficulty=50             # Маленький датасет для демо
)

# Запуск обучения
trainer = DCCircuitRLTrainer(config)
trainer.run()
```

**Вариант B: Через файл**

```python
# Откройте training/rl_trainer.py и измените TrainingConfig:
# 1. max_steps = 100 (вместо 500)
# 2. batch_size = 2 (вместо 4)
# 3. samples_per_difficulty = 50 (вместо 500)

# Затем запустите:
!python training/rl_trainer.py
```

**Ожидаемое время:** 10-15 минут на T4 GPU (100 шагов)

### 3. Оценка модели

```python
from training.evaluate import evaluate_model, generate_evaluation_report
from dc_circuit.game import DCCircuitGame

# Создаём тестовые данные
game = DCCircuitGame()
test_data = game.generate(num_of_questions=50, difficulty=3)

# Определяем функцию модели (заглушка)
def my_model(question):
    # Здесь должна быть ваша модель
    # Пример: return model.generate(question)
    return "<answer>1.0</answer>"  # Заглушка

# Оценка
metrics = evaluate_model(my_model, test_data, max_samples=50)

print(f"📊 Результаты:")
print(f"  Accuracy: {metrics['accuracy']:.2%}")
print(f"  Correct: {metrics['correct']}/{metrics['total']}")
```

### 4. Визуализация

```python
from training.evaluate import plot_model_comparison, generate_evaluation_report

# Пример результатов
baseline_results = {
    1: 0.2,   # Сложность 1: 20% точность
    2: 0.15,  # Сложность 2: 15% точность
    3: 0.1    # Сложность 3: 10% точность
}

trained_results = {
    1: 0.7,   # Сложность 1: 70% точность
    2: 0.6,   # Сложность 2: 60% точность
    3: 0.5    # Сложность 3: 50% точность
}

# Создаём график
plot_model_comparison(
    baseline_results, 
    trained_results,
    save_path="model_comparison.png"
)

# Показываем график в Colab
from IPython.display import Image, display
display(Image("model_comparison.png"))

# Полный отчёт с JSON
generate_evaluation_report(
    baseline_results,
    trained_results,
    baseline_model="Baseline",
    trained_model="Trained",
    save_dir="reports"
)
```

---

## 🔧 Оптимизация для Colab

### GPU Memory Management

```python
import torch

# Очистка памяти GPU
torch.cuda.empty_cache()

# Проверка доступной памяти
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### Для T4 GPU (16GB)

**Рекомендуемая конфигурация:**

```python
TrainingConfig(
    model_name="Qwen/Qwen2.5-1.5B",     # <= 1.5B параметров
    batch_size=2,                        # Маленький batch
    gradient_accumulation_steps=4,       # Накопление градиентов
    lora_r=32,                           # Меньше LoRA параметров
    max_steps=100,                       # Быстрая демонстрация
)
```

### Для A100 GPU (40GB)

**Полная конфигурация:**

```python
TrainingConfig(
    model_name="Qwen/Qwen2.5-3B",       # Можно больше
    batch_size=8,                        # Больший batch
    gradient_accumulation_steps=2,
    lora_r=64,                           # Больше параметров
    max_steps=500,                       # Полное обучение
    samples_per_difficulty=500           # Большой датасет
)
```

---

## 💾 Сохранение результатов

### Сохранение в Google Drive

```python
# Подключаем Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Копируем обученную модель
!cp -r ./trained_model /content/drive/MyDrive/dc_circuit_trained

# Копируем отчёты
!cp -r ./reports /content/drive/MyDrive/dc_circuit_reports
```

### Скачивание локально

```python
# Упаковываем в zip
!zip -r trained_model.zip ./trained_model

# Скачиваем
from google.colab import files
files.download('trained_model.zip')
```

---

## 📊 Мониторинг обучения

### Простой мониторинг

```python
# В процессе обучения вы увидите:
# 📊 Step 20 | Correct: 2.567 | Model: 2.571
# 📊 Step 40 | Correct: 3.142 | Model: 3.145
# ...
```

### Детальный мониторинг

```python
from tqdm.notebook import tqdm

# tqdm автоматически создаст прогресс-бар в Colab
# Используется автоматически в trainer
```

---

## ⚡ Ускорение

### 1. Использование меньшей модели

```python
# Вместо 3B используйте 1.5B
model_name = "Qwen/Qwen2.5-1.5B"  # Быстрее в 2 раза
```

### 2. Меньший датасет для демо

```python
TrainingConfig(
    difficulties=[1, 2],              # Только 2 уровня
    samples_per_difficulty=50,        # 50 вместо 500
    max_steps=50                      # 50 вместо 500
)
```

### 3. Меньше LoRA параметров

```python
TrainingConfig(
    lora_r=16,        # 16 вместо 64
    lora_alpha=16     # 16 вместо 64
)
```

**Время:** 5-7 минут вместо 15-20

---

## 🐛 Troubleshooting

### Проблема: Out of Memory

**Решение 1:** Уменьшить batch_size

```python
config.batch_size = 1  # Вместо 2 или 4
```

**Решение 2:** Увеличить gradient_accumulation_steps

```python
config.gradient_accumulation_steps = 8  # Вместо 2
```

**Решение 3:** Меньшая модель

```python
config.model_name = "Qwen/Qwen2.5-0.5B"  # Самая маленькая
```

### Проблема: Slow training

**Решение:** Используйте меньший датасет

```python
config.samples_per_difficulty = 20  # Минимум для демо
config.max_steps = 20
```

### Проблема: Module not found

**Решение:** Переустановите зависимости

```python
!pip install -q -r requirements.txt --force-reinstall
```

### Проблема: CUDA error

**Решение:** Очистите GPU memory

```python
import torch
torch.cuda.empty_cache()

# Перезапустите runtime
# Runtime → Restart runtime
```

---

## 📝 Checklist перед запуском

- [ ] Выбран GPU runtime (T4 или лучше)
- [ ] Склонирован репозиторий
- [ ] Установлены все зависимости
- [ ] Проверена работа генерации
- [ ] Настроена конфигурация для вашего GPU
- [ ] (Опционально) Подключен Google Drive для сохранения

---

## 🎯 Полный пример для Colab

**Скопируйте этот код в одну ячейку:**

```python
# ============================================================================
# ПОЛНЫЙ ПРИМЕР: DC CIRCUIT ANALYSIS В GOOGLE COLAB
# ============================================================================

print("🚀 Начинаем установку...")

# 1. Клонирование
!git clone https://github.com/your-username/tbank2.git
%cd tbank2

# 2. Установка зависимостей
!pip install -q -r requirements.txt

print("\n✅ Установка завершена!")

# 3. Проверка
print("\n🔍 Проверяем систему...")
from dc_circuit.game import DCCircuitGame

game = DCCircuitGame()
data = game.generate(1, difficulty=1)

print(f"✅ Генерация работает: {len(data[0].question)} символов промпта")

# 4. Демонстрация
print("\n🎲 Демонстрация генерации...")
for diff in [1, 3, 5]:
    data = game.generate(1, difficulty=diff)
    print(f"\nСложность {diff}:")
    print(f"  Вопрос: {data[0].question[:80]}...")
    print(f"  Ответ: {data[0].answer}")

# 5. Быстрое обучение (5 минут)
print("\n🎓 Запускаем быстрое обучение...")
from training.rl_trainer import DCCircuitRLTrainer, TrainingConfig

config = TrainingConfig(
    model_name="Qwen/Qwen2.5-1.5B",
    output_dir="./demo_model",
    max_steps=20,
    batch_size=2,
    lora_r=16,
    difficulties=[1, 2],
    samples_per_difficulty=20
)

trainer = DCCircuitRLTrainer(config)
trainer.run()

print("\n🎉 Всё готово! Модель обучена и сохранена в ./demo_model")
```

---

## 📞 Помощь

**Если что-то не работает:**

1. Проверьте, что выбран GPU runtime
2. Очистите GPU memory: `torch.cuda.empty_cache()`
3. Перезапустите runtime: **Runtime → Restart runtime**
4. Создайте issue: [github.com/your-username/tbank2/issues](https://github.com/your-username/tbank2/issues)

---

## 🌟 Полезные ссылки

- **Colab Tips:** [Официальная документация](https://colab.research.google.com/notebooks/welcome.ipynb)
- **GPU Info:** [Colab GPU FAQ](https://research.google.com/colaboratory/faq.html)
- **unsloth:** [Документация](https://github.com/unslothai/unsloth)
- **TRL:** [Документация GRPO](https://huggingface.co/docs/trl)

---

<div align="center">

**Happy Coding in Colab! 🎓⚡**

</div>
