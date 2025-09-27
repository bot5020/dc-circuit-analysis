# 🤖 DC Circuit Analysis Environment

**Полная система анализа электрических цепей с reinforcement learning и ИИ**

## 🎯 Что это?

Система для:
- ✅ **Генерации задач** анализа электрических цепей
- ✅ **RL обучения моделей** с GRPO алгоритмом
- ✅ **Оценки качества** ответов моделей
- ✅ **Верификации решений** с градиентной системой наград

## 🏗️ Архитектура

```
/Users/stepprog/tbank2/
├── base/              # Базовые классы системы
│   ├── data.py       # Классы данных
│   ├── game.py       # Игровая логика
│   └── verifier.py   # Верификация ответов
├── dc_circuit/        # Ядро анализа цепей
│   ├── game.py       # Генерация и решение задач
│   ├── generator.py  # Генератор цепей
│   ├── prompt.py     # Создание промптов
│   ├── solver.py     # Решатель цепей
│   └── verifier.py   # Верификатор решений
├── training/          # Система обучения
│   ├── rl_trainer.py    # GRPO обучение (основной)
│   ├── evaluate.py      # Оценка моделей
│   ├── datasets.py      # Создание датасетов
│   ├── utils.py         # Утилиты
│   └── README_RL.md     # Документация RL
├── config/            # Конфигурация
├── main.py            # Главный интерфейс
├── requirements.txt   # Зависимости
└── README.md          # Эта документация
```

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
# Базовые зависимости
pip install -r requirements.txt

# RL зависимости (для обучения)
pip install unsloth trl peft bitsandbytes accelerate
```

### 2. Запуск интерфейса

```bash
python main.py
```

### 3. Доступные опции

1. **Демонстрация генерации** - посмотреть примеры задач
2. **Демонстрация верификации** - проверить качество ответов
3. **Простая демонстрация** - базовые примеры
4. **Тестирование модели** - CPU-based тестирование
5. **Создание датасета** - генерация обучающих данных
6. **Валидация системы** - проверка корректности
7. **RL обучение** - GRPO обучение модели
8. **Оценка моделей** - сравнение baseline и обученной модели

## 🎨 Основные возможности

### **Генерация задач:**
- Серии, параллельные и смешанные цепи
- Разные уровни сложности (1-10)
- Физически корректные цепи
- Разнообразные типы вопросов

### **RL обучение (GRPO):**
- **unsloth** для ускоренной загрузки
- **5 reward functions** для качества
- **LoRA адаптеры** для эффективного fine-tuning
- **vLLM ускорение** для inference
- **Градиентная система** наград (0.0-4.5 балла)

### **Оценка моделей:**
- Сравнение baseline vs обученная модель
- Градиентная система оценки
- Визуализация результатов
- Сохранение отчетов

## 🎯 Примеры использования

### **Генерация задачи:**
```python
from dc_circuit.game import DCCircuitGame
game = DCCircuitGame()
data = game.generate(difficulty=5)
print(data.question)  # "Найдите ток через R1..."
print(data.answer)    # "0.123"
```

### **RL обучение:**
```python
from training.rl_trainer import DCCircuitRLTrainer, TrainingConfig

config = TrainingConfig(max_steps=500, lora_r=64)
trainer = DCCircuitRLTrainer(config)
trainer.train()  # Обучает модель с GRPO
```

### **Оценка:**
```python
from training.evaluate import generate_full_report

baseline, trained = generate_full_report()
print(f"Улучшение: {trained['avg'] - baseline['avg']:.2f}")
```

## 🔧 Технические детали

### **Формат данных:**
```text
Промпт: "...Question: Найдите ток... Answer: <gold>0.123</gold>"
Ответ модели: "<reasoning>Рассуждаем...</reasoning><answer>0.123</answer>"
```

### **Reward functions:**
- **XML Count** - правильное количество тегов
- **Format** - правильная структура
- **Numeric** - численный ответ
- **Correctness** - точность значения

### **GPU требования:**
- **8GB+ VRAM** для 3B модели
- **unsloth** для оптимизации памяти
- **Flash Attention 2** для ускорения

## 📊 Результаты

**Система обеспечивает:**
- 🎯 **Градиентное обучение** вместо бинарного
- 🧠 **Понимание физики** цепей
- ⚡ **Быструю генерацию** задач
- 📈 **Качественную оценку** моделей

## 🌐 GitHub Репозиторий

**Полный исходный код доступен на GitHub:**
```bash
git clone https://github.com/your-username/dc-circuit-analysis.git
cd dc-circuit-analysis
```

### **Структура репозитория:**
```
/dc-circuit-analysis/
├── base/              # Базовые классы
├── dc_circuit/        # Ядро анализа цепей
├── training/          # Система обучения
├── config/            # Конфигурация
├── main.py            # Главный интерфейс
├── requirements.txt   # Зависимости
└── README.md          # Документация
```

### **Для 2 T4 GPU:**
```bash
# Запуск с 2 GPU
python3 training/rl_trainer.py \
  --model unsloth/Qwen2.5-3B-Instruct \
  --num-gpus 2 \
  --gpu-memory-utilization 0.8 \
  --batch-size 1 \
  --max-steps 500
```

---

*Создано для демонстрации современных подходов к обучению моделей анализа электрических цепей* 🚀⚡🔬
