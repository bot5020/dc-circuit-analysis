# 🚀 DC Circuit Analysis - Упрощенная система обучения LLM

## 📋 Описание

Система для обучения языковых моделей решению задач анализа электрических цепей постоянного тока (DC). **Упрощенная версия** фокусируется на базовых топологиях: последовательных и параллельных цепях.

## 🎯 Ключевые особенности

### ✅ **Упрощенная архитектура:**
- **Только 2 уровня сложности:** Series (1) и Parallel (2)
- **Базовые физические законы:** Закон Ома + простые правила
- **Оптимизированные параметры обучения** для быстрого сходимости
- **Бонусная система reward** за правильный формат ответов

### 🔧 **Основные компоненты:**
- **Генератор цепей** - создание простых series/parallel цепей
- **Решатель** - эталонные решения через базовые формулы
- **Верификатор** - проверка правильности с градиентной оценкой
- **GRPO тренер** - обучение с подкреплением + LoRA
- **Система оценки** - сравнение методов с красивыми диаграммами

## 🏗️ Архитектура системы

```
┌─────────────────────────────────────────────────────────┐
│                DC Circuit Analysis System              │
│                   (Simplified Version)                 │
└─────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Generator   │   │    Solver    │   │  Verifier    │
│              │   │              │   │              │
│ Series/      │──▶│ Basic Ohm's  │──▶ Gradient    │
│ Parallel     │   │ Law + Rules   │   │ Evaluation   │
│ Circuits     │   │              │   │              │
└──────────────┘   └──────────────┘   └──────────────┘
        │                   │                   │
        └───────────┬───────┴───────┬───────────┘
                            │               │
                            ▼               ▼
                    ┌──────────────┐ ┌──────────────┐
                    │    Prompt    │ │  Calculator  │
                    │              │ │              │
            │ Structured   │ │ Current/     │
            │ Format       │ │ Voltage      │
                    └──────────────┘ └──────────────┘
                            │               │
                            └───────┬───────┘
                                    │
                                    ▼
                            ┌──────────────┐
                            │  LLM Agent   │
                            │              │
                            │ Qwen3-4B +   │
                    │ LoRA + GRPO  │
                            └──────────────┘
```

## 📊 Уровни сложности

### 🔗 **Сложность 1: Series Circuits (Последовательные)**
- **2-3 резистора** в последовательном соединении
- **Формула:** `R_total = R₁ + R₂ + R₃`
- **Ток:** одинаковый через все резисторы
- **Напряжения:** суммируются

### ⚡ **Сложность 2: Parallel Circuits (Параллельные)**
- **2-3 резистора** в параллельном соединении  
- **Формула:** `1/R_total = 1/R₁ + 1/R₂ + 1/R₃`
- **Напряжение:** одинаковое на всех резисторах
- **Токи:** суммируются в узлах

## 🧮 Физические законы

### ✅ **Используемые законы:**
- **Закон Ома:** `V = I × R`, `I = V/R`, `R = V/I`
- **Series правила:** `R_total = R₁ + R₂ + ...`, `I_total = I₁ = I₂`
- **Parallel правила:** `1/R_total = 1/R₁ + 1/R₂ + ...`, `V_total = V₁ = V₂`

### ❌ **Убрано (не нужно для простых цепей):**
- Законы Кирхгофа (KCL, KVL)
- Смешанные цепи
- Эквивалентное сопротивление

## 🎯 Система Reward

### **Базовый reward (за правильность):**
```python
base_reward = accuracy_score * 2.0  # [0, 2]
```

### **Бонус за формат (новое!):**
```python
format_bonus = 0.5  # Если есть <think>...</think><answer>X.XXX</answer>
total_reward = base_reward + format_bonus  # [0, 2.5]
```

### **Градиентная оценка точности:**
- **≤1%** → 1.0 (perfect)
- **≤5%** → 0.75 (good)  
- **≤10%** → 0.5 (ok)
- **≤20%** → 0.25 (fair)
- **>20%** → 0.0 (wrong)

## ⚙️ Параметры обучения

### **Оптимизированная конфигурация:**
```python
TrainingConfig(
    # Модель
    model_name="unsloth/qwen3-4b-instruct-2507-unsloth-bnb-4bit",
    max_seq_length=11000,
    gpu_memory_utilization=0.35,
    
    # LoRA (оптимизировано)
    lora_r=64,
    lora_alpha=64,
    lora_dropout=0.1,
    
    # Обучение (быстрое)
    learning_rate=2e-5,
    max_steps=200,  # Было 500
    batch_size=8,   # Было 4
    save_steps=50,
    
    # Генерация
    num_generations=4,
    max_completion_length=11000,
    temperature=0.7,
    
    # Dataset (упрощенный)
    difficulties=[1, 2],  # Только series и parallel
    samples_per_difficulty=100
)
```

## 🚀 Быстрый старт

### **1. Установка:**
```bash
git clone <repository>
cd tbank2
pip install -r requirements.txt
```

### **2. Демонстрация:**
```bash
python main.py
```

### **3. Запуск тестов:**
```bash
pytest tests/ -v
```

### **4. Обучение модели:**
```bash
python training/rl_trainer.py
```

### **5. Оценка результатов:**
```bash
python training/evaluate.py
```

## 📈 Ожидаемые результаты

### **Сравнение методов:**

| Метод | Series (1) | Parallel (2) | Среднее |
|-------|-------------|---------------|---------|
| **Zero-shot** | ~10-15% | ~5-10% | ~7-12% |
| **Prompt Engineering** | ~20-25% | ~15-20% | ~17-22% |
| **GRPO Trained** | **~85-95%** | **~75-85%** | **~80-90%** |

### **Улучшения:**
- **Точность:** 10-15x улучшение
- **Формат ответов:** 70-80% правильный формат
- **Время обучения:** ~1-2 часа (vs 12+ часов в полной версии)

## 🎨 Красивые диаграммы результатов

Система автоматически генерирует ASCII диаграммы:

```
📈 ВИЗУАЛЬНАЯ ДИАГРАММА РЕЗУЛЬТАТОВ:
============================================================

🎯 ТОЧНОСТЬ ОТВЕТОВ:
  Zero-shot    │███████████░░░░░░░░░░░░░░░░░░░│ 30.0%
  Prompt Eng   │██████████████████████░░░░░░░░│ 60.0%
  GRPO Trained │██████████████████████████████│ 80.0%

📝 ПРАВИЛЬНЫЙ ФОРМАТ:
  Zero-shot    │████░░░░░░░░░░░░░░░░░░░░░░░░░░│ 10.0%
  Prompt Eng   │█████████████████░░░░░░░░░░░░░│ 40.0%
  GRPO Trained │██████████████████████████████│ 70.0%

📊 УЛУЧШЕНИЯ:
🚀 GRPO обучение улучшил точность на 50.0%
🚀 GRPO обучение улучшил формат на 60.0%
```

## 📁 Структура проекта

```
tbank2/
├── base/                    # Базовые классы
│   ├── data.py             # Data класс
│   ├── verifier.py         # Verifier интерфейс  
│   └── utils.py            # Утилиты + системный промпт
│
├── dc_circuit/             # Основные модули
│   ├── generator.py        # Генератор цепей (упрощенный)
│   ├── solver.py           # Решатель (базовые формулы)
│   ├── verifier.py         # Верификатор с градиентной оценкой
│   ├── prompt.py           # Генерация промптов
│   ├── game.py             # DCCircuitGame
│   └── calculators/        # Калькуляторы
│       ├── base.py         # Базовый класс
│       ├── current.py      # Ток через резистор
│       └── voltage.py      # Напряжение на резисторе
│
├── config/                 # Конфигурации
│   ├── circuit_config.py   # Параметры генерации
│   ├── verifier_config.py  # Параметры верификации
│   └── training_config.py  # Параметры обучения
│
├── training/               # Обучение и оценка
│   ├── rl_trainer.py       # GRPO тренер
│   └── evaluate.py         # Система оценки
│
├── tests/                  # Тесты (обновлены)
│   ├── test_generator.py   # Тесты генератора
│   ├── test_physics.py     # Физическая корректность
│   ├── test_calculators.py # Тесты калькуляторов
│   ├── test_integration.py # Интеграционные тесты
│   └── test_utils.py       # Тесты утилит
│
├── main.py                 # Демонстрация системы
├── requirements.txt        # Зависимости
└── README.md              # Документация
```

## 🔧 Конфигурация

### **CircuitConfig (упрощенный):**
```python
CircuitConfig(
    difficulties=[1, 2],           # Только series и parallel
    voltage_range=(5, 24),        # Простые напряжения
    resistance_range=(10, 100),    # Простые сопротивления
    topology_configs={
        "series": {
            "min_resistors": 2,
            "max_resistors": 3,
            "question_types": ["current", "voltage"]
        },
        "parallel": {
            "min_resistors": 2, 
            "max_resistors": 3,
            "question_types": ["current", "voltage"]
        }
    }
)
```

### **VerifierConfig:**
```python
VerifierConfig(
    relative_tolerance=1e-3,      # 0.1% относительная погрешность
    absolute_tolerance=1e-6,      # 1 микро-единица абсолютная
    answer_precision=3,           # 3 знака после запятой
    threshold_perfect=0.01,       # ≤1% → reward 1.0
    threshold_good=0.05,          # ≤5% → reward 0.75
    threshold_ok=0.10,            # ≤10% → reward 0.5
    threshold_fair=0.20           # ≤20% → reward 0.25
)
```

## 🧪 Тестирование

### **Запуск всех тестов:**
```bash
pytest tests/ -v
```

### **Конкретные тесты:**
```bash
pytest tests/test_generator.py -v    # Генератор цепей
pytest tests/test_physics.py -v      # Физическая корректность
pytest tests/test_calculators.py -v  # Калькуляторы
pytest tests/test_integration.py -v  # Интеграционные тесты
```

### **Покрытие кода:**
```bash
pytest tests/ --cov=dc_circuit --cov-report=html
```

## 📊 Примеры использования

### **Генерация задачи:**
```python
from dc_circuit.game import DCCircuitGame
from config import CircuitConfig, VerifierConfig

# Создание игры
config = CircuitConfig()
verifier_config = VerifierConfig()
game = DCCircuitGame(config, verifier_config)

# Генерация задачи
data_list = game.generate(num_of_questions=1, difficulty=1)
data = data_list[0]

print(f"Вопрос: {data.question}")
print(f"Правильный ответ: {data.answer}")
print(f"Тип цепи: {data.metadata['circuit_type']}")
```

### **Проверка ответа:**
```python
# Ответ агента
agent_response = "<think>Step 1: Series circuit\nStep 2: R_total = 4Ω + 6Ω = 10Ω\nStep 3: I = 12V/10Ω = 1.200A</think><answer>1.200</answer>"

# Проверка правильности
is_correct = game.verify(data, agent_response)
print(f"Правильно: {is_correct}")

# Градиентная оценка
accuracy_score = game.verifier.get_accuracy_score(data, agent_response)
print(f"Accuracy: {accuracy_score:.3f}")
```

## 🎯 Ключевые улучшения

### ✅ **Что упрощено:**
- **Только базовые топологии** (series/parallel)
- **Простые физические законы** (Ом + правила)
- **Оптимизированные параметры** обучения
- **Быстрая сходимость** (200 vs 500 шагов)

### ✅ **Что добавлено:**
- **Бонусная система** за правильный формат
- **Красивые диаграммы** результатов
- **Двойная метрика** (точность + формат)
- **Анализ улучшений** между методами

### ✅ **Что сохранено:**
- **Полная функциональность** обучения
- **Качественная верификация** ответов
- **Градиентная оценка** для reward
- **Comprehensive тестирование**

## 🚀 Результаты

### **Ожидаемые улучшения:**
- **Время обучения:** 1-2 часа (vs 12+ часов)
- **Точность:** 80-90% (vs 5-10% baseline)
- **Формат ответов:** 70-80% правильный формат
- **Память GPU:** 8-12GB (vs 20-30GB)

### **Готовность к продакшену:**
- ✅ **Все тесты проходят**
- ✅ **Оптимизированные параметры**
- ✅ **Красивая визуализация**
- ✅ **Полная документация**

## 📞 Поддержка

Для вопросов и предложений:
- **GitHub Issues:** [Создать issue](https://github.com/yourusername/tbank2/issues)
- **Email:** your.email@example.com

## 📄 Лицензия

MIT License - см. файл LICENSE для деталей.

---

**🎯 Система готова к обучению LLM решению задач анализа электрических цепей!**