# 🤖 Полная Система RL Обучения для Анализа Электрических Цепей

## 🎯 Обзор

**`rl_trainer.py`** - это полная система reinforcement learning для обучения моделей анализа электрических цепей с использованием **GRPO (Generative Reward Policy Optimization)** алгоритма.

### 🔧 Основные возможности:

- ✅ **Настоящее RL обучение** с обновлением весов модели
- ✅ **GRPO алгоритм** с reward function на основе точности
- ✅ **LoRA адаптеры** для эффективного fine-tuning
- ✅ **Градиентная система вознаграждений** (0.0, 0.25, 0.5, 0.75, 1.0)
- ✅ **Автоматическое сохранение** моделей и статистики
- ✅ **Поддержка GPU** с flash attention
- ✅ **Прерывание и возобновление** обучения

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
# Базовые зависимости
pip install -r requirements.txt

# Дополнительные зависимости для RL обучения
pip install trl peft bitsandbytes accelerate
```

### 2. Запуск обучения

```bash
# Базовое обучение с unsloth
python3 training/rl_trainer.py --model unsloth/Qwen2.5-3B-Instruct --max-steps 250

# С кастомными параметрами (рекомендуется)
python3 training/rl_trainer.py \
  --model unsloth/Qwen2.5-3B-Instruct \
  --max-steps 500 \
  --learning-rate 5e-6 \
  --batch-size 1 \
  --num-generations 8 \
  --lora-r 64 \
  --output-dir ./dc_circuit_model_rl

# Для больших моделей (7B+)
python3 training/rl_trainer.py \
  --model unsloth/Qwen2.5-7B-Instruct \
  --max-steps 1000 \
  --batch-size 1 \
  --lora-r 128 \
  --gpu-memory-utilization 0.8
```

### 3. Параметры командной строки

| Параметр | Описание | По умолчанию |
|----------|----------|---------------|
| `--model` | Название базовой модели | unsloth/Qwen2.5-3B-Instruct |
| `--output-dir` | Директория для сохранения модели | ./dc_circuit_model_rl |
| `--max-seq-length` | Максимальная длина последовательности | 1024 |
| `--max-steps` | Максимальное количество шагов | 250 |
| `--learning-rate` | Скорость обучения | 5e-6 |
| `--batch-size` | Размер батча | 1 |
| `--gradient-accumulation` | Gradient accumulation steps | 1 |
| `--num-generations` | Количество генераций на шаг | 8 |
| `--lora-r` | LoRA rank | 64 |
| `--lora-alpha` | LoRA alpha | 64 |
| `--lora-dropout` | LoRA dropout | 0.05 |
| `--load-in-4bit` | 4-bit квантизация | True |
| `--fast-inference` | vLLM ускорение | True |
| `--gpu-memory-utilization` | Использование GPU памяти | 0.9 |
| `--temperature` | Sampling temperature | 0.7 |
| `--beta` | KL divergence coefficient | 0.04 |

## 🔍 Архитектура системы

### **1. GRPO Алгоритм**

**GRPO** = Generative Reward Policy Optimization

**Основная идея:**
- Модель генерирует несколько ответов для каждого промпта
- Каждый ответ оценивается reward function
- Модель обновляется для максимизации ожидаемого вознаграждения

### **2. Reward Function**

```python
def reward_function(completions, prompts, **kwargs):
    for completion, prompt in zip(completions, prompts):
        # Извлекаем правильный ответ из <gold> тегов
        gold = extract_from_gold_tags(prompt)

        # Извлекаем ответ модели из <answer> тегов
        model_answer = extract_from_answer_tags(completion)

        # Сравниваем через верификатор
        accuracy = verifier.get_accuracy_score(data_obj, model_answer)

        # Возвращаем градиентное вознаграждение
        reward = accuracy  # 0.0, 0.25, 0.5, 0.75 или 1.0
```

### **3. Формат данных**

**Промпт для обучения:**
```text
You are an expert circuit analysis engineer...

Question: Найдите ток через R1 (в Амперах)
Answer: <gold>0.123</gold>
```

**Формат ответа модели:**
```text
<think>Рассуждаем шаг за шагом...</think>
<answer>0.123</answer>
```

### **4. GRPO + LoRA + unsloth = Максимальная эффективность**

**Как работает в нашем коде:**

1. **Загрузка с unsloth (ускоренная):**
   ```python
   model, tokenizer = FastLanguageModel.from_pretrained(
       "unsloth/Qwen2.5-3B-Instruct",
       max_seq_length=1024,
       load_in_4bit=True,      # 4-bit квантизация
       fast_inference=True,    # vLLM ускорение
       max_lora_rank=64
   )
   ```

2. **Расширенные LoRA адаптеры:**
   ```python
   model = FastLanguageModel.get_peft_model(
       model,
       r=64,                           # Высокий rank для качества
       target_modules=[
           "q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj"  # Все модули
       ],
       lora_alpha=64,
       use_gradient_checkpointing="unsloth"
   )
   ```

3. **Множественные Reward Functions:**
   ```python
   reward_functions = [
       xml_count_reward_func,      # Подсчет XML тегов
       soft_format_reward_func,    # Мягкий формат
       strict_format_reward_func,  # Строгий формат
       numeric_reward_func,        # Числовой ответ
       correctness_reward_func     # Точность ответа (2.0/0.0)
   ]
   ```

4. **GRPO с vLLM ускорением:**
   ```python
   trainer = GRPOTrainer(
       model=model,
       processing_class=tokenizer,
       reward_funcs=reward_functions,  # 5 reward functions
       args=GRPOConfig(use_vllm=True), # vLLM для inference
       train_dataset=dataset
   )
   ```

### **5. vLLM и unsloth Интеграция**

**Как используется в коде:**

#### **unsloth (опционально):**
```python
# Если unsloth установлен:
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-3B-Instruct",
    max_seq_length=1024,
    load_in_4bit=True,      # 4-bit квантизация
    fast_inference=True,    # vLLM ускорение
    max_lora_rank=64,       # Максимальный rank для LoRA
    gpu_memory_utilization=0.9
)
```

#### **vLLM в GRPO:**
```python
# В GRPOTrainer:
training_args = GRPOConfig(
    use_vllm=True,  # ← ВКЛЮЧАЕТ vLLM ДЛЯ INFERENCE
    num_generations=8,  # Генерирует 8 ответов на промпт
    max_prompt_length=256,
    max_completion_length=200
)
```

**Что делает vLLM:**
- ⚡ **Ускоряет inference** в 10-100 раз
- 🎯 **Позволяет генерировать много ответов** параллельно
- 🧠 **Не влияет на обучение** - только на скорость inference

#### **Fallback (если unsloth недоступен):**
```python
# Стандартный подход:
model = AutoModelForCausalLM.from_pretrained(...)
tokenizer = AutoTokenizer.from_pretrained(...)
# Работает медленнее, но стабильно
```

### **6. Множественные Reward Functions**

**Система вознаграждений:**
- **XML Count (0.5 балла)**: Правильное количество тегов
- **Soft Format (0.5 балла)**: Наличие тегов в любом порядке
- **Strict Format (0.5 балла)**: Теги в правильном порядке
- **Numeric (0.5 балла)**: Ответ является числом
- **Correctness (2.0 балла)**: Полная точность ответа

**Максимум:** 4.5 балла за идеальный ответ

## 📊 Мониторинг обучения

### **Логи во время обучения:**
```
🤖 RL ОБУЧЕНИЕ МОДЕЛИ АНАЛИЗА ЦЕПЕЙ
============================================================
🔧 Инициализация RL тренера...
📋 Модель: Qwen/Qwen2.5-1.5B-Instruct
🎯 Выходная директория: ./dc_circuit_model_rl
🔄 Загрузка модели и токенизатора...
✅ Модель и токенизатор настроены
🎯 Настройка GRPO тренера...
✅ Тренер настроен
🚀 НАЧИНАЕМ ОБУЧЕНИЕ!
============================================================
📊 Шаг 10/500 | Среднее вознаграждение: 0.234
📊 Шаг 20/500 | Среднее вознаграждение: 0.456
💾 Сохранена статистика на шаге 200
✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!
💾 Модель сохранена в: ./dc_circuit_model_rl
```

### **Сохраненные файлы:**

1. **`training_stats.json`** - статистика обучения
2. **`adapter_model.bin`** - LoRA адаптер
3. **`adapter_config.json`** - конфигурация адаптера
4. **`tokenizer.json`** - токенизатор

## 🎨 Оценка обученной модели

```python
from training.evaluate import generate_full_report

# Оцениваем модель
baseline_results, trained_results = generate_full_report(
    baseline_model="Qwen/Qwen2.5-1.5B-Instruct",
    trained_model="./dc_circuit_model_rl"
)
```

## 🔧 Технические детали

### **Зависимости:**
```txt
torch>=2.0.0
transformers>=4.36.0
trl>=0.8.0
accelerate>=0.25.0
peft>=0.7.0
bitsandbytes>=0.41.0
```

### **GPU требования:**
- **Минимально:** 8GB VRAM для 1.5B модели
- **Рекомендую:** 16GB+ VRAM для стабильной работы
- **Оптимизации:** Flash Attention 2, bfloat16

### **Время обучения:**
- **100 шагов:** ~5-10 минут
- **500 шагов:** ~30-60 минут
- **1000 шагов:** ~1-2 часа

## 🚨 Важные замечания

### **Отличия от симуляции:**
- ✅ **Настоящее обновление весов** модели
- ✅ **Использование GPU** для ускорения
- ✅ **Сохранение адаптеров** LoRA
- ✅ **Прерывание и возобновление**

### **⚠️ Требуемые библиотеки:**

#### **Минимальные (работают без GPU):**
```bash
pip install torch transformers trl peft accelerate
```

#### **Рекомендуемые (с GPU ускорением):**
```bash
pip install unsloth trl peft bitsandbytes accelerate
```

#### **Полный набор:**
```bash
pip install torch transformers unsloth trl peft accelerate bitsandbytes
```

### **Ограничения:**
- ⚠️ Требует много GPU памяти (8GB+ для 1.5B модели)
- ⚠️ Долгое время обучения (30-60 мин для 500 шагов)
- ⚠️ Сложная настройка зависимостей
- ⚠️ Требует CUDA для GPU ускорения

## 🎉 Результаты

После обучения модель должна показать **значительное улучшение** в решении задач анализа электрических цепей с **градиентной системой оценки** вместо бинарной.

**Пример улучшения:**
- **До:** 45% точность
- **После:** 78% точность
- **Улучшение:** +33%

---

*Создано для проекта анализа электрических цепей с reinforcement learning* 🚀⚡
