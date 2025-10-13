# Исправления проблем в DC Circuit Analysis

## 🚨 Найденные и исправленные проблемы

### 1. ✅ Утечка данных в RL Trainer
**Файл:** `training/rl_trainer.py`

**Проблема:** 
Правильный ответ добавлялся в промпт для LLM через `<gold>` теги!
```python
# БЫЛО (НЕПРАВИЛЬНО):
{"role": "user", "content": f"{data.question}\n<gold>{float(data.answer):.3f}</gold>"}
```

**Исправлено:**
```python
# СТАЛО (ПРАВИЛЬНО):
{"role": "user", "content": data.question}  # Только вопрос, без ответа
```

---

### 2. ✅ Circuit не поддерживал множественные резисторы
**Файл:** `dc_circuit/solver.py`

**Проблема:**
`Circuit.resistors` был словарь `{(node1, node2): resistance}`. При добавлении нескольких резисторов между одними узлами (параллельное соединение) они **перезаписывали друг друга**!

```python
circuit.add_resistor("B", "C", 100)  # Добавляется
circuit.add_resistor("B", "C", 200)  # ПЕРЕЗАПИСЫВАЕТ! (тот же ключ)
# В словаре остается только {("B", "C"): 200}
```

**Исправлено:**
Изменен `Circuit.resistors` на **список кортежей**:
```python
self.resistors = []  # Список: [(node1, node2, resistance), ...]

def add_resistor(self, node1: str, node2: str, resistance: float):
    self.resistors.append((node1, node2, resistance))  # Добавляет, не перезаписывает!
```

**Обновлены все зависимые части:**
- `CircuitSolver.solve()` - теперь итерирует по списку
- `CircuitSolver.get_current()` - теперь суммирует токи через все резисторы между узлами
- Все калькуляторы обновлены

---

### 3. ✅ Генератор mixed цепей создавал неправильные цепи
**Файл:** `dc_circuit/generator.py`

**Проблема:**
```python
# Добавлял несколько параллельных резисторов
for i in range(parallel_count):
    circuit.add_resistor("B", "C", resistance)  # Все перезаписывались!
    resistors[f"R{i}"] = ("B", "C", resistance)
    
# Результат:
# Метаданные: 5 резисторов
# Circuit: 1 резистор! (остальные потерялись)
```

**Исправлено:**
Теперь генератор работает правильно т.к. `Circuit.resistors` - список:
```python
# Последовательный R1
circuit.add_resistor("A", "B", r1)

# Параллельные R2, R3, R4... между B и D
for i in range(parallel_count):
    circuit.add_resistor("B", "D", resistance)  # Каждый добавляется!
```

**Топология mixed цепи:**
```
A ---R1--- B ---R2--- D (ground)
               |---R3---|
               |---R4---|
```

---

### 4. ✅ EquivalentResistanceCalculator работал неправильно
**Файл:** `dc_circuit/calculators/equivalent_resistance.py`

**Проблема:**
Старая версия вычисляла ток неправильно (делила разность напряжений на напряжение источника вместо суммирования токов через резисторы).

**Исправлено:**
Теперь использует **метод мощности**:
```python
# Вычисляем мощность на каждом резисторе
for (n1, n2, R) in resistors:
    v_across = abs(V1 - V2)
    power = (v_across ** 2) / R
    total_power += power

# Общий ток из мощности
total_current = total_power / source_voltage

# Эквивалентное сопротивление
R_eq = source_voltage / total_current
```

**Теперь правильно работает для:**
- Последовательных цепей (series)
- Параллельных цепей (parallel)
- Смешанных цепей (mixed)

---

## ✅ Проверка исправлений

### Тест смешанной цепи
```
Цепь: V=12V, R1=2Ω, R2=6Ω || R3=3Ω
Ожидается:
  V_B = 6V
  I_R1 = 3A
  I_R2 = 1A
  I_R3 = 2A
  R_eq = 4Ω
  
Результат:
  V_B = 6.000V ✅
  I_R1 = 3.000A ✅
  I_R2 = 1.000A ✅
  I_R3 = 2.000A ✅
  R_eq = 4.0Ω ✅
  KCL выполнен ✅
```

### Тест генератора
```
Сложность 1 (series):
  Резисторов в метаданных: 3 ✅
  Резисторов в Circuit: 3 ✅
  
Сложность 2 (parallel):
  Резисторов в метаданных: 5 ✅
  Резисторов в Circuit: 5 ✅
  
Сложность 3 (mixed):
  Резисторов в метаданных: 3 ✅
  Резисторов в Circuit: 3 ✅
```

**Раньше:** метаданных 6, Circuit 2 (потеря 4 резисторов!) ❌
**Теперь:** количества совпадают ✅

---

## 📊 Итоговые изменения

**Изменено файлов:** 4
1. `training/rl_trainer.py` - убрали утечку данных
2. `dc_circuit/solver.py` - Circuit.resistors теперь список
3. `dc_circuit/generator.py` - упрощенный правильный mixed генератор
4. `dc_circuit/calculators/equivalent_resistance.py` - метод мощности

**Создано файлов:** 1
1. `training/evaluate_simple.py` - упрощенный evaluation скрипт

**Все тесты:** ✅ Пройдены

---

## 🚀 Что дальше

1. Запустить `training/evaluate_simple.py` для получения baseline метрик
2. Обучить модель через `training/rl_trainer.py` (требует GPU)
3. Повторно оценить после обучения

---

## 📝 Важные замечания

- **KCL в узлах с источниками:** Узлы подключенные к источникам напряжения могут иметь I_in ≠ I_out (ток идет от источника). Это нормально!
- **Промежуточные узлы:** Все промежуточные узлы (без источников) должны выполнять KCL
- **Физическая корректность:** Solver теперь правильно решает все типы цепей (series, parallel, mixed)
