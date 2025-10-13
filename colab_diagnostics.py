#!/usr/bin/env python3
"""
ДИАГНОСТИКА ДЛЯ COLAB
Запусти этот скрипт В COLAB перед обучением
"""
import sys
import os
import shutil
from pathlib import Path


print("=" * 80)
print("🔍 ДИАГНОСТИКА COLAB ПЕРЕД ОБУЧЕНИЕМ")
print("=" * 80)

# 1. ОЧИСТКА КЭША
print("\n1️⃣  ОЧИСТКА PYTHON КЭША:")
cache_dirs = list(Path('.').rglob('__pycache__'))
pyc_files = list(Path('.').rglob('*.pyc'))

print(f"   Найдено __pycache__ директорий: {len(cache_dirs)}")
print(f"   Найдено *.pyc файлов: {len(pyc_files)}")

if cache_dirs or pyc_files:
    print("   🧹 Удаляю кэш...")
    for cache_dir in cache_dirs:
        try:
            shutil.rmtree(cache_dir)
            print(f"      ✅ Удален: {cache_dir}")
        except:
            pass
    
    for pyc in pyc_files:
        try:
            pyc.unlink()
        except:
            pass
    
    print("   ✅ Кэш очищен!")
else:
    print("   ✅ Кэш уже чист")

# 2. ПРОВЕРКА ВЕРСИИ solver.py
print("\n2️⃣  ПРОВЕРКА ВЕРСИИ solver.py:")

solver_path = Path("dc_circuit/solver.py")
if solver_path.exists():
    content = solver_path.read_text()
    
    # Ищем маркеры исправленного кода
    markers = {
        "_order_series_resistors": "✅ Метод упорядочивания резисторов",
        "V_node1 = node_voltages.get(node1": "✅ Исправленный get_current (использует node1/node2 из аргументов)",
        "def _find_path": "✅ Метод поиска пути в графе"
    }
    
    print("   Проверка маркеров исправленного кода:")
    all_ok = True
    for marker, description in markers.items():
        if marker in content:
            print(f"      ✅ {description}")
        else:
            print(f"      ❌ {description} - НЕ НАЙДЕН!")
            all_ok = False
    
    if all_ok:
        print("\n   ✅ solver.py содержит ВСЕ исправления!")
    else:
        print("\n   ❌ solver.py НЕПОЛНЫЙ! Нужно обновить из Git!")
        print("   Выполни: !git pull")
else:
    print("   ❌ solver.py не найден!")

# 3. ТЕСТ ГЕНЕРАЦИИ
print("\n3️⃣  ТЕСТ ГЕНЕРАЦИИ ДАННЫХ:")

try:
    sys.path.insert(0, '.')
    
    from dc_circuit.game import DCCircuitGame
    from dc_circuit.solver import Circuit, CircuitSolver
    from dc_circuit.calculators.voltage import VoltageCalculator
    import random
    
    print("   Генерирую тестовую задачу...")
    
    # Прямой тест проблемной задачи
    circuit = Circuit()
    circuit.add_voltage_source("A", "C", 9.0)
    circuit.set_ground("C")
    circuit.add_resistor("A", "B", 44.0)
    circuit.add_resistor("B", "C", 94.0)
    
    solver = CircuitSolver()
    node_voltages = solver.solve(circuit)
    
    metadata = {'resistors': {'R1': ('A', 'B', 44.0), 'R2': ('B', 'C', 94.0)}}
    voltage_calc = VoltageCalculator(solver, 3)
    answer = voltage_calc.calculate(circuit, node_voltages, metadata, 'R2')
    
    # Ручной расчет
    expected = round(9.0 / 138.0 * 94.0, 3)
    
    print(f"\n   Тестовая задача: V=9V, R1=44Ω, R2=94Ω")
    print(f"   Вопрос: Напряжение на R2")
    print(f"   Ответ системы: {answer} V")
    print(f"   Ожидаемый:     {expected} V")
    
    if abs(answer - expected) < 0.001:
        print(f"   ✅ ТЕСТ ПРОЙДЕН! Система работает правильно!")
    else:
        print(f"   ❌ ТЕСТ ПРОВАЛЕН! Система возвращает неправильный ответ!")
        print(f"   ⚠️  ПРОБЛЕМА: Загружена старая версия кода или кэш не очищен!")
        
except Exception as e:
    print(f"   ❌ ОШИБКА при импорте: {e}")
    print(f"   Проверь, что все файлы загружены правильно")

# 4. ПРОВЕРКА RUNTIME
print("\n4️⃣  ИНФОРМАЦИЯ О RUNTIME:")
print(f"   Python версия: {sys.version}")
print(f"   Текущая директория: {os.getcwd()}")
print(f"   sys.path:")
for p in sys.path[:5]:
    print(f"      {p}")

# 5. ПЕРЕЗАГРУЗКА МОДУЛЕЙ
print("\n5️⃣  ПЕРЕЗАГРУЗКА МОДУЛЕЙ:")
print("   ⚠️  Если ты уже импортировал модули в Colab:")
print("      Они могут быть закэшированы в памяти!")
print()
print("   🔄 Для полной перезагрузки выполни:")
print("      import importlib")
print("      import sys")
print()
print("      # Удаляем все наши модули из памяти")
print("      modules_to_remove = [k for k in sys.modules.keys() if k.startswith('dc_circuit') or k.startswith('base') or k.startswith('training')]")
print("      for module in modules_to_remove:")
print("          del sys.modules[module]")
print()
print("      # Теперь можно импортировать заново")
print("      from training.rl_trainer import DCCircuitRLTrainer")

# 6. ИТОГ
print("\n" + "=" * 80)
print("📋 ЧЕКЛИСТ ПЕРЕД ОБУЧЕНИЕМ В COLAB")
print("=" * 80)

print("""
ВЫПОЛНИ ЭТИ ШАГИ В COLAB:

1. 🔄 ОБНОВИ КОД ИЗ GIT:
   !git pull origin main
   
2. 🧹 ОЧИСТИ КЭШ:
   !find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
   !find . -name "*.pyc" -delete 2>/dev/null
   
3. 🔄 ПЕРЕЗАГРУЗИ RUNTIME:
   Runtime → Restart runtime (ВАЖНО!)
   
4. ✅ ЗАПУСТИ ЭТОТ СКРИПТ:
   !python colab_diagnostics.py
   
5. 🎯 ЕСЛИ ВСЕ ✅ - ЗАПУСКАЙ ОБУЧЕНИЕ:
   from training.rl_trainer import DCCircuitRLTrainer
   trainer = DCCircuitRLTrainer()
   trainer.run()

⚠️  КРИТИЧЕСКИ ВАЖНО:
   - Не используй старые ячейки с импортами!
   - Перезагрузи runtime перед обучением!
   - Убедись, что тест (шаг 3️⃣) возвращает 6.13 V, а не 3.977 V
""")

print("\n" + "=" * 80)
print("✅ ДИАГНОСТИКА ЗАВЕРШЕНА")
print("=" * 80)
