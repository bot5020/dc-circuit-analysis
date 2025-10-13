#!/usr/bin/env python3
"""Проверка: откуда берется correct_answer в RL обучении"""

from dc_circuit.game import DCCircuitGame
from config import CircuitConfig, VerifierConfig

print("=" * 80)
print("ПРОВЕРКА ПОТОКА ДАННЫХ: от генерации до датасета")
print("=" * 80)

# Создаем game как в RL trainer
circuit_config = CircuitConfig()
verifier_config = VerifierConfig()
game = DCCircuitGame(circuit_config, verifier_config)

print("\n1️⃣ Генерируем задачи (как в DCCircuitDataset._generate_data):")
data_list = game.generate(num_of_questions=5, difficulty=1)

print(f"\n   Сгенерировано: {len(data_list)} задач\n")

# Проверяем каждую задачу
print("2️⃣ Проверяем Data.answer для каждой задачи:\n")

for i, data in enumerate(data_list):
    print(f"{'='*70}")
    print(f"ЗАДАЧА #{i+1}")
    print(f"{'='*70}")
    
    metadata = data.metadata
    V = metadata.get('voltage_source')
    resistors = metadata.get('resistors', {})
    target = metadata.get('target_resistor')
    question_type = metadata.get('question_type')
    
    print(f"\n📋 Параметры цепи:")
    print(f"   V = {V} V")
    for name, (n1, n2, R) in resistors.items():
        print(f"   {name} ({n1}-{n2}) = {R} Ω")
    print(f"   Вопрос: {question_type} на {target}")
    
    print(f"\n📝 Ответ из Data.answer: {data.answer}")
    
    # Ручной пересчет для сравнения
    if question_type == 'voltage' and metadata.get('circuit_type') == 'series':
        R_total = sum(r[2] for r in resistors.values())
        I = V / R_total
        target_R = resistors[target][2]
        V_manual = I * target_R
        
        print(f"\n🧮 Ручной расчет:")
        print(f"   R_total = {R_total} Ω")
        print(f"   I = {V}/{R_total} = {I:.6f} A")
        print(f"   V({target}) = {I:.6f} × {target_R} = {V_manual:.6f} V")
        print(f"   Округлено: {round(V_manual, 3)} V")
        
        print(f"\n📊 Сравнение:")
        print(f"   Data.answer:    {data.answer}")
        print(f"   Ручной расчет:  {round(V_manual, 3)}")
        
        if abs(float(data.answer) - round(V_manual, 3)) < 0.001:
            print(f"   ✅ СОВПАДАЮТ")
        else:
            print(f"   ❌ НЕ СОВПАДАЮТ! Разница: {abs(float(data.answer) - round(V_manual, 3)):.6f}")
    
    print()

print("\n" + "=" * 80)
print("3️⃣ Эмулируем DCCircuitDataset (форматирование):")
print("=" * 80)

dataset_items = []
for data in data_list:
    # Такое же форматирование как в DCCircuitDataset.__init__
    item = {
        "question": data.question,
        "answer": f"{float(data.answer):.3f}",  # ← ЗДЕСЬ форматируется ответ
        "difficulty": data.difficulty
    }
    dataset_items.append(item)

print("\nПроверка форматирования:")
for i, item in enumerate(dataset_items[:3]):  # Первые 3
    print(f"\n   Задача #{i+1}:")
    print(f"   answer в датасете: {item['answer']}")

print("\n" + "=" * 80)
print("4️⃣ Эмулируем reward_function (где берется correct_answer):")
print("=" * 80)

# Как в reward_function:
idx = 0
if idx < len(dataset_items):
    correct_answer = dataset_items[idx]["answer"]
    print(f"\n   correct_answer = dataset[{idx}]['answer'] = {correct_answer}")
    print(f"\n   ← ЭТО ЗНАЧЕНИЕ попадает в логи как 'correct_answer'")

print("\n" + "=" * 80)
print("ИТОГ:")
print("=" * 80)

print("\nЕсли видишь неправильные ответы выше:")
print("   → Проблема в DCCircuitGame._calculate_answer()")
print("\nЕсли ответы правильные:")
print("   → Проблема в старых данных или кэше")
print("\n" + "=" * 80)
