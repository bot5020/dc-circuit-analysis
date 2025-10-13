"""
Комплексная проверка всей системы DC Circuit Analysis
Проверяет каждый компонент на реальных данных
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from dc_circuit.game import DCCircuitGame
from dc_circuit.generator import CircuitGenerator
from dc_circuit.solver import CircuitSolver
from dc_circuit.calculators import get_calculator_registry
from config import CircuitConfig, VerifierConfig


def test_physics_laws(circuit, node_voltages, metadata):
    """Проверка законов Кирхгофа"""
    issues = []
    
    # Проверка KCL для всех узлов (кроме источников и ground)
    ground = metadata['ground_node']
    source_nodes = set()
    for (n_pos, n_neg) in metadata['voltage_sources'].keys():
        source_nodes.update([n_pos, n_neg])
    
    for node in metadata['nodes']:
        if node == ground or node in source_nodes:
            continue
        
        v_node = node_voltages.get(node, 0.0)
        i_in = 0.0
        i_out = 0.0
        
        for (n1, n2, r) in circuit.resistors:
            if n1 == node:
                i_out += (v_node - node_voltages.get(n2, 0.0)) / r
            elif n2 == node:
                i_in += (node_voltages.get(n1, 0.0) - v_node) / r
        
        if abs(i_in - i_out) > 0.01:  # 10mA погрешность
            issues.append(f"KCL нарушен в узле {node}: I_in={i_in:.6f}A, I_out={i_out:.6f}A")
    
    return issues


def test_calculator(calc_name, circuit, node_voltages, metadata, target_resistor):
    """Тестирование конкретного калькулятора"""
    solver = CircuitSolver()
    calculators = get_calculator_registry(solver, 3)
    
    if calc_name not in calculators:
        return None, f"Калькулятор {calc_name} не найден"
    
    calc = calculators[calc_name]
    
    try:
        result = calc.calculate(circuit, node_voltages, metadata, target_resistor)
        
        if result is None:
            return None, f"Калькулятор вернул None"
        
        # Проверка разумности результата
        if calc_name == "current":
            if result < 0 or result > 1000:  # Токи в разумных пределах
                return result, f"Подозрительное значение тока: {result}A"
        elif calc_name == "voltage":
            source_voltage = list(metadata['voltage_sources'].values())[0]
            if result < 0 or result > source_voltage * 1.1:  # Напряжение не больше источника
                return result, f"Подозрительное напряжение: {result}V (источник {source_voltage}V)"
        elif calc_name == "equivalent_resistance":
            if result <= 0 or result > 1e6:
                return result, f"Подозрительное сопротивление: {result}Ω"
        
        return result, None
    except Exception as e:
        return None, f"Ошибка: {str(e)}"


def test_metadata_consistency(circuit, metadata):
    """Проверка что metadata соответствует circuit"""
    issues = []
    
    # Проверка количества резисторов
    metadata_count = len(metadata['resistors'])
    circuit_count = len(circuit.resistors)
    
    if metadata_count != circuit_count:
        issues.append(f"Несоответствие резисторов: metadata={metadata_count}, circuit={circuit_count}")
    
    # Проверка что все резисторы из metadata есть в circuit
    for r_name, (n1, n2, r_val) in metadata['resistors'].items():
        found = False
        for (c_n1, c_n2, c_r) in circuit.resistors:
            if ((c_n1 == n1 and c_n2 == n2) or (c_n1 == n2 and c_n2 == n1)) and abs(c_r - r_val) < 0.01:
                found = True
                break
        if not found:
            issues.append(f"{r_name} ({n1}-{n2}, {r_val}Ω) не найден в circuit")
    
    return issues


def comprehensive_test():
    """Полная проверка системы"""
    print("="*80)
    print("🧪 КОМПЛЕКСНАЯ ПРОВЕРКА СИСТЕМЫ DC CIRCUIT ANALYSIS")
    print("="*80)
    
    config = CircuitConfig()
    verifier_config = VerifierConfig()
    game = DCCircuitGame(config, verifier_config)
    gen = CircuitGenerator(config)
    solver = CircuitSolver()
    
    total_tests = 0
    passed_tests = 0
    
    # Тестируем каждую сложность
    for difficulty in [1, 2, 3]:
        circuit_types = ["series", "parallel", "mixed"]
        circuit_type = circuit_types[difficulty - 1]
        
        print(f"\n{'='*80}")
        print(f"📊 СЛОЖНОСТЬ {difficulty} ({circuit_type.upper()})")
        print(f"{'='*80}")
        
        # Генерируем несколько примеров
        for test_num in range(3):
            print(f"\n  🔬 Тест #{test_num + 1}:")
            total_tests += 1
            test_passed = True
            
            try:
                # Генерация
                circuit, q_type, metadata = gen.generate_circuit(difficulty=difficulty)
                
                print(f"    Тип вопроса: {q_type}")
                print(f"    Целевой резистор: {metadata['target_resistor']}")
                print(f"    Напряжение источника: {metadata['voltage_source']}V")
                print(f"    Резисторов: {len(metadata['resistors'])}")
                
                # 1. Проверка metadata consistency
                print(f"\n    ✓ Проверка metadata consistency:")
                issues = test_metadata_consistency(circuit, metadata)
                if issues:
                    print(f"      ❌ Найдены проблемы:")
                    for issue in issues:
                        print(f"         - {issue}")
                    test_passed = False
                else:
                    print(f"      ✅ Metadata соответствует circuit")
                
                # 2. Решение цепи
                node_voltages = solver.solve(circuit)
                
                if not node_voltages:
                    print(f"      ❌ Solver не смог решить цепь")
                    test_passed = False
                    continue
                
                print(f"      ✅ Цепь решена ({len(node_voltages)} узлов)")
                
                # 3. Проверка физических законов
                print(f"\n    ✓ Проверка законов Кирхгофа:")
                physics_issues = test_physics_laws(circuit, node_voltages, metadata)
                if physics_issues:
                    print(f"      ❌ Найдены нарушения:")
                    for issue in physics_issues:
                        print(f"         - {issue}")
                    test_passed = False
                else:
                    print(f"      ✅ KCL выполнен для всех промежуточных узлов")
                
                # 4. Тестирование всех калькуляторов
                print(f"\n    ✓ Проверка калькуляторов:")
                target = metadata['target_resistor']
                
                for calc_name in ['current', 'voltage', 'equivalent_resistance']:
                    result, error = test_calculator(calc_name, circuit, node_voltages, metadata, target)
                    
                    if error:
                        print(f"      ❌ {calc_name}: {error}")
                        test_passed = False
                    else:
                        print(f"      ✅ {calc_name}: {result}")
                
                # 5. Проверка через game.generate (интеграция)
                print(f"\n    ✓ Проверка интеграции (game.generate):")
                data_list = game.generate(num_of_questions=1, difficulty=difficulty)
                
                if not data_list:
                    print(f"      ❌ game.generate вернул пустой список")
                    test_passed = False
                else:
                    data = data_list[0]
                    print(f"      ✅ Задача сгенерирована")
                    print(f"         Ответ: {data.answer}")
                    
                    # Проверка верификации
                    test_response = f"<think>test</think><answer>{data.answer}</answer>"
                    is_correct = game.verify(data, test_response)
                    
                    if is_correct:
                        print(f"      ✅ Верификация работает")
                    else:
                        print(f"      ❌ Верификация не работает")
                        test_passed = False
                
                # 6. Проверка специфических случаев для каждого типа цепи
                print(f"\n    ✓ Специфические проверки для {circuit_type}:")
                
                if circuit_type == "series":
                    # Проверка что ток одинаков через все резисторы
                    currents = []
                    for (n1, n2, r) in circuit.resistors:
                        v1 = node_voltages.get(n1, 0.0)
                        v2 = node_voltages.get(n2, 0.0)
                        i = abs(v1 - v2) / r
                        currents.append(i)
                    
                    if len(set([round(i, 4) for i in currents])) == 1:
                        print(f"      ✅ Ток одинаков через все резисторы: {currents[0]:.6f}A")
                    else:
                        print(f"      ❌ Токи различаются: {[f'{i:.6f}' for i in currents]}")
                        test_passed = False
                
                elif circuit_type == "parallel":
                    # Проверка что напряжение одинаково на всех резисторах
                    if len(metadata['nodes']) == 2:  # Простая параллельная цепь
                        n1, n2 = metadata['nodes']
                        v_diff = abs(node_voltages.get(n1, 0) - node_voltages.get(n2, 0))
                        v_source = metadata['voltage_source']
                        
                        if abs(v_diff - v_source) < 0.01:
                            print(f"      ✅ Напряжение на параллельных резисторах: {v_diff:.3f}V")
                        else:
                            print(f"      ❌ Напряжение не совпадает: {v_diff:.3f}V != {v_source}V")
                            test_passed = False
                    else:
                        print(f"      ⚠️  Сложная топология, пропускаем проверку")
                
                elif circuit_type == "mixed":
                    # Проверка что есть и последовательные и параллельные части
                    # Считаем резисторы между одними узлами
                    node_pairs = {}
                    for (n1, n2, r) in circuit.resistors:
                        key = tuple(sorted([n1, n2]))
                        node_pairs[key] = node_pairs.get(key, 0) + 1
                    
                    has_parallel = any(count > 1 for count in node_pairs.values())
                    has_series = len(circuit.resistors) > max(node_pairs.values())
                    
                    if has_parallel:
                        print(f"      ✅ Есть параллельные резисторы")
                    else:
                        print(f"      ⚠️  Нет параллельных резисторов (может быть ошибка генерации)")
                    
                    if has_series:
                        print(f"      ✅ Есть последовательные резисторы")
                
                if test_passed:
                    passed_tests += 1
                    print(f"\n    ✅ ТЕСТ ПРОЙДЕН")
                else:
                    print(f"\n    ❌ ТЕСТ НЕ ПРОЙДЕН")
                
            except Exception as e:
                print(f"      ❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
                import traceback
                traceback.print_exc()
                test_passed = False
    
    # Итоговая статистика
    print(f"\n{'='*80}")
    print(f"📊 ИТОГОВАЯ СТАТИСТИКА")
    print(f"{'='*80}")
    print(f"  Всего тестов: {total_tests}")
    print(f"  Пройдено: {passed_tests}")
    print(f"  Провалено: {total_tests - passed_tests}")
    print(f"  Успешность: {passed_tests / total_tests * 100:.1f}%")
    
    if passed_tests == total_tests:
        print(f"\n  🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
    else:
        print(f"\n  ⚠️  ЕСТЬ ПРОБЛЕМЫ, ТРЕБУЕТСЯ ИСПРАВЛЕНИЕ")
    
    print(f"{'='*80}")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = comprehensive_test()
    sys.exit(0 if success else 1)
