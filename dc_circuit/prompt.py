def create_circuit_prompt(metadata: dict, question_type: str, target_resistor: str) -> str:
    """
    Создает промпт для задач анализа DC цепей
    
    @param metadata: метаданные цепи
    @param question_type: тип вопроса (current, voltage, power, и т.д.)
    @param target_resistor: целевой резистор для анализа
    @return: промпт на английском языке
    """
    circuit_type = metadata.get("circuit_type", "unknown")
    voltage = metadata.get("voltage_source", 10)
    resistors = metadata.get("resistors", {})

    # Создаем подробное описание цепи
    resistor_details = []
    for r_name, (n1, n2, r_val) in resistors.items():
        resistor_details.append(f"{r_name}={r_val}Ω (between nodes {n1} and {n2})")
    
    resistor_list = ", ".join(resistor_details)
    
    if circuit_type == "series":
        circuit_desc = f"Series circuit with voltage source V={voltage}V and resistors: {resistor_list}"
    elif circuit_type == "parallel":
        circuit_desc = f"Parallel circuit with voltage source V={voltage}V and resistors: {resistor_list}"
    elif circuit_type == "mixed":
        circuit_desc = f"Mixed (series-parallel) circuit with voltage source V={voltage}V and resistors: {resistor_list}"
    else:
        circuit_desc = f"Complex circuit with voltage source V={voltage}V and resistors: {resistor_list}"

    # Формулируем вопрос
    if question_type == "current":
        question = f"Find the current through {target_resistor} (in Amperes)"
    elif question_type == "voltage":
        question = f"Find the voltage across {target_resistor} (in Volts)"
    elif question_type == "power":
        question = f"Find the power dissipated by {target_resistor} (in Watts)"
    elif question_type == "total_current":
        question = "Find the total current from the voltage source (in Amperes)"
    elif question_type == "equivalent_resistance":
        question = "Find the equivalent resistance of the circuit (in Ohms)"
    elif question_type == "voltage_divider":
        question = f"Find the voltage across {target_resistor} using voltage divider rule (in Volts)"
    elif question_type == "current_divider":
        question = f"Find the current through {target_resistor} using current divider rule (in Amperes)"
    elif question_type == "power_total":
        question = "Find the total power dissipated by all resistors (in Watts)"
    else:
        question = f"Find the {question_type} for {target_resistor}"

    # Создаем полный промпт с всеми правилами
    prompt = f"""DC Circuit Analysis Problem

Fundamental Laws and Principles:

1. Ohm's Law: V = I × R
   - Voltage (V) equals Current (I) times Resistance (R)
   - Can be rearranged: I = V/R or R = V/I

2. Kirchhoff's Current Law (KCL): 
   - The sum of currents entering a node equals the sum of currents leaving
   - ΣI_in = ΣI_out

3. Kirchhoff's Voltage Law (KVL): 
   - The sum of voltages around any closed loop equals zero
   - ΣV = 0

4. Series Connection:
   - Same current flows through all resistors: I_total = I₁ = I₂ = I₃
   - Voltages add up: V_total = V₁ + V₂ + V₃
   - Total resistance: R_total = R₁ + R₂ + R₃

5. Parallel Connection:
   - Same voltage across all resistors: V_total = V₁ = V₂ = V₃
   - Currents add up: I_total = I₁ + I₂ + I₃
   - Total resistance: 1/R_total = 1/R₁ + 1/R₂ + 1/R₃

6. Power: P = I²R = V²/R = VI

Circuit Description:
{circuit_desc}

Question to answer:
{question}

Instructions:
- Show your reasoning step by step inside <think> tags
- Apply the appropriate laws (Ohm's Law, KCL, KVL)
- Provide ONLY the final numerical answer with exactly 3 decimal places inside <answer> tags
- Do NOT include units in the answer (just the number)

Example Format:

question:
Find the current through R1 (in Amperes)

solution:
<think>
Step 1: Apply Ohm's Law to find current: I = V/R = 12V / 4Ω = 3.000 A
</think>
<answer>3.000</answer>
"""

    return prompt