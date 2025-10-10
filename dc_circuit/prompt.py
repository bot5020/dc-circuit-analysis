def create_circuit_prompt(metadata: dict, question_type: str, target_resistor: str) -> str:
    """
    Создает промпт для задач анализа DC цепей

    Args:
        metadata: Метаданные цепи
        question_type: Тип вопроса (current, voltage, power, и т.д.)
        target_resistor: Целевой резистор для анализа

    Returns:
        Промпт на английском языке
    """
    circuit_type = metadata.get("circuit_type", "unknown")
    voltage = metadata.get("voltage_source", 10)
    resistors = metadata.get("resistors", {})

    # Создание подробного описания цепи
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

    # Формулирование вопроса
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

    prompt = f"""You are an expert circuit analysis engineer.
            Solve electrical circuit problems using physics laws.

            FUNDAMENTAL LAWS:
            1. Ohm: V=IR, I=V/R
            2. KCL: ΣI_in=ΣI_out
            3. KVL: ΣV=0
            4. Series: R_total=R₁+R₂+..., I_total=I₁=I₂
            5. Parallel: 1/R_total=1/R₁+1/R₂+..., V_total=V₁=V₂
            6. Power: P=I²R=V²/R

            Circuit: {circuit_desc}

            Question: {question}

            YOU MUST USE THE FOLLOWING FORMAT:
            <think>Your step-by-step reasoning</think>
            <answer>X.XXX</answer>

            PROVIDE ANSWER WITH EXACTLY 3 DECIMAL PLACES, NO UNITS.
            """

    return prompt