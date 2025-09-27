def create_circuit_prompt(metadata: dict, question_type: str, target_resistor: str) -> str:
    """
    Создает компактный промпт для задач (сокращенная версия)
    @param metadata: метаданные цепи
    @param question_type: тип вопроса
    @param target_resistor: целевой резистор
    @return: текст компактного промпта
    """
    circuit_type = metadata.get("circuit_type", "unknown")
    voltage = metadata.get("voltage_source", 10)
    resistors = metadata.get("resistors", {})

    # Компактное описание цепи
    if circuit_type == "series":
        num_resistors = len(resistors)
        resistance_values = [r for _, _, r in resistors.values()]
        circuit_desc = f"Последовательная цепь: V={voltage}V, R={resistance_values}Ω"
    elif circuit_type == "parallel":
        resistance_values = [r for _, _, r in resistors.values()]
        circuit_desc = f"Параллельная цепь: V={voltage}V, R={resistance_values}Ω"
    else:
        circuit_desc = f"Смешанная цепь: V={voltage}V, {len(resistors)} резисторов"

    # Краткая формулировка вопроса
    if question_type == "current":
        question = f"Найдите ток через {target_resistor} (в Амперах)"
    elif question_type == "voltage":
        question = f"Найдите напряжение на {target_resistor} (в Вольтах)"
    elif question_type == "power":
        question = f"Найдите мощность {target_resistor} (в Ваттах)"
    elif question_type == "total_current":
        question = "Найдите общий ток от источника (в Амперах)"
    elif question_type == "equivalent_resistance":
        question = "Найдите эквивалентное сопротивление цепи (в Омах)"
    elif question_type == "voltage_divider":
        question = f"Найдите напряжение на {target_resistor} в делителе (в Вольтах)"
    elif question_type == "current_divider":
        question = f"Найдите ток через {target_resistor} в делителе (в Амперах)"
    elif question_type == "power_total":
        question = "Найдите общую мощность всех резисторов (в Ваттах)"
    else:
        question = f"Найдите {question_type} для {target_resistor}"

    # Компактный промпт
    prompt = f"""Цепь: {circuit_desc}

Вопрос: {question}

Ответьте только числом с 3 знаками после запятой."""

    return prompt