"""Модуль общих утилит для всего проекта.

Содержит унифицированные функции для извлечения ответов и создания промптов,
используемые во всех модулях проекта для избежания дублирования кода.
"""

import re


def extract_answer(test_solution: str) -> str:
    """Извлекает численный ответ из решения агента.
    
    Функция пробует несколько форматов извлечения в следующем порядке:
    1. Число внутри <answer>NUMBER</answer> тегов
    2. Число после знака = в конце строки
    3. Число после "Answer:" или "Ответ:"
    4. Последнее число в тексте (fallback)
    
    Args:
        test_solution: Полный текст решения агента (может содержать рассуждения, теги)
    
    Returns:
        Извлеченное численное значение как строка, или пустая строка если не найдено
        
    Example:
        >>> extract_answer("<think>Step 1...</think><answer>1.234</answer>")
        '1.234'
        >>> extract_answer("Current = 5V / 10Ω = 0.500 A")
        '0.500'
    """
    # Чистим текст
    text = test_solution.strip()

    # 1. Ищем число в <answer> тегах (основной формат)
    tag_match = re.search(r'<answer>\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*</answer>', text)
    if tag_match:
        return tag_match.group(1)

    # 2. Ищем "= число" в конце (альтернативный формат)
    equals_match = re.search(r'=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(?:A|V|W|Ω|Ohms?)?\s*$', text, re.MULTILINE)
    if equals_match:
        return equals_match.group(1)

    # 3. Ищем "Answer: число" или "Ответ: число"
    answer_match = re.search(r'(?:Answer|Ответ):\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1)

    # 4. Ищем первое число в тексте
    numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text)
    if numbers:
        return numbers[-1]  # Берем последнее число (обычно финальный ответ)

    return ""


def get_system_prompt() -> str:
    """Возвращает унифицированный системный промпт для модели.
    
    Промпт содержит четкие инструкции для анализа электрических цепей
    с использованием структурированного формата ответа и few-shot примеров.
    
    Returns:
        Текст системного промпта на английском языке
        
    Note:
        Используется во всех модулях (rl_trainer, evaluate, final_test) для
        обеспечения единообразия промптов.
    """
    return (
        "You are an expert electrical engineer specializing in DC circuit analysis. "
        "Your task is to solve electrical circuit problems using fundamental physics laws.\n\n"
        
        "FUNDAMENTAL LAWS:\n"
        "• Ohm's Law: V = I × R, I = V/R, R = V/I\n"
        "• Kirchhoff's Current Law (KCL): ΣI_in = ΣI_out\n"
        "• Kirchhoff's Voltage Law (KVL): ΣV = 0\n"
        "• Series: R_total = R₁ + R₂ + ..., I_total = I₁ = I₂\n"
        "• Parallel: 1/R_total = 1/R₁ + 1/R₂ + ..., V_total = V₁ = V₂\n"
        "• Power: P = I²R = V²/R = VI\n\n"
        
        "APPROACH:\n"
        "1. Identify circuit topology (series, parallel, mixed)\n"
        "2. Apply appropriate laws (Ohm, KCL, KVL)\n"
        "3. Calculate equivalent resistance if needed\n"
        "4. Solve for the requested quantity step by step\n"
        "5. Verify your answer makes physical sense\n\n"
        
        "RESPONSE FORMAT:\n"
        "Always respond in this exact format:\n"
        "<think>\n"
        "Step-by-step reasoning and calculations\n"
        "</think>\n"
        "<answer>X.XXX</answer>\n\n"
        
        "EXAMPLE 1 - Series Circuit:\n"
        "User: Find current through R1 in series circuit with V=12V, R1=4Ω, R2=8Ω\n"
        "Assistant: <think>\n"
        "Step 1: This is a series circuit\n"
        "Step 2: Total resistance R_total = R1 + R2 = 4Ω + 8Ω = 12Ω\n"
        "Step 3: Current I = V/R_total = 12V/12Ω = 1.000A\n"
        "Step 4: In series, current through R1 equals total current = 1.000A\n"
        "</think>\n"
        "<answer>1.000</answer>\n\n"
        
        "EXAMPLE 2 - Parallel Circuit:\n"
        "User: Find voltage across R1 in parallel circuit with V=9V, R1=3Ω, R2=6Ω\n"
        "Assistant: <think>\n"
        "Step 1: This is a parallel circuit\n"
        "Step 2: In parallel, voltage across R1 equals source voltage = 9V\n"
        "Step 3: V_R1 = V_source = 9.000V\n"
        "</think>\n"
        "<answer>9.000</answer>\n\n"
        
        "IMPORTANT RULES:\n"
        "- Show ALL calculations in <think> tags\n"
        "- Provide final answer in <answer> tags\n"
        "- Use exactly 3 decimal places (e.g., 1.234)\n"
        "- No units in the final answer\n"
        "- Be precise and methodical\n"
        "- Check your work for reasonableness"
    )


