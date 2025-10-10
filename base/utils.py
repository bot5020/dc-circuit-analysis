"""Модуль общих утилит для всего проекта.

Содержит унифицированные функции для извлечения ответов и создания промптов,
используемые во всех модулях проекта для избежания дублирования кода.
"""

import re
from typing import Optional


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
    
    Промпт на английском языке содержит инструкции по формату ответа
    с использованием <think> и <answer> тегов для структурированного вывода.
    
    Returns:
        Текст системного промпта на английском языке
        
    Note:
        Используется во всех модулях (rl_trainer, evaluate, final_test) для
        обеспечения единообразия промптов.
    """
    return (
        "You are an expert circuit analysis engineer. "
        "Solve electrical circuit problems using physics laws.\n\n"
        "Respond in the following format:\n"
        "<think>Reason step by step briefly.</think>\n"
        "<answer>Return ONLY the final number with exactly 3 decimal places "
        "(e.g., 1.234), no units.</answer>"
    )
