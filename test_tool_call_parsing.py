#!/usr/bin/env python3
"""Тестирование парсинга tool_call ответов."""

import re


def parse_tool_call_response(completion_str):
    """Парсит ответ из tool_call формата"""
    try:
        # Ищем <answer>X.XXX</answer> внутри tool_call
        answer_match = re.search(r'<answer>([0-9.-]+)</answer>', completion_str)
        if answer_match:
            return answer_match.group(1)
        else:
            return None
    except Exception as e:
        print(f"Ошибка парсинга: {e}")
        return None


def test_parsing():
    """Тестирует парсинг на примерах"""
    
    # Примеры tool_call ответов из логов
    test_cases = [
        {
            "name": "Правильный tool_call с answer",
            "completion": """<tool_call>
Step 1: This is a series circuit
Step 2: Total resistance R_total = R1 + R2 = 4Ω + 8Ω = 12Ω
Step 3: Current I = V/R_total = 12V/12Ω = 1.000A
Step 4: In series, current through R1 equals total current = 1.000A
</tool_call>
<answer>1.000</answer>""",
            "expected": "1.000"
        },
        {
            "name": "Tool_call без answer",
            "completion": """<tool_call>
Step 1: This is a series circuit
Step 2: Total resistance R_total = R1 + R2 = 4Ω + 8Ω = 12Ω
Step 3: Current I = V/R_total = 12V/12Ω = 1.000A
</tool_call>""",
            "expected": None
        },
        {
            "name": "Правильный формат think/answer",
            "completion": """<think>
Step 1: This is a series circuit
Step 2: Total resistance R_total = R1 + R2 = 4Ω + 8Ω = 12Ω
Step 3: Current I = V/R_total = 12V/12Ω = 1.000A
</think>
<answer>1.000</answer>""",
            "expected": "1.000"
        }
    ]
    
    print("🧪 Тестирование парсинга tool_call ответов\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Тест {i}: {test_case['name']}")
        print(f"Вход: {test_case['completion'][:100]}...")
        
        result = parse_tool_call_response(test_case['completion'])
        expected = test_case['expected']
        
        if result == expected:
            print(f"✅ Успех: {result}")
        else:
            print(f"❌ Ошибка: ожидалось {expected}, получено {result}")
        
        print("-" * 50)


if __name__ == "__main__":
    test_parsing()
