"""
Общие утилиты для обучения и инференса моделей
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from typing import List



class LMStudioClient:
    """
    Клиент для работы с LM Studio Server API

    Использует OpenAI-совместимый API: POST /v1/chat/completions
    """

    def __init__(self, api_url: str = "http://localhost:1234"):
        """
        Инициализация клиента LM Studio

        Args:
            api_url: URL LM Studio сервера (по умолчанию localhost:1234)
        """
        self.api_url = api_url.rstrip('/')
        self.session = requests.Session()

    def generate(self,
                 prompt: str,
                 max_tokens: int = 128,
                 temperature: float = 0.0,
                 stop_sequences: List[str] = None,
                 model: str = "default") -> str:
        """
        Генерирует ответ от LM Studio через API

        Args:
            prompt: Входной промпт
            max_tokens: Максимальное количество токенов
            temperature: Температура сэмплирования (0.0 = детерминированно)
            stop_sequences: Последовательности для остановки генерации
            model: Название модели для использования

        Returns:
            Сгенерированный текст
        """
        # Конвертируем промпт в формат чата
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 4000,
            "temperature": temperature,
            "stop": stop_sequences or [],
            "stream": False
        }

        try:
            response = self.session.post(
                f"{self.api_url}/v1/chat/completions",
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
            return result["choices"][0]["message"]["content"].strip()

        except requests.exceptions.RequestException as e:
            print(f"❌ Ошибка LM Studio API: {e}")
            return ""
        except (KeyError, IndexError) as e:
            print(f"❌ Ошибка обработки ответа LM Studio: {e}")
            return ""
        except Exception as e:
            print(f"❌ Неожиданная ошибка: {e}")
            return ""

    def health_check(self) -> bool:
        """
        Проверяет доступность LM Studio сервера

        Returns:
            True если сервер доступен
        """
        try:
            response = self.session.get(f"{self.api_url}/v1/models", timeout=5)
            return response.status_code == 200
        except:
            return False


