"""Конфигурация для верификатора DC цепей."""

from dataclasses import dataclass


@dataclass
class VerifierConfig:    
    # Точность верификации
    relative_tolerance: float = 1e-3  # 0.1% относительная погрешность для LLM
    absolute_tolerance: float = 1e-6  # 1мк абсолютная погрешность
    answer_precision: int = 3       # Количество знаков после запятой
    
    # Пороги для градиентной оценки (accuracy_score)
    threshold_perfect: float = 0.01   # 1% - максимальная оценка 1.0 
    threshold_good: float = 0.05       # 5% - оценка 0.75
    threshold_ok: float = 0.10        # 10% - оценка 0.5
    threshold_fair: float = 0.20     # 20% - оценка 0.25
    
    # Математические константы
    min_divisor: float = 1e-12       # Минимальный делитель для избежания деления на ноль
    
    # Настройки извлечения ответа
    max_attempts: int = 3            # Максимальное количество попыток извлечения ответа
    fallback_tolerance: float = 0.5  # Fallback толерантность при неудачном извлечении
