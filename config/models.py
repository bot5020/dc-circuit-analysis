"""
Централизованная конфигурация моделей
"""

# Основная модель для всех компонентов (небольшая и быстрая)
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Конфигурации для разных режимов
MODEL_CONFIGS = {
    "production": {
        "model_name": DEFAULT_MODEL,
        "load_in_4bit": True,
        "max_seq_length": 2048,
        "dtype": None
    },
    "fast": {
        "model_name": DEFAULT_MODEL,
        "load_in_4bit": True,
        "max_seq_length": 2048,
        "dtype": None
    },
    "colab": {
        "model_name": DEFAULT_MODEL,
        "load_in_4bit": True,
        "max_seq_length": 2048,
        "dtype": None
    },
    "debug": {
        "model_name": DEFAULT_MODEL,
        "load_in_4bit": True,
        "max_seq_length": 1024,
        "dtype": None
    }
}

def get_model_name(config_type: str = "production") -> str:
    """Получить имя модели для конфигурации"""
    return MODEL_CONFIGS.get(config_type, MODEL_CONFIGS["production"])["model_name"]

def get_model_config(config_type: str = "production") -> dict:
    """Получить полную конфигурацию модели"""
    return MODEL_CONFIGS.get(config_type, MODEL_CONFIGS["production"])
