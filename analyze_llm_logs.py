#!/usr/bin/env python3
"""Анализ логов LLM для обучения GRPO."""

import json
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def load_logs(log_file):
    """Загружает логи из файла"""
    logs = []
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                logs.append(json.loads(line))
    return logs


def analyze_rewards(logs):
    """Анализирует распределение rewards"""
    rewards = [log['reward'] for log in logs if log['reward'] is not None]
    
    print(f"📊 Анализ rewards:")
    print(f"  Всего взаимодействий: {len(rewards)}")
    print(f"  Средний reward: {np.mean(rewards):.3f}")
    print(f"  Медианный reward: {np.median(rewards):.3f}")
    print(f"  Мин reward: {np.min(rewards):.3f}")
    print(f"  Макс reward: {np.max(rewards):.3f}")
    print(f"  Стандартное отклонение: {np.std(rewards):.3f}")
    
    return rewards


def analyze_prompts(logs):
    """Анализирует промпты"""
    print(f"\n📝 Анализ промптов:")
    print(f"  Уникальных промптов: {len(set(log['prompt'] for log in logs))}")
    
    # Анализ длины промптов
    prompt_lengths = [len(log['prompt']) for log in logs]
    print(f"  Средняя длина промпта: {np.mean(prompt_lengths):.1f} символов")
    print(f"  Мин длина: {np.min(prompt_lengths)}")
    print(f"  Макс длина: {np.max(prompt_lengths)}")


def analyze_completions(logs):
    """Анализирует ответы LLM"""
    print(f"\n🤖 Анализ ответов LLM:")
    
    # Анализ длины ответов
    completion_lengths = [len(log['completion']) for log in logs]
    print(f"  Средняя длина ответа: {np.mean(completion_lengths):.1f} символов")
    print(f"  Мин длина: {np.min(completion_lengths)}")
    print(f"  Макс длина: {np.max(completion_lengths)}")
    
    # Анализ качества по rewards
    high_reward_logs = [log for log in logs if log['reward'] and log['reward'] > 1.5]
    print(f"  Высококачественных ответов (reward > 1.5): {len(high_reward_logs)}")


def plot_reward_distribution(rewards, output_file="reward_distribution.png"):
    """Строит график распределения rewards"""
    plt.figure(figsize=(10, 6))
    plt.hist(rewards, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Reward')
    plt.ylabel('Частота')
    plt.title('Распределение Rewards')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📈 График сохранен в {output_file}")


def export_sample_logs(logs, output_file="sample_logs.json", n_samples=10):
    """Экспортирует примеры логов"""
    sample_logs = logs[:n_samples]
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sample_logs, f, ensure_ascii=False, indent=2)
    print(f"📄 Примеры логов сохранены в {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Анализ логов LLM')
    parser.add_argument('log_file', help='Путь к файлу логов')
    parser.add_argument('--output', '-o', default='analysis_output', help='Префикс для выходных файлов')
    parser.add_argument('--samples', '-s', type=int, default=10, help='Количество примеров для экспорта')
    
    args = parser.parse_args()
    
    print(f"🔍 Анализ логов из {args.log_file}")
    
    # Загружаем логи
    logs = load_logs(args.log_file)
    print(f"📁 Загружено {len(logs)} записей")
    
    # Анализируем
    rewards = analyze_rewards(logs)
    analyze_prompts(logs)
    analyze_completions(logs)
    
    # Строим графики
    if rewards:
        plot_reward_distribution(rewards, f"{args.output}_reward_distribution.png")
    
    # Экспортируем примеры
    export_sample_logs(logs, f"{args.output}_samples.json", args.samples)
    
    print(f"\n✅ Анализ завершен! Результаты сохранены с префиксом '{args.output}'")


if __name__ == "__main__":
    main()
