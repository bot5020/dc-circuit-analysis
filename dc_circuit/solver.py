import numpy as np
from typing import Dict


class Circuit:
    """Представляет электрическую цепь
    
    ИСПРАВЛЕНО: Теперь поддерживает множественные резисторы между одними узлами
    через список вместо словаря.
    """
    def __init__(self):
        self.nodes = {}  # node_id -> voltage
        self.resistors = []  # Список кортежей: [(node1, node2, resistance), ...]
        self.voltage_sources = {}  # (node_pos, node_neg) -> voltage
        self.ground_node = None
        
    def add_resistor(self, node1: str, node2: str, resistance: float):
        """Добавляет резистор между узлами
        
        ИСПРАВЛЕНО: Теперь добавляет в список, а не перезаписывает в словаре.
        """
        self.resistors.append((node1, node2, resistance))
        
    def add_voltage_source(self, node_pos: str, node_neg: str, voltage: float):
        """Добавляет источник напряжения"""
        self.voltage_sources[(node_pos, node_neg)] = voltage
        
    def set_ground(self, node: str):
        """Устанавливает заземленный узел"""
        self.ground_node = node


class CircuitSolver:
    """Решает электрические цепи методом узловых потенциалов"""
    
    def solve(self, circuit: Circuit) -> Dict[str, float]:
        """
        Решает цепь и возвращает потенциалы узлов

        Args:
            circuit: Объект Circuit с цепью

        Returns:
            Словарь {node: voltage}
        """
        # Получение всех узлов кроме земли
        all_nodes = set()
        # ИСПРАВЛЕНО: resistors теперь список, не словарь
        for (n1, n2, _) in circuit.resistors:
            all_nodes.update([n1, n2])
        for (node_pos, node_neg) in circuit.voltage_sources.keys():
            all_nodes.update([node_pos, node_neg])
            
        if circuit.ground_node:
            all_nodes.discard(circuit.ground_node)
        
        nodes = list(all_nodes)
        n = len(nodes)
        
        if n == 0:
            return {}
            
        # Матрица проводимостей G
        G = np.zeros((n, n))
        # Вектор токов I
        I = np.zeros(n)
        
        # Заполнение матрицы проводимостей
        # ИСПРАВЛЕНО: resistors теперь список кортежей
        for (n1, n2, R) in circuit.resistors:
            if R == 0:
                continue  # Пропускаем нулевые сопротивления
                
            G_val = 1.0 / R
            
            if n1 in nodes and n2 in nodes:
                i1, i2 = nodes.index(n1), nodes.index(n2)
                G[i1, i1] += G_val
                G[i2, i2] += G_val
                G[i1, i2] -= G_val
                G[i2, i1] -= G_val
            elif n1 in nodes:  # n2 - заземленный узел
                i1 = nodes.index(n1)
                G[i1, i1] += G_val
            elif n2 in nodes:  # n1 - заземленный узел
                i2 = nodes.index(n2)
                G[i2, i2] += G_val
                
        # Добавление источников напряжения
        # Используем метод фиктивных токов для источников напряжения
        VOLTAGE_SOURCE_RESISTANCE = 1e-9  # Маленькое сопротивление для фиксации потенциала
        for (node_pos, node_neg), V in circuit.voltage_sources.items():
            if node_pos in nodes:
                i_pos = nodes.index(node_pos)
                # Фиксация потенциала источника напряжения
                G[i_pos, i_pos] += 1.0 / VOLTAGE_SOURCE_RESISTANCE
                I[i_pos] += V / VOLTAGE_SOURCE_RESISTANCE
            if node_neg in nodes:
                i_neg = nodes.index(node_neg)
                G[i_neg, i_neg] += 1.0 / VOLTAGE_SOURCE_RESISTANCE
                I[i_neg] -= V / VOLTAGE_SOURCE_RESISTANCE
                
        # Решение системы G * V = I
        try:
            voltages = np.linalg.solve(G, I)
            result = {}
            for i, node in enumerate(nodes):
                result[node] = voltages[i]
            if circuit.ground_node:
                result[circuit.ground_node] = 0.0
            return result
        except np.linalg.LinAlgError:
            return {}
    
    def get_current(self, circuit: Circuit, node_voltages: Dict[str, float], 
                   node1: str, node2: str) -> float:
        """Вычисление тока через резистор между узлами
        
        ИСПРАВЛЕНО: Теперь работает со списком резисторов.
        Если между узлами несколько резисторов, возвращает суммарный ток.
        """
        total_current = 0.0
        
        for (n1, n2, R) in circuit.resistors:
            if (n1 == node1 and n2 == node2) or (n1 == node2 and n2 == node1):
                if R == 0:
                    continue
                V1 = node_voltages.get(n1, 0.0)
                V2 = node_voltages.get(n2, 0.0)
                current = (V1 - V2) / R
                total_current += abs(current)
        
        return total_current