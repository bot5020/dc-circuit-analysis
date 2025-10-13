from typing import Dict


class Circuit:
    """Упрощенное представление электрической цепи"""
    def __init__(self):
        self.resistors = []  # Список кортежей: [(node1, node2, resistance), ...]
        self.voltage_sources = {}  # (node_pos, node_neg) -> voltage
        self.ground_node = None
        
    def add_resistor(self, node1: str, node2: str, resistance: float):
        """Добавляет резистор между узлами"""
        self.resistors.append((node1, node2, resistance))
        
    def add_voltage_source(self, node_pos: str, node_neg: str, voltage: float):
        """Добавляет источник напряжения"""
        self.voltage_sources[(node_pos, node_neg)] = voltage
        
    def set_ground(self, node: str):
        """Устанавливает заземленный узел"""
        self.ground_node = node


class CircuitSolver:
    """Упрощенный решатель электрических цепей"""
    
    def solve(self, circuit: Circuit) -> Dict[str, float]:
        """
        Решает простые цепи (последовательные и параллельные) без матриц

        Args:
            circuit: Объект Circuit с цепью

        Returns:
            Словарь {node: voltage}
        """
        # Получаем источник напряжения
        if not circuit.voltage_sources:
            return {}
        
        voltage_source = list(circuit.voltage_sources.items())[0]
        (pos_node, neg_node), voltage = voltage_source
        
        # Определяем тип цепи
        circuit_type = self._detect_circuit_type(circuit)
        
        if circuit_type == "series":
            return self._solve_series(circuit, pos_node, neg_node, voltage)
        elif circuit_type == "parallel":
            return self._solve_parallel(circuit, pos_node, neg_node, voltage)
        else:
            return {}
    
    def _detect_circuit_type(self, circuit: Circuit) -> str:
        """Определяет тип цепи: series или parallel"""
        if not circuit.resistors:
            return "unknown"
        
        # Если все резисторы между одними узлами - параллельная цепь
        first_resistor = circuit.resistors[0]
        n1, n2 = first_resistor[0], first_resistor[1]
        
        all_same_nodes = all(
            (r[0] == n1 and r[1] == n2) or (r[0] == n2 and r[1] == n1)
            for r in circuit.resistors
        )
        
        return "parallel" if all_same_nodes else "series"
    
    def _solve_series(self, circuit: Circuit, pos_node: str, neg_node: str, voltage: float) -> Dict[str, float]:
        """Решает последовательную цепь"""
        # Находим общее сопротивление
        total_resistance = sum(r[2] for r in circuit.resistors)
        
        if total_resistance == 0:
            return {}
        
        # Ток в последовательной цепи
        current = voltage / total_resistance
        
        # Вычисляем напряжения на узлах
        result = {neg_node: 0.0}  # Заземленный узел
        
        # Проходим по резисторам и вычисляем напряжения
        remaining_voltage = voltage
        for r in circuit.resistors:
            n1, n2, R = r
            voltage_drop = current * R
            remaining_voltage -= voltage_drop
            
            if n1 not in result:
                result[n1] = remaining_voltage + voltage_drop
            if n2 not in result:
                result[n2] = remaining_voltage
        
        return result
    
    def _solve_parallel(self, circuit: Circuit, pos_node: str, neg_node: str, voltage: float) -> Dict[str, float]:
        """Решает параллельную цепь"""
        # В параллельной цепи напряжение одинаково на всех резисторах
        return {
            pos_node: voltage,
            neg_node: 0.0
        }
    
    def get_current(self, circuit: Circuit, node_voltages: Dict[str, float], 
                   node1: str, node2: str) -> float:
        """Вычисление тока через резистор между узлами"""
        for (n1, n2, R) in circuit.resistors:
            if (n1 == node1 and n2 == node2) or (n1 == node2 and n2 == node1):
                if R == 0:
                    continue
                V1 = node_voltages.get(n1, 0.0)
                V2 = node_voltages.get(n2, 0.0)
                current = (V1 - V2) / R
                return abs(current)
        
        return 0.0