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
        
        # Строим правильную последовательность резисторов от источника к земле
        ordered_resistors = self._order_series_resistors(circuit.resistors, pos_node, neg_node)
        
        if not ordered_resistors:
            # Не удалось построить последовательность, используем упрощенный подход
            return {pos_node: voltage, neg_node: 0.0}
        
        # Вычисляем напряжения на узлах
        result = {neg_node: 0.0}  # Заземленный узел
        
        # Проходим по резисторам в правильном порядке от заземленного узла к источнику
        current_voltage = 0.0
        current_node = neg_node
        
        # Идем в обратном порядке (от земли к источнику)
        for n1, n2, R in reversed(ordered_resistors):
            # Определяем, в каком направлении идет резистор
            if n1 == current_node:
                next_node = n2
            elif n2 == current_node:
                next_node = n1
            else:
                # Резистор не соединен с текущим узлом, пропускаем
                continue
            
            # Напряжение увеличивается по направлению к источнику
            current_voltage += current * R
            result[next_node] = current_voltage
            current_node = next_node
        
        return result
    
    def _order_series_resistors(self, resistors, start_node, end_node):
        """
        Строит правильную последовательность резисторов от start_node к end_node
        
        Args:
            resistors: Список резисторов [(n1, n2, R), ...]
            start_node: Начальный узел (источник +)
            end_node: Конечный узел (земля)
        
        Returns:
            Список резисторов в правильном порядке
        """
        if not resistors:
            return []
        
        # Создаем граф соединений
        graph = {}
        resistor_map = {}
        
        for n1, n2, R in resistors:
            if n1 not in graph:
                graph[n1] = []
            if n2 not in graph:
                graph[n2] = []
            
            graph[n1].append(n2)
            graph[n2].append(n1)
            resistor_map[(n1, n2)] = (n1, n2, R)
            resistor_map[(n2, n1)] = (n1, n2, R)
        
        # Ищем путь от start_node к end_node
        path = self._find_path(graph, start_node, end_node)
        
        if not path or len(path) < 2:
            return resistors  # Fallback к исходному порядку
        
        # Строим список резисторов по найденному пути
        ordered = []
        for i in range(len(path) - 1):
            n1, n2 = path[i], path[i + 1]
            if (n1, n2) in resistor_map:
                ordered.append(resistor_map[(n1, n2)])
        
        return ordered
    
    def _find_path(self, graph, start, end, path=None):
        """
        Ищет путь от start до end в графе (DFS)
        
        Args:
            graph: Словарь {node: [connected_nodes]}
            start: Начальный узел
            end: Конечный узел
            path: Текущий путь (для рекурсии)
        
        Returns:
            Список узлов от start до end или None
        """
        if path is None:
            path = []
        
        path = path + [start]
        
        if start == end:
            return path
        
        if start not in graph:
            return None
        
        for node in graph[start]:
            if node not in path:  # Избегаем циклов
                new_path = self._find_path(graph, node, end, path)
                if new_path:
                    return new_path
        
        return None
    
    def _solve_parallel(self, circuit: Circuit, pos_node: str, neg_node: str, voltage: float) -> Dict[str, float]:
        """Решает параллельную цепь"""
        # В параллельной цепи напряжение одинаково на всех резисторах
        return {
            pos_node: voltage,
            neg_node: 0.0
        }
    
    def get_current(self, circuit: Circuit, node_voltages: Dict[str, float], 
                   node1: str, node2: str) -> float:
        """Вычисление тока через резистор между узлами.
        
        Возвращает абсолютное значение тока (величину), игнорируя направление.
        Для последовательных и параллельных цепей этого достаточно.
        
        Args:
            circuit: Цепь
            node_voltages: Узловые потенциалы
            node1: Первый узел
            node2: Второй узел
            
        Returns:
            Абсолютное значение тока в амперах
        """
        for (n1, n2, R) in circuit.resistors:
            if (n1 == node1 and n2 == node2) or (n1 == node2 and n2 == node1):
                if R == 0:
                    continue
                # Вычисляем ток в запрошенном направлении (от node1 к node2)
                V_node1 = node_voltages.get(node1, 0.0)
                V_node2 = node_voltages.get(node2, 0.0)
                current = (V_node1 - V_node2) / R
                return abs(current)
        
        return 0.0