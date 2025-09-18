"""
Лабораторная работа 1: Генератор последовательности равномерно распределенных 
случайных чисел на основе смешанного алгоритма Лемера

Смешанный алгоритм Лемера (Linear Congruential Generator):
X(n+1) = (a * X(n) + c) mod m

где:
- a - множитель (multiplier)
- c - приращение (increment)
- m - модуль (modulus)
- X(0) - начальное значение (seed)

Для максимального периода необходимо выполнение условий:
1. gcd(c, m) = 1
2. a - 1 кратно всем простым делителям m
3. Если m кратно 4, то a - 1 должно быть кратно 4
"""

import math
from typing import Iterator, List


class LehmerGenerator:
    """
    Генератор псевдослучайных чисел на основе смешанного алгоритма Лемера
    """

    # Параметры для максимального периода (m = 2^31 - 1)
    DEFAULT_M: int = 2**31 - 1  # Простое число Мерсенна
    DEFAULT_A: int = 16807  # Рекомендуемый множитель для m = 2^31 - 1
    DEFAULT_C: int = 0  # Мультипликативный метод по умолчанию
    
    def __init__(self, seed: int = 1, a: int = None, c: int = None, m: int = None):
        """
        Инициализация генератора
        
        Args:
            seed: начальное значение (X0)
            a: множитель
            c: приращение
            m: модуль
        """
        self.m = self._fill_param(m, self.DEFAULT_M)
        self.a = self._fill_param(a, self.DEFAULT_A)
        self.c = self._fill_param(c, self.DEFAULT_C)
            
        self.seed = seed
        self.current = seed
        self.initial_seed = seed
        
        self._validate_parameters()

    @staticmethod
    def _fill_param(num: int | None, default_value: int) -> int:
        """Заполняет параметр значением по умолчанию, если он None"""
        if num is None:
            return default_value
        return num
    
    def _validate_parameters(self):
        """Проверка корректности параметров для максимального периода"""
        if self.c != 0:
            # Для смешанного генератора (c != 0) иначе мультипликативный (c == 0)
            gcd_cm = math.gcd(self.c, self.m)
            if gcd_cm != 1:
                print(f"Предупреждение: gcd(c, m) = {gcd_cm} != 1")
        
        if self.a <= 0 or self.a >= self.m:
            print(f"Предупреждение: множитель a должен быть в диапазоне (0, m)")
        
        if self.seed <= 0 or self.seed >= self.m:
            print(f"Предупреждение: начальное значение должно быть в диапазоне (0, m)")
    
    def next_int(self) -> int:
        """
        Генерация следующего целого числа
        
        Returns:
            Следующее псевдослучайное целое число
        """
        self.current = (self.a * self.current + self.c) % self.m
        return self.current
    
    def next_float(self) -> float:
        """
        Генерация следующего числа с плавающей точкой в диапазоне [0, 1)
        
        Returns:
            Следующее псевдослучайное число с плавающей точкой
        """
        return self.next_int() / self.m
    
    def generate_sequence(self, count: int) -> List[int]:
        """
        Генерация последовательности целых чисел
        
        Args:
            count: количество чисел для генерации
            
        Returns:
            Список псевдослучайных целых чисел
        """
        return [self.next_int() for _ in range(count)]
    
    def generate_float_sequence(self, count: int) -> List[float]:
        """
        Генерация последовательности чисел с плавающей точкой
        
        Args:
            count: количество чисел для генерации
            
        Returns:
            Список псевдослучайных чисел с плавающей точкой в диапазоне [0, 1)
        """
        return [self.next_float() for _ in range(count)]
    
    def reset(self):
        """Сброс генератора к начальному состоянию"""
        self.current = self.initial_seed
    
    def set_seed(self, seed: int):
        """Установка нового начального значения"""
        self.seed = seed
        self.initial_seed = seed
        self.current = seed
    
    def get_period(self, max_iterations: int = None) -> int:
        """
        Определение периода генератора
        
        Args:
            max_iterations: максимальное количество итераций для поиска периода
            
        Returns:
            Период генератора (или -1 если не найден в пределах max_iterations)
        """
        if max_iterations is None:
            max_iterations = min(self.m, 10**6)  # Ограничиваем поиск
        
        self.reset()
        
        seen_values = set()
        sequence = []
        
        for i in range(max_iterations):
            value = self.next_int()
            if value in seen_values:
                # Найдено повторение, ищем первое вхождение этого числа
                try:
                    exist_number_index = sequence.index(value)
                    period = i - exist_number_index
                    self.reset() 
                    return period
                except ValueError:
                    pass
            
            seen_values.add(value)
            sequence.append(value)
        
        self.reset()
        return -1  # Период не найден в пределах max_iterations
    
    def __iter__(self) -> Iterator[int]:
        """Итератор для генерации бесконечной последовательности"""
        while True:
            yield self.next_int()
