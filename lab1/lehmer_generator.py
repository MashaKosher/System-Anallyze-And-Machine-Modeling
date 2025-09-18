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
    
    def __init__(self, seed: int = 1, a: int = None, c: int = None, m: int = None):
        """
        Инициализация генератора
        
        Args:
            seed: начальное значение (X0)
            a: множитель
            c: приращение
            m: модуль
        """
        # Параметры для максимального периода (m = 2^31 - 1)
        if m is None:
            self.m = 2**31 - 1  # Простое число Мерсенна
        else:
            self.m = m
            
        if a is None:
            self.a = 16807  # Рекомендуемый множитель для m = 2^31 - 1
        else:
            self.a = a
            
        if c is None:
            self.c = 0  # Мультипликативный генератор (c = 0)
        else:
            self.c = c
            
        self.seed = seed
        self.current = seed
        self.initial_seed = seed
        
        # Проверка параметров
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Проверка корректности параметров для максимального периода"""
        if self.c != 0:
            # Для смешанного генератора (c != 0)
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
        
        original_current = self.current
        self.reset()
        
        seen_values = set()
        sequence = []
        
        for i in range(max_iterations):
            value = self.next_int()
            if value in seen_values:
                # Найдено повторение, ищем начало цикла
                try:
                    cycle_start = sequence.index(value)
                    period = i - cycle_start
                    self.current = original_current
                    return period
                except ValueError:
                    pass
            
            seen_values.add(value)
            sequence.append(value)
        
        self.current = original_current
        return -1  # Период не найден в пределах max_iterations
    
    def __iter__(self) -> Iterator[int]:
        """Итератор для генерации бесконечной последовательности"""
        while True:
            yield self.next_int()
    
    def __str__(self) -> str:
        return f"LehmerGenerator(a={self.a}, c={self.c}, m={self.m}, seed={self.seed})"


class OptimalLehmerGenerator(LehmerGenerator):
    """
    Генератор с оптимальными параметрами для максимального периода
    """
    
    def __init__(self, seed: int = 1, generator_type: str = "multiplicative"):
        """
        Args:
            seed: начальное значение
            generator_type: тип генератора ("multiplicative" или "mixed")
        """
        if generator_type == "multiplicative":
            # Мультипликативный генератор с максимальным периодом
            # Параметры Park and Miller (1988)
            super().__init__(
                seed=seed,
                a=16807,        # 7^5
                c=0,            # Мультипликативный
                m=2**31 - 1     # Простое число Мерсенна 2147483647
            )
        elif generator_type == "mixed":
            # Смешанный генератор с максимальным периодом
            # Параметры Numerical Recipes
            super().__init__(
                seed=seed,
                a=1664525,      # Множитель
                c=1013904223,   # Приращение
                m=2**32         # Модуль степени 2
            )
        else:
            raise ValueError("generator_type должен быть 'multiplicative' или 'mixed'")
        
        self.generator_type = generator_type


def test_generators():
    """Тестирование различных генераторов"""
    print("=== Тестирование генераторов псевдослучайных чисел ===\n")
    
    # Тест 1: Мультипликативный генератор
    print("1. Мультипликативный генератор (Park & Miller):")
    gen1 = OptimalLehmerGenerator(seed=1, generator_type="multiplicative")
    print(f"   Параметры: {gen1}")
    
    # Генерация первых 10 чисел
    sequence1 = gen1.generate_sequence(10)
    print(f"   Первые 10 чисел: {sequence1}")
    
    # Генерация чисел с плавающей точкой
    gen1.reset()
    float_sequence1 = gen1.generate_float_sequence(5)
    print(f"   Первые 5 чисел [0,1): {[f'{x:.6f}' for x in float_sequence1]}")
    
    # Тест периода (ограниченный)
    gen1.reset()
    period1 = gen1.get_period(max_iterations=10000)
    print(f"   Период (первые 10000 итераций): {period1 if period1 != -1 else 'не найден'}")
    
    print()
    
    # Тест 2: Смешанный генератор
    print("2. Смешанный генератор (Numerical Recipes):")
    gen2 = OptimalLehmerGenerator(seed=1, generator_type="mixed")
    print(f"   Параметры: {gen2}")
    
    # Генерация первых 10 чисел
    sequence2 = gen2.generate_sequence(10)
    print(f"   Первые 10 чисел: {sequence2}")
    
    # Генерация чисел с плавающей точкой
    gen2.reset()
    float_sequence2 = gen2.generate_float_sequence(5)
    print(f"   Первые 5 чисел [0,1): {[f'{x:.6f}' for x in float_sequence2]}")
    
    print()
    
    # Тест 3: Сравнение с простыми параметрами
    print("3. Генератор с простыми параметрами (для сравнения):")
    gen3 = LehmerGenerator(seed=1, a=5, c=7, m=16)
    print(f"   Параметры: {gen3}")
    
    sequence3 = gen3.generate_sequence(20)
    print(f"   Первые 20 чисел: {sequence3}")
    
    gen3.reset()
    period3 = gen3.get_period()
    print(f"   Период: {period3}")
    
    print()
    
    # Демонстрация использования как итератора
    print("4. Использование как итератор:")
    gen4 = OptimalLehmerGenerator(seed=12345)
    iterator_values = []
    gen_iter = iter(gen4)
    for i in range(5):
        iterator_values.append(next(gen_iter))
    print(f"   Первые 5 значений через итератор: {iterator_values}")
