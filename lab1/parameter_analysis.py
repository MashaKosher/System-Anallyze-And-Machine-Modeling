"""
Анализ параметров генератора Лемера для достижения максимального периода

Теоретические основы для максимального периода:

1. Мультипликативный генератор (c = 0):
   X(n+1) = (a * X(n)) mod m
   Максимальный период: m - 1 (если m - простое число и a - примитивный корень по модулю m)

2. Смешанный генератор (c != 0):
   X(n+1) = (a * X(n) + c) mod m
   Максимальный период: m (при выполнении условий Халла)
   
   Условия Халла для максимального периода m:
   - gcd(c, m) = 1
   - a ≡ 1 (mod p) для всех простых делителей p числа m
   - a ≡ 1 (mod 4) если m ≡ 0 (mod 4)
"""

import time
from typing import List, Tuple, Dict
from lehmer_generator import LehmerGenerator


def gcd(a: int, b: int) -> int:
    """Наибольший общий делитель"""
    while b:
        a, b = b, a % b
    return a


def get_prime_factors(n: int) -> List[int]:
    """Получение списка простых делителей числа"""
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            if d not in factors:
                factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def check_hull_conditions(a: int, c: int, m: int) -> Dict[str, bool]:
    """
    Проверка условий Халла для максимального периода
    
    Returns:
        Словарь с результатами проверки каждого условия
    """
    results = {}
    
    # Условие 1: gcd(c, m) = 1
    results['gcd_c_m'] = gcd(c, m) == 1
    
    # Условие 2: a ≡ 1 (mod p) для всех простых делителей p числа m
    prime_factors = get_prime_factors(m)
    results['prime_factors_condition'] = all((a - 1) % p == 0 for p in prime_factors)
    results['prime_factors'] = prime_factors
    
    # Условие 3: a ≡ 1 (mod 4) если m ≡ 0 (mod 4)
    if m % 4 == 0:
        results['mod4_condition'] = (a - 1) % 4 == 0
    else:
        results['mod4_condition'] = True  # Условие не применимо
    
    results['all_conditions_met'] = (
        results['gcd_c_m'] and 
        results['prime_factors_condition'] and 
        results['mod4_condition']
    )
    
    return results


def find_optimal_parameters(m: int, max_attempts: int = 1000) -> List[Tuple[int, int, int]]:
    """
    Поиск оптимальных параметров (a, c) для заданного модуля m
    
    Returns:
        Список кортежей (a, c, expected_period)
    """
    optimal_params = []
    
    # Для смешанного генератора ищем параметры с максимальным периодом m
    for a in range(2, min(m, max_attempts)):
        for c in range(1, min(m, 100)):  # Ограничиваем поиск c
            if gcd(c, m) == 1:  # Первое условие Халла
                conditions = check_hull_conditions(a, c, m)
                if conditions['all_conditions_met']:
                    optimal_params.append((a, c, m))
                    if len(optimal_params) >= 10:  # Ограничиваем количество найденных параметров
                        break
        if len(optimal_params) >= 10:
            break
    
    return optimal_params


def measure_actual_period(a: int, c: int, m: int, seed: int = 1, max_iterations: int = None) -> int:
    """
    Измерение фактического периода генератора
    
    Returns:
        Фактический период или -1 если не найден
    """
    if max_iterations is None:
        max_iterations = min(m * 2, 100000)  # Разумное ограничение
    
    gen = LehmerGenerator(seed=seed, a=a, c=c, m=m)
    return gen.get_period(max_iterations)


def analyze_known_good_parameters():
    """Анализ известных хороших параметров"""
    print("=== Анализ известных хороших параметров ===\n")
    
    # Известные хорошие параметры
    good_params = [
        # Park & Miller (мультипликативный)
        {"name": "Park & Miller", "a": 16807, "c": 0, "m": 2**31 - 1, "type": "multiplicative"},
        # Numerical Recipes (смешанный)
        {"name": "Numerical Recipes", "a": 1664525, "c": 1013904223, "m": 2**32, "type": "mixed"},
        # Borland C++ (смешанный)
        {"name": "Borland C++", "a": 22695477, "c": 1, "m": 2**32, "type": "mixed"},
        # Microsoft C (смешанный)
        {"name": "Microsoft C", "a": 214013, "c": 2531011, "m": 2**32, "type": "mixed"},
    ]
    
    for params in good_params:
        print(f"Параметры: {params['name']} ({params['type']})")
        print(f"   a = {params['a']}, c = {params['c']}, m = {params['m']}")
        
        # Проверка условий для смешанного генератора
        if params['c'] != 0:
            conditions = check_hull_conditions(params['a'], params['c'], params['m'])
            print(f"   Условия Халла:")
            print(f"     gcd(c, m) = 1: {conditions['gcd_c_m']}")
            print(f"     Условие для простых делителей: {conditions['prime_factors_condition']}")
            print(f"     Условие mod 4: {conditions['mod4_condition']}")
            print(f"     Все условия выполнены: {conditions['all_conditions_met']}")
            expected_period = params['m'] if conditions['all_conditions_met'] else "неопределен"
        else:
            expected_period = params['m'] - 1  # Для мультипликативного генератора
        
        print(f"   Ожидаемый период: {expected_period}")
        
        # Измерение фактического периода (ограниченно)
        start_time = time.time()
        actual_period = measure_actual_period(
            params['a'], params['c'], params['m'], 
            max_iterations=10000
        )
        end_time = time.time()
        
        if actual_period != -1:
            print(f"   Фактический период (первые 10000): {actual_period}")
        else:
            print(f"   Фактический период: > 10000 (не найден повтор)")
        
        print(f"   Время измерения: {end_time - start_time:.4f} сек")
        print()


def analyze_small_modulus_examples():
    """Анализ примеров с малым модулем для демонстрации"""
    print("=== Анализ примеров с малым модулем ===\n")
    
    # Пример 1: m = 16 (степень 2)
    print("1. Модуль m = 16 (2^4):")
    m = 16
    optimal = find_optimal_parameters(m, max_attempts=16)
    
    if optimal:
        print(f"   Найдено {len(optimal)} оптимальных наборов параметров:")
        for i, (a, c, expected_period) in enumerate(optimal[:5]):  # Показываем первые 5
            actual_period = measure_actual_period(a, c, m)
            print(f"     #{i+1}: a={a}, c={c}, ожидаемый период={expected_period}, "
                  f"фактический период={actual_period}")
    else:
        print("   Оптимальных параметров не найдено")
    
    print()
    
    # Пример 2: m = 31 (простое число)
    print("2. Модуль m = 31 (простое число):")
    m = 31
    optimal = find_optimal_parameters(m, max_attempts=31)
    
    if optimal:
        print(f"   Найдено {len(optimal)} оптимальных наборов параметров:")
        for i, (a, c, expected_period) in enumerate(optimal[:5]):
            actual_period = measure_actual_period(a, c, m)
            print(f"     #{i+1}: a={a}, c={c}, ожидаемый период={expected_period}, "
                  f"фактический период={actual_period}")
    else:
        print("   Оптимальных параметров не найдено")
    
    print()


def demonstrate_period_behavior():
    """Демонстрация поведения периода для разных параметров"""
    print("=== Демонстрация поведения периода ===\n")
    
    # Плохие параметры (не выполняющие условия Халла)
    print("1. Плохие параметры (a=3, c=5, m=16):")
    gen_bad = LehmerGenerator(seed=1, a=3, c=5, m=16)
    sequence_bad = gen_bad.generate_sequence(20)
    print(f"   Последовательность: {sequence_bad}")
    
    gen_bad.reset()
    period_bad = gen_bad.get_period()
    print(f"   Период: {period_bad}")
    
    conditions_bad = check_hull_conditions(3, 5, 16)
    print(f"   Условия Халла выполнены: {conditions_bad['all_conditions_met']}")
    print()
    
    # Хорошие параметры
    print("2. Хорошие параметры (a=5, c=1, m=16):")
    gen_good = LehmerGenerator(seed=1, a=5, c=1, m=16)
    sequence_good = gen_good.generate_sequence(20)
    print(f"   Последовательность: {sequence_good}")
    
    gen_good.reset()
    period_good = gen_good.get_period()
    print(f"   Период: {period_good}")
    
    conditions_good = check_hull_conditions(5, 1, 16)
    print(f"   Условия Халла выполнены: {conditions_good['all_conditions_met']}")
    print()


def performance_comparison():
    """Сравнение производительности разных генераторов"""
    print("=== Сравнение производительности ===\n")
    
    generators = [
        {"name": "Park & Miller", "a": 16807, "c": 0, "m": 2**31 - 1},
        {"name": "Numerical Recipes", "a": 1664525, "c": 1013904223, "m": 2**32},
        {"name": "Простой пример", "a": 5, "c": 1, "m": 16},
    ]
    
    num_generations = 100000
    
    for gen_params in generators:
        gen = LehmerGenerator(
            seed=1, 
            a=gen_params['a'], 
            c=gen_params['c'], 
            m=gen_params['m']
        )
        
        # Измерение времени генерации целых чисел
        start_time = time.time()
        for _ in range(num_generations):
            gen.next_int()
        int_time = time.time() - start_time
        
        # Измерение времени генерации чисел с плавающей точкой
        gen.reset()
        start_time = time.time()
        for _ in range(num_generations):
            gen.next_float()
        float_time = time.time() - start_time
        
        print(f"{gen_params['name']}:")
        print(f"   {num_generations} целых чисел: {int_time:.4f} сек "
              f"({num_generations/int_time:.0f} чисел/сек)")
        print(f"   {num_generations} float чисел: {float_time:.4f} сек "
              f"({num_generations/float_time:.0f} чисел/сек)")
        print()


def main():
    """Основная функция для запуска всех анализов"""
    print("АНАЛИЗ ПАРАМЕТРОВ ГЕНЕРАТОРА ЛЕМЕРА")
    print("=" * 50)
    print()
    
    analyze_known_good_parameters()
    analyze_small_modulus_examples()
    demonstrate_period_behavior()
    performance_comparison()
    
    print("=== Рекомендации ===")
    print("1. Для мультипликативного генератора используйте параметры Park & Miller:")
    print("   a = 16807, c = 0, m = 2^31 - 1")
    print("   Период: 2^31 - 2 = 2,147,483,646")
    print()
    print("2. Для смешанного генератора используйте параметры Numerical Recipes:")
    print("   a = 1664525, c = 1013904223, m = 2^32")
    print("   Период: 2^32 = 4,294,967,296")
    print()
    print("3. Всегда проверяйте условия Халла для смешанного генератора")
    print("4. Используйте достаточно большой модуль для практических применений")


if __name__ == "__main__":
    main()
