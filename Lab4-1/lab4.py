import heapq
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from mpl_toolkits.mplot3d import Axes3D

@dataclass
class SimulationResult:
    """Класс для хранения результатов одного прогона"""
    average_wait_time_type1: float
    average_wait_time_type2: float
    average_repair_queue_length: float
    workstation_utilization: List[float]
    painting_utilization: float
    interruptions_count: int
    simulation_time: float
    cars_processed: int
    time_series_data: Dict[str, List[float]]

class AutoRepairStationModel:
    """Имитационная модель участка авторемонтной станции"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.reset()
        
    def reset(self):
        """Сброс состояния модели"""
        self.current_time = 0.0
        self.car_counter = 0
        self.repair_queue = []
        self.painting_buffer = []
        self.workstations = [None] * self.config['num_workstations']
        self.painting_booth = None
        self.failed_workstations = set()
        
        # Статистика
        self.stats = {
            'cars_processed': 0, 'cars_type1': 0, 'cars_type2': 0,
            'total_wait_time_type1': 0.0, 'total_wait_time_type2': 0.0,
            'interruptions': 0, 
            'workstation_busy_time': [0.0] * self.config['num_workstations'],
            'painting_busy_time': 0.0,
            'wait_times_type1': [], 'wait_times_type2': [],
            'repair_queue_lengths': [], 'time_points': []
        }
        
        self.event_queue = []
        
    def exponential_random(self, mean: float) -> float:
        return -mean * math.log(1 - random.random() + 1e-10)
    
    def schedule_event(self, event_type: str, time: float, data: Any = None):
        heapq.heappush(self.event_queue, (time, event_type, data))
    
    def record_state(self):
        """Фиксирует текущее состояние очереди"""
        self.stats['repair_queue_lengths'].append(len(self.repair_queue))
        self.stats['time_points'].append(self.current_time)
    
    def run(self) -> SimulationResult:
        """Запуск одного прогона модели"""
        max_time = self.config.get('max_time', 2000)
        
        # Планируем начальное прибытие
        self.schedule_event('arrival', self.exponential_random(self.config['arrival_rates'][0]))
        self.record_state()
        
        while self.event_queue and self.current_time < max_time:
            time, event_type, data = heapq.heappop(self.event_queue)
            self.current_time = time
            
            if event_type == 'arrival':
                self.process_arrival()
                # Планируем следующее прибытие
                if self.stats['cars_processed'] < self.config.get('max_cars', 1000):
                    next_arrival = self.current_time + self.exponential_random(self.config['arrival_rates'][0])
                    self.schedule_event('arrival', next_arrival)
                    
            elif event_type == 'repair_end':
                self.process_repair_end(data)
                
            elif event_type == 'painting_end':
                self.process_painting_end()
            
            self.record_state()
        
        return self.collect_results()
    
    def process_arrival(self):
        """Обработка прибытия автомобиля"""
        car_type = 1 if random.random() < self.config['priority_prob'] else 2
        self.car_counter += 1
        
        if car_type == 1:
            self.stats['cars_type1'] += 1
        else:
            self.stats['cars_type2'] += 1
        
        # Поиск свободного рабочего места (не отказавшего)
        free_ws = None
        for i in range(len(self.workstations)):
            if i not in self.failed_workstations and self.workstations[i] is None:
                free_ws = i
                break
        
        if free_ws is not None:
            # Немедленное обслуживание
            repair_time = self.exponential_random(self.config['repair_time_mean'])
            self.workstations[free_ws] = {
                'car_type': car_type, 
                'arrival_time': self.current_time,
                'start_time': self.current_time,
                'end_time': self.current_time + repair_time
            }
            self.schedule_event('repair_end', self.current_time + repair_time, free_ws)
            wait_time = 0
        else:
            # Ожидание в очереди
            self.repair_queue.append({
                'type': car_type, 
                'arrival_time': self.current_time
            })
            wait_time = -1
        
        # Сохраняем время ожидания
        if wait_time >= 0:
            if car_type == 1:
                self.stats['total_wait_time_type1'] += wait_time
                self.stats['wait_times_type1'].append(wait_time)
            else:
                self.stats['total_wait_time_type2'] += wait_time
                self.stats['wait_times_type2'].append(wait_time)
    
    def process_repair_end(self, workstation_id: int):
        """Обработка завершения ремонта"""
        car_data = self.workstations[workstation_id]
        
        # Проверяем, что данные автомобиля существуют
        if car_data is None:
            self.workstations[workstation_id] = None
            return
            
        car_type = car_data['car_type']
        
        # Переход на окраску
        if self.painting_booth is None:
            # Немедленная окраска
            painting_time = self.exponential_random(self.config['painting_time_mean'])
            self.painting_booth = {
                'start_time': self.current_time,
                'end_time': self.current_time + painting_time
            }
            self.schedule_event('painting_end', self.current_time + painting_time)
        else:
            # Ожидание в буфере окраски
            self.painting_buffer.append(self.current_time)
        
        # Освобождаем рабочее место
        service_start = car_data.get('start_time', car_data['arrival_time'])
        if workstation_id < len(self.stats['workstation_busy_time']):
            self.stats['workstation_busy_time'][workstation_id] += max(0.0, self.current_time - service_start)
        
        self.workstations[workstation_id] = None
        self.stats['cars_processed'] += 1
        
        # Обрабатываем очередь
        if self.repair_queue:
            next_car = self.repair_queue.pop(0)
            repair_time = self.exponential_random(self.config['repair_time_mean'])
            self.workstations[workstation_id] = {
                'car_type': next_car['type'],
                'arrival_time': next_car['arrival_time'],
                'start_time': self.current_time,
                'end_time': self.current_time + repair_time
            }
            self.schedule_event('repair_end', self.current_time + repair_time, workstation_id)
            
            # Вычисляем время ожидания
            wait_time = self.current_time - next_car['arrival_time']
            if next_car['type'] == 1:
                self.stats['total_wait_time_type1'] += wait_time
                self.stats['wait_times_type1'].append(wait_time)
            else:
                self.stats['total_wait_time_type2'] += wait_time
                self.stats['wait_times_type2'].append(wait_time)
        else:
            # Обновляем статистику занятости
            if workstation_id < len(self.stats['workstation_busy_time']):
                repair_duration = self.current_time - car_data['arrival_time']
                self.stats['workstation_busy_time'][workstation_id] += repair_duration
    
    def process_painting_end(self):
        """Обработка завершения окраски"""
        if self.painting_booth is not None:
            busy_duration = self.current_time - self.painting_booth.get('start_time', self.current_time)
            if busy_duration > 0:
                self.stats['painting_busy_time'] += busy_duration
        
        self.painting_booth = None
        
        # Обрабатываем буфер окраски
        if self.painting_buffer:
            painting_time = self.exponential_random(self.config['painting_time_mean'])
            self.painting_booth = {
                'start_time': self.current_time,
                'end_time': self.current_time + painting_time
            }
            self.schedule_event('painting_end', self.current_time + painting_time)
            self.painting_buffer.pop(0)
    
    def collect_results(self) -> SimulationResult:
        """Сбор результатов прогона"""
        # Вычисляем средние времена ожидания
        avg_wait_1 = (np.mean(self.stats['wait_times_type1']) 
                     if self.stats['wait_times_type1'] else 0.0)
        avg_wait_2 = (np.mean(self.stats['wait_times_type2']) 
                     if self.stats['wait_times_type2'] else 0.0)
        
        # Средняя длина очереди
        avg_queue = (np.mean(self.stats['repair_queue_lengths']) 
                    if self.stats['repair_queue_lengths'] else 0.0)
        
        # Загрузка оборудования
        workstation_util = []
        for i, busy_time in enumerate(self.stats['workstation_busy_time']):
            if i < len(self.workstations):
                util = min(1.0, busy_time / self.current_time) if self.current_time > 0 else 0.0
                workstation_util.append(util)
        
        painting_util = min(1.0, self.stats['painting_busy_time'] / self.current_time) if self.current_time > 0 else 0.0
        
        return SimulationResult(
            average_wait_time_type1=avg_wait_1,
            average_wait_time_type2=avg_wait_2,
            average_repair_queue_length=avg_queue,
            workstation_utilization=workstation_util,
            painting_utilization=painting_util,
            interruptions_count=self.stats['interruptions'],
            simulation_time=self.current_time,
            cars_processed=self.stats['cars_processed'],
            time_series_data={
                'queue_lengths': self.stats['repair_queue_lengths'],
                'time_points': self.stats['time_points']
            }
        )

class ExperimentController:
    """Контроллер для проведения экспериментов"""
    
    def __init__(self):
        self.base_config = {
            'num_workstations': 3,
            'arrival_rates': [8.0],
            'repair_time_mean': 12.0,
            'painting_time_mean': 8.0,
            'priority_prob': 0.3,
            'max_time': 2000,
            'max_cars': 500
        }
    
    @staticmethod
    def _check_stationarity(time_series: List[float]) -> bool:
        """Проверяет, возвращается ли система к стационарному режиму"""
        if not time_series or len(time_series) < 60:
            return False
        
        window = max(20, len(time_series) // 6)
        recent = np.mean(time_series[-window:])
        prev = np.mean(time_series[-2*window:-window])
        
        if prev == 0 and recent == 0:
            return True
        
        reference = max(abs(prev), 1.0)
        drift = abs(recent - prev) / reference
        
        # Дополнительно проверяем тренд методом линейной регрессии
        tail_series = time_series[-3*window:]
        tail_time = np.arange(len(tail_series))
        slope = np.polyfit(tail_time, tail_series, 1)[0]
        trend = abs(slope) * window
        
        return drift < 0.2 and trend < 0.5
    
    def run_multiple_simulations(self, config: Dict[str, Any], num_runs: int = 5) -> List[SimulationResult]:
        """Запуск множества симуляций с одинаковыми параметрами"""
        results = []
        for i in range(num_runs):
            model = AutoRepairStationModel(config)
            result = model.run()
            results.append(result)
        return results
    
    def task1_parameter_variation(self):
        """Задача 1: Зависимость отклика от варьирования параметра"""
        print("=" * 80)
        print("ЗАДАЧА 1: ЗАВИСИМОСТЬ ОТКЛИКА ОТ ПАРАМЕТРА")
        print("=" * 80)
        
        # Варьируем интенсивность прибытия на 7 уровнях
        arrival_rates = np.linspace(4.0, 20.0, 7)
        avg_wait_times = []
        avg_queue_lengths = []
        
        print("Варьирование интенсивности прибытия:")
        for rate in arrival_rates:
            config = self.base_config.copy()
            config['arrival_rates'] = [rate]
            
            results = self.run_multiple_simulations(config, 3)
            avg_wait = np.mean([r.average_wait_time_type1 for r in results if r.average_wait_time_type1 > 0])
            avg_queue = np.mean([r.average_repair_queue_length for r in results])
            
            avg_wait_times.append(avg_wait if not np.isnan(avg_wait) else 0)
            avg_queue_lengths.append(avg_queue if not np.isnan(avg_queue) else 0)
            print(f"  Интенсивность {rate:.1f}: время ожидания = {avg_wait:.2f}, очередь = {avg_queue:.2f}")
        
        # Удаляем нулевые значения для аппроксимации
        valid_indices = [i for i, x in enumerate(avg_wait_times) if x > 0]
        if len(valid_indices) < 3:
            print("Недостаточно данных для аппроксимации")
            return arrival_rates, avg_wait_times, "Недостаточно данных"
        
        valid_rates = [arrival_rates[i] for i in valid_indices]
        valid_wait_times = [avg_wait_times[i] for i in valid_indices]
        
        # Линейная аппроксимация
        try:
            linear_coeffs = np.polyfit(valid_rates, valid_wait_times, 1)
            linear_func = np.poly1d(linear_coeffs)
            linear_pred = linear_func(valid_rates)
        except:
            linear_pred = valid_wait_times
            linear_func = None
        
        # Квадратичная аппроксимация
        try:
            quadratic_coeffs = np.polyfit(valid_rates, valid_wait_times, 2)
            quadratic_func = np.poly1d(quadratic_coeffs)
            quadratic_pred = quadratic_func(valid_rates)
        except:
            quadratic_pred = valid_wait_times
            quadratic_func = None
        
        # Экспоненциальная аппроксимация
        def exp_func(x, a, b, c):
            return a * np.exp(b * x) + c
        
        try:
            exp_coeffs, _ = curve_fit(exp_func, valid_rates, valid_wait_times, 
                                    p0=[1, 0.1, 1], maxfev=1000)
            exp_pred = exp_func(valid_rates, *exp_coeffs)
        except:
            exp_pred = valid_wait_times
            exp_coeffs = [0, 0, 0]
            exp_func = None
        
        # Вычисление R² для каждой модели
        ss_tot = np.sum((valid_wait_times - np.mean(valid_wait_times)) ** 2) if len(valid_wait_times) > 1 else 1
        
        r2_linear = 0
        if linear_func is not None:
            ss_res_linear = np.sum((valid_wait_times - linear_pred) ** 2)
            r2_linear = 1 - (ss_res_linear / ss_tot) if ss_tot > 0 else 0
        
        r2_quadratic = 0
        if quadratic_func is not None:
            ss_res_quad = np.sum((valid_wait_times - quadratic_pred) ** 2)
            r2_quadratic = 1 - (ss_res_quad / ss_tot) if ss_tot > 0 else 0
        
        r2_exp = 0
        if exp_func is not None:
            ss_res_exp = np.sum((valid_wait_times - exp_pred) ** 2)
            r2_exp = 1 - (ss_res_exp / ss_tot) if ss_tot > 0 else 0
        
        print(f"\nКачество аппроксимации (R²):")
        print(f"  Линейная: {r2_linear:.4f}")
        print(f"  Квадратичная: {r2_quadratic:.4f}")
        print(f"  Экспоненциальная: {r2_exp:.4f}")
        
        # Визуализация
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.scatter(arrival_rates, avg_wait_times, color='blue', s=50, label='Данные модели')
        
        if linear_func is not None:
            x_fine = np.linspace(min(valid_rates), max(valid_rates), 100)
            plt.plot(x_fine, linear_func(x_fine), 'r-', label=f'Линейная (R²={r2_linear:.3f})')
        if quadratic_func is not None:
            plt.plot(x_fine, quadratic_func(x_fine), 'g-', label=f'Квадратичная (R²={r2_quadratic:.3f})')
        if exp_func is not None:
            plt.plot(x_fine, exp_func(x_fine, *exp_coeffs), 'orange', label=f'Экспоненциальная (R²={r2_exp:.3f})')
        
        plt.xlabel('Интенсивность прибытия (авт/ед. времени)')
        plt.ylabel('Среднее время ожидания')
        plt.title('Зависимость времени ожидания от интенсивности прибытия')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.scatter(arrival_rates, avg_queue_lengths, color='red', s=50)
        plt.plot(arrival_rates, avg_queue_lengths, 'r-')
        plt.xlabel('Интенсивность прибытия (авт/ед. времени)')
        plt.ylabel('Средняя длина очереди')
        plt.title('Зависимость длины очереди от интенсивности прибытия')
        plt.grid(True, alpha=0.3)
        
        # Сравнение ошибок аппроксимации
        plt.subplot(2, 2, 3)
        models = ['Линейная', 'Квадратичная', 'Экспоненциальная']
        r2_scores = [r2_linear, r2_quadratic, r2_exp]
        bars = plt.bar(models, r2_scores, color=['red', 'green', 'orange'])
        plt.ylabel('R² коэффициент')
        plt.title('Сравнение качества аппроксимации')
        for bar, score in zip(bars, r2_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom')
        
        # Определение лучшей модели
        if max(r2_scores) > 0:
            best_model_idx = np.argmax(r2_scores)
            best_model = models[best_model_idx]
        else:
            best_model = "Нет подходящей модели"
        
        print(f"\nЛучшая модель аппроксимации: {best_model}")
        
        plt.tight_layout()
        plt.savefig('task1_parameter_variation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return arrival_rates, avg_wait_times, best_model
    
    def task2_resource_failure_analysis(self):
        """Задача 2: Анализ отказа ресурсов"""
        print("\n" + "=" * 80)
        print("ЗАДАЧА 2: АНАЛИЗ ОТКАЗА РЕСУРСОВ")
        print("=" * 80)
        
        # Тестируем различное количество отказавших рабочих мест
        max_workstations = 5
        failure_scenarios = list(range(max_workstations))
        
        stability_results = []
        
        print("Анализ устойчивости системы к отказам:")
        for num_failures in failure_scenarios:
            config = self.base_config.copy()
            config['num_workstations'] = max_workstations
            config['max_time'] = 1500
            
            results = []
            for _ in range(3):
                model = AutoRepairStationModel(config)
                # Имитируем отказы - просто уменьшаем количество доступных рабочих мест
                model.failed_workstations = set(range(num_failures))
                result = model.run()
                results.append(result)
            
            avg_wait = np.mean([r.average_wait_time_type1 for r in results if r.average_wait_time_type1 > 0])
            avg_throughput = np.mean([r.cars_processed / r.simulation_time for r in results if r.simulation_time > 0])
            stationarity_flags = [
                self._check_stationarity(r.time_series_data.get('queue_lengths', []))
                for r in results
            ]
            stationarity_ratio = np.mean(stationarity_flags) if stationarity_flags else 0
            
            # Критерий стабильности: время ожидания не превышает 30 ед. времени
            is_stable = (avg_wait <= 30.0 and not np.isnan(avg_wait)
                         and stationarity_ratio >= 0.6)
            
            stability_results.append({
                'num_failures': num_failures,
                'avg_wait_time': avg_wait,
                'throughput': avg_throughput,
                'is_stable': is_stable,
                'stationarity_ratio': stationarity_ratio
            })
            
            status = "СТАБИЛЬНА" if is_stable else "НЕСТАБИЛЬНА"
            print(f"  Отказов: {num_failures}/{max_workstations}, "
                  f"Время ожидания: {avg_wait:.2f}, Пропускная способность: {avg_throughput:.3f}, "
                  f"Стационарность: {stationarity_ratio:.2f} - {status}")
        
        # Находим максимальное количество отказов для стабильности
        stable_failures = [r['num_failures'] for r in stability_results if r['is_stable']]
        max_stable_failures = max(stable_failures) if stable_failures else -1
        
        print(f"\nМаксимальное количество отказов для стабильности: {max_stable_failures}")
        
        # Визуализация
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        failures = [r['num_failures'] for r in stability_results]
        wait_times = [r['avg_wait_time'] for r in stability_results]
        colors = ['green' if r['is_stable'] else 'red' for r in stability_results]
        
        plt.bar(failures, wait_times, color=colors, alpha=0.7)
        plt.axhline(y=30, color='red', linestyle='--', label='Порог стабильности')
        plt.xlabel('Количество отказавших рабочих мест')
        plt.ylabel('Среднее время ожидания')
        plt.title('Влияние отказов на время ожидания')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        throughputs = [r['throughput'] for r in stability_results]
        sta_ratios = [r['stationarity_ratio'] for r in stability_results]
        plt.bar(failures, throughputs, color=colors, alpha=0.7, label='Пропускная способность')
        plt.plot(failures, sta_ratios, 'ko--', label='Доля стационарных прогонов')
        plt.xlabel('Количество отказавших рабочих мест')
        plt.ylabel('Пропускная способность / доля стационарности')
        plt.title('Влияние отказов на поведение системы')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('task2_resource_failure.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return max_stable_failures
    
    def task3_alternatives_comparison(self):
        """Задача 3: Сравнение альтернатив"""
        print("\n" + "=" * 80)
        print("ЗАДАЧА 3: СРАВНЕНИЕ АЛЬТЕРНАТИВ")
        print("=" * 80)
        
        alternatives = {
            'Базовая': {
                'num_workstations': 3,
                'arrival_rates': [8.0],
                'repair_time_mean': 12.0,
                'painting_time_mean': 8.0
            },
            'Ускоренный ремонт': {
                'num_workstations': 3,
                'arrival_rates': [8.0],
                'repair_time_mean': 8.0,
                'painting_time_mean': 8.0
            },
            'Дополнительные рабочие': {
                'num_workstations': 4,
                'arrival_rates': [8.0],
                'repair_time_mean': 12.0,
                'painting_time_mean': 8.0
            },
            'Приоритетная система': {
                'num_workstations': 3,
                'arrival_rates': [8.0],
                'repair_time_mean': 12.0,
                'painting_time_mean': 8.0,
                'priority_prob': 0.5
            }
        }
        
        comparison_results = []
        
        print("Сравнение альтернативных конфигураций:")
        for alt_name, config in alternatives.items():
            full_config = self.base_config.copy()
            full_config.update(config)
            
            results = self.run_multiple_simulations(full_config, 4)
            
            valid_wait_times = [r.average_wait_time_type1 for r in results if r.average_wait_time_type1 > 0]
            avg_wait = np.mean(valid_wait_times) if valid_wait_times else 0
            avg_wait_std = np.std(valid_wait_times) if valid_wait_times else 0
            throughput = np.mean([r.cars_processed / r.simulation_time for r in results if r.simulation_time > 0])
            avg_queue = np.mean([r.average_repair_queue_length for r in results])
            
            comparison_results.append({
                'name': alt_name,
                'avg_wait': avg_wait,
                'avg_wait_std': avg_wait_std,
                'throughput': throughput,
                'avg_queue': avg_queue,
                'config': config
            })
            
            print(f"  {alt_name}: время ожидания = {avg_wait:.2f} ± {avg_wait_std:.2f}, "
                  f"пропускная способность = {throughput:.3f}")
        
        # Визуализация сравнения
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
        ax_wait, ax_tp, ax_queue = axes
        
        names = [r['name'] for r in comparison_results]
        wait_times = [r['avg_wait'] for r in comparison_results]
        wait_stds = [r['avg_wait_std'] for r in comparison_results]
        x_positions = np.arange(len(names))
        
        bars_wait = ax_wait.bar(x_positions, wait_times, yerr=wait_stds, capsize=5, alpha=0.7,
                                color=['blue', 'green', 'orange', 'red'])
        ax_wait.set_ylabel('Среднее время ожидания')
        ax_wait.set_title('Сравнение времени ожидания')
        ax_wait.set_xticks(x_positions)
        ax_wait.set_xticklabels(names, rotation=45, ha='right')
        
        for bar, time in zip(bars_wait, wait_times):
            ax_wait.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                         f'{time:.1f}', ha='center', va='bottom')
        
        throughputs = [r['throughput'] for r in comparison_results]
        bars_tp = ax_tp.bar(x_positions, throughputs, alpha=0.7, color=['blue', 'green', 'orange', 'red'])
        ax_tp.set_ylabel('Пропускная способность')
        ax_tp.set_title('Сравнение пропускной способности')
        ax_tp.set_xticks(x_positions)
        ax_tp.set_xticklabels(names, rotation=45, ha='right')
        
        for bar, tp in zip(bars_tp, throughputs):
            ax_tp.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{tp:.3f}', ha='center', va='bottom')
        
        queues = [r['avg_queue'] for r in comparison_results]
        bars_queue = ax_queue.bar(x_positions, queues, alpha=0.7, color=['blue', 'green', 'orange', 'red'])
        ax_queue.set_ylabel('Средняя длина очереди')
        ax_queue.set_title('Сравнение длины очереди')
        ax_queue.set_xticks(x_positions)
        ax_queue.set_xticklabels(names, rotation=45, ha='right')
        
        for bar, queue in zip(bars_queue, queues):
            ax_queue.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                          f'{queue:.2f}', ha='center', va='bottom')
        
        plt.savefig('task3_alternatives_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Определение лучшей альтернативы
        valid_results = [r for r in comparison_results if r['avg_wait'] > 0]
        if valid_results:
            best_alt_idx = np.argmin([r['avg_wait'] for r in valid_results])
            best_alternative = valid_results[best_alt_idx]
            print(f"\nЛучшая альтернатива: {best_alternative['name']} "
                  f"(время ожидания: {best_alternative['avg_wait']:.2f})")
        else:
            best_alternative = comparison_results[0]
            print(f"\nНе удалось определить лучшую альтернативу")
        
        return comparison_results, best_alternative
    
    def task4_two_factor_experiment(self):
        """Задача 4: Двухфакторный эксперимент"""
        print("\n" + "=" * 80)
        print("ЗАДАЧА 4: ДВУХФАКТОРНЫЙ ЭКСПЕРИМЕНТ")
        print("=" * 80)
        
        # Факторы: интенсивность прибытия и количество рабочих мест
        arrival_levels = np.linspace(5.0, 15.0, 5)
        workstation_levels = [2, 3, 4, 5]
        
        results_matrix = np.zeros((len(workstation_levels), len(arrival_levels)))
        
        print("Двухфакторный эксперимент:")
        print("Фактор A (рабочие места):", workstation_levels)
        print("Фактор B (интенсивность):", [f"{x:.1f}" for x in arrival_levels])
        
        for i, num_ws in enumerate(workstation_levels):
            for j, arrival_rate in enumerate(arrival_levels):
                config = self.base_config.copy()
                config['num_workstations'] = num_ws
                config['arrival_rates'] = [arrival_rate]
                
                results = self.run_multiple_simulations(config, 3)
                valid_wait_times = [r.average_wait_time_type1 for r in results if r.average_wait_time_type1 > 0]
                avg_wait = np.mean(valid_wait_times) if valid_wait_times else 0
                results_matrix[i, j] = avg_wait
                
                print(f"  Рабочих мест: {num_ws}, Интенсивность: {arrival_rate:.1f} -> "
                      f"Время ожидания: {avg_wait:.2f}")
        
        # Анализ значимости факторов
        print(f"\nАнализ значимости факторов:")
        
        # Упрощенный анализ значимости
        mean_by_workstations = []
        for i in range(len(workstation_levels)):
            valid_values = [results_matrix[i, j] for j in range(len(arrival_levels)) if results_matrix[i, j] > 0]
            mean_by_workstations.append(np.mean(valid_values) if valid_values else 0)
        
        mean_by_arrival = []
        for j in range(len(arrival_levels)):
            valid_values = [results_matrix[i, j] for i in range(len(workstation_levels)) if results_matrix[i, j] > 0]
            mean_by_arrival.append(np.mean(valid_values) if valid_values else 0)
        
        valid_ws_means = [x for x in mean_by_workstations if x > 0]
        valid_arr_means = [x for x in mean_by_arrival if x > 0]
        
        if len(valid_ws_means) > 1 and len(valid_arr_means) > 1:
            range_workstations = max(valid_ws_means) - min(valid_ws_means)
            range_arrival = max(valid_arr_means) - min(valid_arr_means)
            
            total_range = range_workstations + range_arrival
            importance_workstations = range_workstations / total_range if total_range > 0 else 0.5
            importance_arrival = range_arrival / total_range if total_range > 0 else 0.5
        else:
            importance_workstations = 0.5
            importance_arrival = 0.5
        
        print(f"  Вариация от рабочих мест: {importance_workstations:.1%}")
        print(f"  Вариация от интенсивности: {importance_arrival:.1%}")
        
        if importance_workstations > importance_arrival:
            print(f"  Более значимый фактор: Количество рабочих мест")
        else:
            print(f"  Более значимый фактор: Интенсивность прибытия")
        
        # 3D визуализация поверхности отклика
        fig = plt.figure(figsize=(15, 6))
        
        # 3D поверхность
        ax1 = fig.add_subplot(121, projection='3d')
        X, Y = np.meshgrid(arrival_levels, workstation_levels)
        surf = ax1.plot_surface(X, Y, results_matrix, cmap='viridis', alpha=0.8)
        
        ax1.set_xlabel('Интенсивность прибытия')
        ax1.set_ylabel('Количество рабочих мест')
        ax1.set_zlabel('Время ожидания')
        ax1.set_title('Поверхность отклика: время ожидания')
        fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
        
        # 2D тепловая карта
        ax2 = fig.add_subplot(122)
        im = ax2.imshow(results_matrix, cmap='viridis', aspect='auto', 
                       extent=[min(arrival_levels), max(arrival_levels), 
                              min(workstation_levels), max(workstation_levels)])
        
        ax2.set_xlabel('Интенсивность прибытия')
        ax2.set_ylabel('Количество рабочих мест')
        ax2.set_title('Тепловая карта времени ожидания')
        
        # Добавляем аннотации
        for i in range(len(workstation_levels)):
            for j in range(len(arrival_levels)):
                if results_matrix[i, j] > 0:
                    ax2.text(arrival_levels[j], workstation_levels[i], 
                            f'{results_matrix[i, j]:.1f}', 
                            ha='center', va='center', color='white', fontweight='bold')
        
        fig.colorbar(im, ax=ax2, shrink=0.8)
        
        plt.tight_layout()
        plt.savefig('task4_two_factor_experiment.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return results_matrix, importance_workstations, importance_arrival
    
    def run_all_experiments(self):
        """Запуск всех экспериментов"""
        print("ЛАБОРАТОРНАЯ РАБОТА №4: ПОСТАНОВКА ЭКСПЕРИМЕНТОВ")
        print("=" * 80)
        
        try:
            # Задача 1
            arrival_rates, wait_times, best_model = self.task1_parameter_variation()
            
            # Задача 2
            max_stable_failures = self.task2_resource_failure_analysis()
            
            # Задача 3
            alternatives, best_alternative = self.task3_alternatives_comparison()
            
            # Задача 4
            response_surface, importance_ws, importance_arrival = self.task4_two_factor_experiment()
            
            # Сводка результатов
            print("\n" + "=" * 80)
            print("СВОДКА РЕЗУЛЬТАТОВ ЛАБОРАТОРНОЙ РАБОТЫ №4")
            print("=" * 80)
            
            print(f"1. Аппроксимация зависимости: лучшая модель - {best_model}")
            print(f"2. Устойчивость к отказам: система выдерживает до {max_stable_failures} отказов")
            print(f"3. Лучшая альтернатива: {best_alternative['name']} "
                  f"(время ожидания: {best_alternative['avg_wait']:.2f})")
            print(f"4. Двухфакторный эксперимент:")
            print(f"   - Значимость рабочих мест: {importance_ws:.1%}")
            print(f"   - Значимость интенсивности: {importance_arrival:.1%}")
            
            if importance_ws > importance_arrival:
                print(f"   - Более значимый фактор: Количество рабочих мест")
            else:
                print(f"   - Более значимый фактор: Интенсивность прибытия")
                
        except Exception as e:
            print(f"Произошла ошибка при выполнении экспериментов: {e}")
            print("Попробуйте увеличить количество прогонов или изменить параметры модели")

# Запуск экспериментов
if __name__ == "__main__":
    controller = ExperimentController()
    controller.run_all_experiments()