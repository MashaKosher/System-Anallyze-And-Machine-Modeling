import heapq
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

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
        
        # Статистика
        self.stats = {
            'cars_processed': 0, 'cars_type1': 0, 'cars_type2': 0,
            'total_wait_time_type1': 0.0, 'total_wait_time_type2': 0.0,
            'interruptions': 0, 
            'workstation_busy_time': [0.0] * self.config['num_workstations'],
            'painting_busy_time': 0.0, 
            'repair_queue_lengths': [], 'painting_queue_lengths': [],
            'time_points': [], 
            'workstation_utilization_history': [],
            'wait_times_type1': [], 'wait_times_type2': [],
            'cars_in_system': [],
            'instantaneous_wait_times': []
        }
        
        self.event_queue = []
        
    def exponential_random(self, mean: float) -> float:
        return -mean * math.log(1 - random.random() + 1e-10)
    
    def schedule_event(self, event_type: str, time: float, data: Any = None):
        heapq.heappush(self.event_queue, (time, event_type, data))
    
    def run(self, continuous_mode: bool = False) -> SimulationResult:
        """Запуск одного прогона модели"""
        max_time = self.config.get('max_time', 5000)
        stat_interval = self.config.get('stat_interval', 10.0)
        
        # Планируем начальное прибытие
        self.schedule_event('arrival', self.exponential_random(self.config['arrival_rates'][0]))
        
        # Фиксируем начальное состояние системы
        self.update_stats()
        
        last_stat_time = 0
        
        while self.event_queue and self.current_time < max_time:
            time, event_type, data = heapq.heappop(self.event_queue)
            self.current_time = time
            
            # Обновление статистики для непрерывного режима
            if continuous_mode and self.current_time >= last_stat_time:
                self.update_continuous_stats()
                last_stat_time += stat_interval
            
            if event_type == 'arrival':
                self.process_arrival()
                # Планируем следующее прибытие
                if self.stats['cars_processed'] < self.config.get('max_cars', 2000):
                    next_arrival = self.current_time + self.exponential_random(self.config['arrival_rates'][0])
                    self.schedule_event('arrival', next_arrival)
                    
            elif event_type == 'repair_end':
                self.process_repair_end(data)
                
            elif event_type == 'painting_end':
                self.process_painting_end()
            
            # Фиксируем состояние после обработки события
            self.update_stats()
        
        # Финальное обновление статистики
        self.update_stats()
        
        return self.collect_results()
    
    def process_arrival(self):
        """Обработка прибытия автомобиля"""
        car_type = 1 if random.random() < self.config['priority_prob'] else 2
        
        if car_type == 1:
            self.stats['cars_type1'] += 1
        else:
            self.stats['cars_type2'] += 1
        
        # Поиск свободного рабочего места
        free_ws = None
        for i, ws in enumerate(self.workstations):
            if ws is None:
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
        if car_data is None:
            return
            
        car_type = car_data['car_type']
        
        # Переход на окраску
        if self.painting_booth is None:
            painting_time = self.exponential_random(self.config['painting_time_mean'])
            self.painting_booth = {
                'start_time': self.current_time,
                'end_time': self.current_time + painting_time
            }
            self.schedule_event('painting_end', self.painting_booth['end_time'])
        else:
            self.painting_buffer.append(self.current_time)
        
        # Освобождаем рабочее место
        self.workstations[workstation_id] = None
        self.stats['cars_processed'] += 1
        
        # Обновляем статистику занятости
        repair_duration = self.current_time - car_data.get('start_time', car_data['arrival_time'])
        if workstation_id < len(self.stats['workstation_busy_time']):
            self.stats['workstation_busy_time'][workstation_id] += repair_duration
        
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
            
            wait_time = self.current_time - next_car['arrival_time']
            if next_car['type'] == 1:
                self.stats['total_wait_time_type1'] += wait_time
                self.stats['wait_times_type1'].append(wait_time)
            else:
                self.stats['total_wait_time_type2'] += wait_time
                self.stats['wait_times_type2'].append(wait_time)
    
    def process_painting_end(self):
        """Обработка завершения окраски"""
        if self.painting_booth is not None:
            busy_duration = self.current_time - self.painting_booth.get('start_time', self.current_time)
            if busy_duration > 0:
                self.stats['painting_busy_time'] += busy_duration
        
        self.painting_booth = None
        
        if self.painting_buffer:
            painting_time = self.exponential_random(self.config['painting_time_mean'])
            self.painting_booth = {
                'start_time': self.current_time,
                'end_time': self.current_time + painting_time
            }
            self.schedule_event('painting_end', self.painting_booth['end_time'])
            self.painting_buffer.pop(0)
    
    def update_stats(self):
        """Обновление статистики"""
        self.stats['repair_queue_lengths'].append(len(self.repair_queue))
        self.stats['painting_queue_lengths'].append(len(self.painting_buffer))
        self.stats['time_points'].append(self.current_time)
        
        busy_count = sum(1 for ws in self.workstations if ws is not None)
        utilization = busy_count / len(self.workstations)
        self.stats['workstation_utilization_history'].append(utilization)
        
        cars_in_system = (len(self.repair_queue) + len(self.painting_buffer) +
                         sum(1 for ws in self.workstations if ws is not None) +
                         (1 if self.painting_booth else 0))
        self.stats['cars_in_system'].append(cars_in_system)
    
    def update_continuous_stats(self):
        """Обновление статистики для непрерывного режима"""
        current_wait = np.mean(self.stats['wait_times_type1'][-10:]) if self.stats['wait_times_type1'] else 0
        self.stats['instantaneous_wait_times'].append({
            'time': self.current_time,
            'wait_time': current_wait,
            'queue_length': len(self.repair_queue)
        })
    
    def collect_results(self) -> SimulationResult:
        """Сбор результатов прогона"""
        avg_wait_1 = (np.mean(self.stats['wait_times_type1']) 
                     if self.stats['wait_times_type1'] else 0.0)
        avg_wait_2 = (np.mean(self.stats['wait_times_type2']) 
                     if self.stats['wait_times_type2'] else 0.0)
        
        avg_queue = (np.mean(self.stats['repair_queue_lengths']) 
                    if self.stats['repair_queue_lengths'] else 0.0)
        
        workstation_util = [min(1.0, busy_time / self.current_time) 
                           for busy_time in self.stats['workstation_busy_time']]
        painting_util = min(1.0, self.stats['painting_busy_time'] / self.current_time)
        
        return SimulationResult(
            average_wait_time_type1=avg_wait_1,
            average_wait_time_type2=avg_wait_2,
            average_repair_queue_length=avg_queue,
            workstation_utilization=workstation_util,
            painting_utilization=painting_util,
            interruptions_count=self.stats['interruptions'],
            simulation_time=self.current_time,
            time_series_data={
                'time_points': self.stats['time_points'],
                'queue_lengths': self.stats['repair_queue_lengths'],
                'utilization': self.stats['workstation_utilization_history'],
                'cars_in_system': self.stats['cars_in_system'],
                'instantaneous_wait_times': self.stats['instantaneous_wait_times']
            }
        )

class StatisticalAnalyzer:
    """Класс для статистического анализа результатов моделирования"""
    
    def __init__(self):
        self.results = []
    
    def test_normality_chi2(self, data: List[float], alpha: float = 0.05) -> Dict[str, Any]:
        """Проверка гипотезы о нормальности с помощью критерия хи-квадрат"""
        if len(data) < 10:
            return {'is_normal': False, 'p_value': 0, 'reason': 'Недостаточно данных'}
        
        clean_data = [x for x in data if np.isfinite(x)]
        if len(clean_data) < 10:
            return {'is_normal': False, 'p_value': 0, 'reason': 'Недостаточно корректных данных'}
        
        try:
            # Группируем данные в интервалы
            n_bins = max(3, min(8, int(1 + 3.322 * math.log10(len(clean_data)))))
            hist, bin_edges = np.histogram(clean_data, bins=n_bins, density=False)
            
            # Параметры нормального распределения
            mean = np.mean(clean_data)
            std = np.std(clean_data, ddof=1)
            
            if std == 0:
                return {'is_normal': False, 'p_value': 0, 'reason': 'Нулевое стандартное отклонение'}
            
            # Ожидаемые частоты
            expected_freq = []
            for i in range(len(hist)):
                p_low = stats.norm.cdf(bin_edges[i], mean, std)
                p_high = stats.norm.cdf(bin_edges[i+1], mean, std)
                expected = (p_high - p_low) * len(clean_data)
                expected_freq.append(max(expected, 1.0))
            
            # Объединяем интервалы с ожидаемой частотой < 5
            observed_combined = []
            expected_combined = []
            current_obs = 0
            current_exp = 0
            
            for i in range(len(hist)):
                current_obs += hist[i]
                current_exp += expected_freq[i]
                
                if current_exp >= 5 or i == len(hist) - 1:
                    observed_combined.append(current_obs)
                    expected_combined.append(current_exp)
                    current_obs = 0
                    current_exp = 0
            
            if len(observed_combined) < 3:
                return {'is_normal': False, 'p_value': 0, 'reason': 'Слишком мало интервалов'}
            
            # Статистика хи-квадрат
            chi2_stat = sum((o - e)**2 / e for o, e in zip(observed_combined, expected_combined))
            df = len(observed_combined) - 3
            
            if df <= 0:
                return {'is_normal': False, 'p_value': 0, 'reason': 'Недостаточно степеней свободы'}
            
            p_value = 1 - stats.chi2.cdf(chi2_stat, df)
            
            return {
                'is_normal': p_value > alpha,
                'p_value': p_value,
                'chi2_statistic': chi2_stat,
                'degrees_of_freedom': df,
                'mean': mean,
                'std': std
            }
            
        except Exception as e:
            return {'is_normal': False, 'p_value': 0, 'reason': f'Ошибка вычислений: {str(e)}'}
    
    def calculate_confidence_intervals(self, data: List[float], alpha: float = 0.05) -> Dict[str, float]:
        """Вычисление точечных и интервальных оценок"""
        n = len(data)
        if n < 2:
            return {}
        
        clean_data = [x for x in data if np.isfinite(x)]
        if len(clean_data) < 2:
            return {}
        
        mean = np.mean(clean_data)
        std = np.std(clean_data, ddof=1)
        se = std / math.sqrt(len(clean_data))
        
        t_critical = stats.t.ppf(1 - alpha/2, len(clean_data)-1)
        margin_of_error = t_critical * se
        
        return {
            'point_estimate': mean,
            'std_error': se,
            'confidence_interval': (mean - margin_of_error, mean + margin_of_error),
            'margin_of_error': margin_of_error,
            'relative_error': margin_of_error / mean if mean != 0 else float('inf')
        }
    
    def find_required_runs_for_precision(self, data: List[float], target_relative_error: float = 0.05) -> Dict[str, Any]:
        """Определение необходимого количества прогонов для достижения точности"""
        if len(data) < 2:
            return {'required_runs': 0, 'current_error': float('inf')}
        
        clean_data = [x for x in data if np.isfinite(x) and x > 0]
        if len(clean_data) < 2:
            return {'required_runs': 0, 'current_error': float('inf')}
        
        current_mean = np.mean(clean_data)
        current_std = np.std(clean_data, ddof=1)
        
        t_critical = stats.t.ppf(0.975, len(clean_data)-1)
        required_n = math.ceil((t_critical * current_std / (current_mean * target_relative_error))**2)
        
        current_relative_error = (t_critical * current_std / (current_mean * math.sqrt(len(clean_data))))
        
        return {
            'required_runs': required_n,
            'current_runs': len(clean_data),
            'current_relative_error': current_relative_error,
            'target_relative_error': target_relative_error
        }
    
    def analyze_transient_period(self, time_series: List[float], threshold: float = 0.05) -> Dict[str, Any]:
        """Анализ переходного периода"""
        if len(time_series) < 50:
            full_mean = np.mean(time_series) if time_series else 0
            return {
                'transient_period': 0,
                'stationary_mean': full_mean,
                'full_mean': full_mean,
                'reduction_ratio': 0
            }
        
        # Используем метод скользящего среднего для определения стабильности
        window_size = min(20, len(time_series) // 10)
        stationary_start = 0
        
        for i in range(window_size, len(time_series) - window_size, window_size):
            window_before = time_series[i-window_size:i]
            window_after = time_series[i:i+window_size]
            
            mean_before = np.mean(window_before)
            mean_after = np.mean(window_after)
            
            if mean_before > 0 and abs(mean_after - mean_before) / mean_before < threshold:
                stationary_start = i
                break
        
        full_mean = np.mean(time_series) if time_series else 0
        stationary_data = time_series[stationary_start:]
        stationary_mean = np.mean(stationary_data) if len(stationary_data) > 0 else full_mean
        
        return {
            'transient_period': stationary_start,
            'stationary_mean': stationary_mean,
            'full_mean': full_mean,
            'reduction_ratio': stationary_start / len(time_series) if time_series else 0
        }
    
    def test_continuous_run_feasibility(self, time_series: List[float], segment_count: int = 4) -> Dict[str, Any]:
        """Проверка возможности непрерывного прогона"""
        if len(time_series) < segment_count * 10:
            return {
                'feasible': False,
                'p_value': 0,
                'segment_means': [],
                'f_statistic': 0,
                'reason': 'Недостаточно данных'
            }
        
        segment_size = len(time_series) // segment_count
        segment_means = []
        
        for i in range(segment_count):
            segment = time_series[i*segment_size:(i+1)*segment_size]
            segment_means.append(np.mean(segment))
        
        # Проверка стационарности с помощью критерия ANOVA
        segments = [time_series[i*segment_size:(i+1)*segment_size] for i in range(segment_count)]
        try:
            f_stat, p_value = stats.f_oneway(*segments)
        except:
            p_value = 0.0
            f_stat = 0.0
        
        # Критерий: если p-value > 0.05, то средние не отличаются значимо
        is_feasible = p_value > 0.05
        
        return {
            'feasible': is_feasible,
            'p_value': p_value,
            'segment_means': segment_means,
            'f_statistic': f_stat
        }
    
    def sensitivity_analysis(self, base_results: List[float], varied_results: Dict[str, List[float]]) -> Dict[str, Any]:
        """Анализ чувствительности к вариациям параметров"""
        base_mean = np.mean(base_results) if base_results else 0
        sensitivity_scores = {}
        
        for param_name, results in varied_results.items():
            varied_mean = np.mean(results) if results else 0
            sensitivity = abs(varied_mean - base_mean) / base_mean if base_mean > 0 else 0
            sensitivity_scores[param_name] = sensitivity
        
        # Определение достаточной точности (когда изменение < 1%)
        sufficient_precision = 0.01
        critical_params = {k: v for k, v in sensitivity_scores.items() if v > sufficient_precision}
        
        return {
            'sensitivity_scores': sensitivity_scores,
            'critical_params': critical_params,
            'sufficient_precision': sufficient_precision
        }


def evaluate_config_for_stationarity(config: Dict[str, Any],
                                     analyzer: StatisticalAnalyzer,
                                     long_runs: int = 15) -> Dict[str, Any]:
    """Запускает серию прогонов и оценивает нормальность/стационарность"""
    results = []
    for _ in range(long_runs):
        model = AutoRepairStationModel(config)
        results.append(model.run())
    
    wait_times = [r.average_wait_time_type1 for r in results]
    queue_lengths = [r.average_repair_queue_length for r in results]
    
    normal_wait = analyzer.test_normality_chi2(wait_times)
    normal_queue = analyzer.test_normality_chi2(queue_lengths)
    
    time_series = results[-1].time_series_data['queue_lengths']
    transient = analyzer.analyze_transient_period(time_series)
    continuous = analyzer.test_continuous_run_feasibility(time_series)
    
    normal_pass = normal_wait['is_normal'] and normal_queue['is_normal']
    stationary_pass = (transient['reduction_ratio'] < 0.2) and continuous['feasible']
    
    # Интегральный скор для выбора лучшей конфигурации
    combined_score = (
        normal_wait.get('p_value', 0) +
        normal_queue.get('p_value', 0) +
        continuous.get('p_value', 0)
        - transient.get('reduction_ratio', 1)
    )
    
    return {
        'normality': {'wait': normal_wait, 'queue': normal_queue},
        'transient': transient,
        'continuous': continuous,
        'normal_pass': normal_pass,
        'stationary_pass': stationary_pass,
        'score': combined_score,
        'results': results
    }


def tune_parameters_for_stationarity(base_config: Dict[str, Any],
                                     analyzer: StatisticalAnalyzer) -> Dict[str, Any]:
    """Подбор параметров, обеспечивающих нормальность и стационарность"""
    arrival_candidates = [7.0, 8.0, 9.0, 10.0]
    repair_candidates = [10.0, 11.0, 12.0]
    painting_candidates = [6.5, 7.5, 8.5]
    workstation_candidates = [3, 4]
    
    best_config = base_config.copy()
    best_metrics = None
    
    print("\nАВТОКАЛИБРОВКА ПАРАМЕТРОВ ДЛЯ СТАЦИОНАРНОСТИ...")
    
    for arrival in arrival_candidates:
        for repair in repair_candidates:
            for painting in painting_candidates:
                for ws in workstation_candidates:
                    candidate = base_config.copy()
                    candidate.update({
                        'arrival_rates': [arrival],
                        'repair_time_mean': repair,
                        'painting_time_mean': painting,
                        'num_workstations': ws,
                        'max_time': max(base_config.get('max_time', 3000), 3000)
                    })
                    
                    metrics = evaluate_config_for_stationarity(candidate, analyzer, long_runs=12)
                    
                    print(f"  Проверка конфигурации: arrivals={arrival}, repair={repair}, "
                          f"painting={painting}, workstations={ws} "
                          f"| p_norm(wait)={metrics['normality']['wait'].get('p_value', 0):.3f}, "
                          f"p_anova={metrics['continuous'].get('p_value', 0):.3f}, "
                          f"transient_ratio={metrics['transient'].get('reduction_ratio', 1):.2f}")
                    
                    if metrics['normal_pass'] and metrics['stationary_pass']:
                        print("  --> Конфигурация удовлетворяет критериям. Используем её далее.")
                        metrics['chosen_config'] = candidate
                        return metrics
                    
                    if best_metrics is None or metrics['score'] > best_metrics['score']:
                        best_metrics = metrics
                        best_metrics['chosen_config'] = candidate
                        best_config = candidate.copy()
    
    print("  --> Полностью удовлетворяющая конфигурация не найдена, "
          "используем наилучшую по совокупности критериев.")
    return best_metrics if best_metrics else {'chosen_config': best_config}

def run_comprehensive_analysis():
    """Комплексный анализ свойств имитационной модели"""
    
    BASE_CONFIG = {
        'num_workstations': 3,
        'arrival_rates': [8.0],
        'repair_time_mean': 12.0,
        'painting_time_mean': 8.0,
        'priority_prob': 0.3,
        'max_time': 3000,  # Уменьшим для скорости
        'stat_interval': 10.0,
        'max_cars': 500
    }
    
    analyzer = StatisticalAnalyzer()

    tuning_result = tune_parameters_for_stationarity(BASE_CONFIG, analyzer)
    if tuning_result and 'chosen_config' in tuning_result:
        BASE_CONFIG = tuning_result['chosen_config']
        print("\nИспользуем откалиброванные параметры:")
        print(f"  Рабочих мест: {BASE_CONFIG['num_workstations']}")
        print(f"  Среднее межприходное время: {BASE_CONFIG['arrival_rates'][0]}")
        print(f"  Среднее время ремонта: {BASE_CONFIG['repair_time_mean']}")
        print(f"  Среднее время окраски: {BASE_CONFIG['painting_time_mean']}")
        print(f"  Макс. модельное время: {BASE_CONFIG['max_time']}")
    else:
        print("\nАвтокалибровка не изменила параметры (используем значения по умолчанию).")
    
    print("=" * 80)
    print("ЛАБОРАТОРНАЯ РАБОТА: ИССЛЕДОВАНИЕ СВОЙСТВ ИМИТАЦИОННОЙ МОДЕЛИ")
    print("=" * 80)
    
    # Задача 1: Проверка нормальности распределения откликов
    print("\n1. ПРОВЕРКА НОРМАЛЬНОСТИ РАСПРЕДЕЛЕНИЯ ОТКЛИКОВ (хи-квадрат)")
    print("-" * 60)
    
    # Длинные прогоны (30 прогонов для скорости)
    print("Проведение 30 длинных прогонов...")
    long_run_results = []
    for i in range(1000):
        if i % 10 == 0:
            print(f"  Прогон {i+1}/30...")
        model = AutoRepairStationModel(BASE_CONFIG)
        result = model.run()
        long_run_results.append(result)
    
    wait_times_1 = [r.average_wait_time_type1 for r in long_run_results]
    queue_lengths = [r.average_repair_queue_length for r in long_run_results]
    
    # Проверка нормальности критерием хи-квадрат
    normality_chi2_1 = analyzer.test_normality_chi2(wait_times_1)
    normality_chi2_2 = analyzer.test_normality_chi2(queue_lengths)
    
    print(f"Время ожидания приоритетных автомобилей:")
    print(f"  Нормальное распределение: {'ДА' if normality_chi2_1['is_normal'] else 'НЕТ'}")
    print(f"  p-value: {normality_chi2_1['p_value']:.4f}")
    if 'chi2_statistic' in normality_chi2_1:
        print(f"  Статистика хи-квадрат: {normality_chi2_1['chi2_statistic']:.4f}")
    
    print(f"\nСредняя длина очереди:")
    print(f"  Нормальное распределение: {'ДА' if normality_chi2_2['is_normal'] else 'НЕТ'}")
    print(f"  p-value: {normality_chi2_2['p_value']:.4f}")
    if 'chi2_statistic' in normality_chi2_2:
        print(f"  Статистика хи-квадрат: {normality_chi2_2['chi2_statistic']:.4f}")
    
    # Визуализация распределений
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].hist(wait_times_1, bins=15, alpha=0.7, color='skyblue', edgecolor='black', density=True)
    axes[0, 0].set_title('Распределение времени ожидания')
    axes[0, 0].set_xlabel('Время ожидания')
    axes[0, 0].set_ylabel('Плотность')
    
    if 'mean' in normality_chi2_1 and 'std' in normality_chi2_1 and normality_chi2_1['std'] > 0:
        x = np.linspace(min(wait_times_1), max(wait_times_1), 100)
        y = stats.norm.pdf(x, normality_chi2_1['mean'], normality_chi2_1['std'])
        axes[0, 0].plot(x, y, 'r-', linewidth=2, label='Нормальное распределение')
        axes[0, 0].legend()
    
    axes[0, 1].hist(queue_lengths, bins=15, alpha=0.7, color='lightcoral', edgecolor='black', density=True)
    axes[0, 1].set_title('Распределение длины очереди')
    axes[0, 1].set_xlabel('Длина очереди')
    axes[0, 1].set_ylabel('Плотность')
    
    if 'mean' in normality_chi2_2 and 'std' in normality_chi2_2 and normality_chi2_2['std'] > 0:
        x = np.linspace(min(queue_lengths), max(queue_lengths), 100)
        y = stats.norm.pdf(x, normality_chi2_2['mean'], normality_chi2_2['std'])
        axes[0, 1].plot(x, y, 'r-', linewidth=2, label='Нормальное распределение')
        axes[0, 1].legend()
    
    # QQ-plot для дополнительной проверки
    stats.probplot(wait_times_1, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('QQ-plot времени ожидания')
    
    stats.probplot(queue_lengths, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('QQ-plot длины очереди')
    
    plt.tight_layout()
    plt.savefig('task1_normality_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Задача 2: Точечные и интервальные оценки (20 прогонов)
    print("\n2. ТОЧЕЧНЫЕ И ИНТЕРВАЛЬНЫЕ ОЦЕНКИ (20 прогонов)")
    print("-" * 60)
    
    twenty_run_results = long_run_results[:20]
    wait_times_20 = [r.average_wait_time_type1 for r in twenty_run_results]
    queue_lengths_20 = [r.average_repair_queue_length for r in twenty_run_results]
    
    ci_wait = analyzer.calculate_confidence_intervals(wait_times_20)
    ci_queue = analyzer.calculate_confidence_intervals(queue_lengths_20)
    
    if ci_wait:
        print("Время ожидания приоритетных автомобилей:")
        print(f"  Точечная оценка: {ci_wait['point_estimate']:.3f}")
        print(f"  95% доверительный интервал: [{ci_wait['confidence_interval'][0]:.3f}, {ci_wait['confidence_interval'][1]:.3f}]")
        print(f"  Относительная погрешность: {ci_wait['relative_error']:.1%}")
    
    if ci_queue:
        print("\nСредняя длина очереди:")
        print(f"  Точечная оценка: {ci_queue['point_estimate']:.3f}")
        print(f"  95% доверительный интервал: [{ci_queue['confidence_interval'][0]:.3f}, {ci_queue['confidence_interval'][1]:.3f}]")
        print(f"  Относительная погрешность: {ci_queue['relative_error']:.1%}")
    
    # Задача 3: Зависимость точности от количества прогонов
    print("\n3. ЗАВИСИМОСТЬ ТОЧНОСТИ ОТ КОЛИЧЕСТВА ПРОГОНОВ")
    print("-" * 60)
    
    precision_analysis = analyzer.find_required_runs_for_precision(wait_times_20, 0.05)
    
    print(f"Текущее количество прогонов: {precision_analysis['current_runs']}")
    print(f"Текущая относительная погрешность: {precision_analysis['current_relative_error']:.3f}")
    print(f"Целевая относительная погрешность: {precision_analysis['target_relative_error']:.3f}")
    print(f"Необходимое количество прогонов для 5% погрешности: {precision_analysis['required_runs']}")
    
    # График зависимости точности от количества прогонов
    run_counts = list(range(5, len(wait_times_20) + 1))
    errors = []
    
    for n in run_counts:
        subset = wait_times_20[:n]
        ci = analyzer.calculate_confidence_intervals(subset)
        if ci:
            errors.append(ci['relative_error'])
        else:
            errors.append(float('inf'))
    
    plt.figure(figsize=(10, 6))
    plt.plot(run_counts, errors, 'b-', linewidth=2, marker='o')
    plt.axhline(y=0.05, color='r', linestyle='--', label='5% погрешность')
    if precision_analysis['required_runs'] <= len(wait_times_20):
        plt.axvline(x=precision_analysis['required_runs'], color='g', linestyle='--', 
                    label=f'Требуется прогонов: {precision_analysis["required_runs"]}')
    plt.xlabel('Количество прогонов')
    plt.ylabel('Относительная погрешность')
    plt.title('Зависимость точности от количества прогонов')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('task3_precision_vs_runs.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Задача 4: Анализ переходного периода
    print("\n4. АНАЛИЗ ПЕРЕХОДНОГО ПЕРИОДА В МОДЕЛЬНОМ ВРЕМЕНИ")
    print("-" * 60)
    
    if long_run_results:
        last_result = long_run_results[-1]
        time_series = last_result.time_series_data['queue_lengths']
        time_points = last_result.time_series_data['time_points']
        
        if time_series:
            transient_analysis = analyzer.analyze_transient_period(time_series)
            
            print(f"Длина временного ряда: {len(time_series)}")
            print(f"Начало стационарного периода: {transient_analysis['transient_period']}")
            print(f"Среднее за весь период: {transient_analysis['full_mean']:.3f}")
            print(f"Среднее в стационарном периоде: {transient_analysis['stationary_mean']:.3f}")
            print(f"Доля переходного периода: {transient_analysis['reduction_ratio']:.1%}")
            
            # Проверка гипотезы об уменьшении времени прогона
            reduction_possible = transient_analysis['reduction_ratio'] > 0.1
            print(f"Гипотеза об уменьшении времени прогона: {'ПОДТВЕРЖДЕНА' if reduction_possible else 'НЕ ПОДТВЕРЖДЕНА'}")
            
            # Визуализация переходного периода
            plt.figure(figsize=(12, 6))
            plt.plot(time_points[:len(time_series)], time_series, 'b-', alpha=0.7, label='Длина очереди')
            
            if transient_analysis['transient_period'] > 0 and transient_analysis['transient_period'] < len(time_points):
                transient_time = time_points[transient_analysis['transient_period']]
                plt.axvline(x=transient_time, color='r', linestyle='--', 
                           label=f'Конец переходного периода (t={transient_time:.0f})')
                plt.axhline(y=transient_analysis['stationary_mean'], color='g', linestyle='--',
                           label=f'Стационарное среднее')
            
            plt.xlabel('Модельное время')
            plt.ylabel('Длина очереди')
            plt.title('Анализ переходного периода в модельном времени')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('task4_transient_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    # Задача 5: Проверка возможности непрерывного прогона
    print("\n5. ПРОВЕРКА ВОЗМОЖНОСТИ НЕПРЕРЫВНОГО ПРОГОНА")
    print("-" * 60)
    
    # Запускаем один длинный прогон в непрерывном режиме
    print("Запуск длинного прогона в непрерывном режиме...")
    continuous_config = BASE_CONFIG.copy()
    continuous_config['max_time'] = 5000
    continuous_model = AutoRepairStationModel(continuous_config)
    continuous_result = continuous_model.run(continuous_mode=True)
    
    continuous_data = continuous_result.time_series_data['queue_lengths']
    feasibility_test = analyzer.test_continuous_run_feasibility(continuous_data)
    
    print(f"Стационарность системы: {'ДА' if feasibility_test['feasible'] else 'НЕТ'}")
    print(f"ANOVA p-value: {feasibility_test['p_value']:.4f}")
    print(f"Средние по сегментам: {[f'{m:.3f}' for m in feasibility_test['segment_means']]}")
    print(f"Непрерывный прогон: {'ВОЗМОЖЕН' if feasibility_test['feasible'] else 'НЕВОЗМОЖЕН'}")
    
    # Задача 6: Анализ чувствительности
    print("\n6. АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ К ВАРИАЦИЯМ ПАРАМЕТРОВ")
    print("-" * 60)
    
    # Вариации параметров (упрощенный вариант для скорости)
    parameters_variations = {
        'arrival_rate_+10%': {'arrival_rates': [8.8]},
        'arrival_rate_-10%': {'arrival_rates': [7.2]},
        'repair_time_+10%': {'repair_time_mean': 13.2},
        'repair_time_-10%': {'repair_time_mean': 10.8},
        'workstations_+1': {'num_workstations': 4},
        'workstations_-1': {'num_workstations': 2}
    }
    
    varied_results = {}
    base_wait_times = wait_times_20
    
    print("Анализ чувствительности:")
    for param_name, variation in parameters_variations.items():
        config = BASE_CONFIG.copy()
        config.update(variation)
        
        results = []
        for _ in range(5):  # Уменьшим количество прогонов для скорости
            model = AutoRepairStationModel(config)
            result = model.run()
            results.append(result.average_wait_time_type1)
        
        varied_results[param_name] = results
        avg_wait = np.mean(results)
        print(f"  {param_name}: среднее время ожидания = {avg_wait:.3f}")
    
    sensitivity_analysis = analyzer.sensitivity_analysis(base_wait_times, varied_results)
    
    print(f"\nКоэффициенты чувствительности:")
    for param, sensitivity in sensitivity_analysis['sensitivity_scores'].items():
        print(f"  {param}: {sensitivity:.3f}")
    
    print(f"\nКритические параметры (чувствительность > {sensitivity_analysis['sufficient_precision']:.1%}):")
    if sensitivity_analysis['critical_params']:
        for param, sensitivity in sensitivity_analysis['critical_params'].items():
            print(f"  {param}: {sensitivity:.3f}")
    else:
        print("  Нет критических параметров")
    
    # Визуализация чувствительности
    plt.figure(figsize=(12, 6))
    params = list(sensitivity_analysis['sensitivity_scores'].keys())
    sensitivities = list(sensitivity_analysis['sensitivity_scores'].values())
    
    bars = plt.bar(params, sensitivities, color=['red' if s > 0.01 else 'blue' for s in sensitivities])
    plt.axhline(y=0.01, color='r', linestyle='--', label='Порог значимости (1%)')
    plt.ylabel('Коэффициент чувствительности')
    plt.title('Анализ чувствительности времени ожидания к вариациям параметров')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    for bar, sensitivity in zip(bars, sensitivities):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{sensitivity:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('task6_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 80)
    print("АНАЛИЗ ЗАВЕРШЕН!")
    print("=" * 80)


run_comprehensive_analysis()