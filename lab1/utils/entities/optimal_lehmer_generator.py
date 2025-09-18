from .lehmer_generator import LehmerGenerator

class OptimalLehmerGenerator(LehmerGenerator):
    """
    Генератор с оптимальными параметрами для максимального периода
    """
    
    # Оптимальные параметры для различных типов генераторов
    OPTIMAL_PARAMS = {
        "multiplicative": {
            "a": 16807,         # 7^5 - Park & Miller
            "c": 0,             # Мультипликативный
            "m": 2**31 - 1,     # Простое число Мерсенна
            "period": 2**31 - 2
        },
        "mixed_numerical": {
            "a": 1664525,       # Numerical Recipes
            "c": 1013904223,    # Приращение
            "m": 2**32,         # Модуль степени 2
            "period": 2**32
        },
        "mixed_borland": {
            "a": 22695477,      # Borland C++
            "c": 1,             # Приращение
            "m": 2**32,         # Модуль степени 2
            "period": 2**32
        },
        "mixed_microsoft": {
            "a": 214013,        # Microsoft C
            "c": 2531011,       # Приращение
            "m": 2**32,         # Модуль степени 2
            "period": 2**32
        }
    }
     
    def __init__(self, seed: int = 1, generator_type: str = "multiplicative"):
        """
        Args:
            seed: начальное значение
            generator_type: тип генератора из OPTIMAL_PARAMS
        """
        if generator_type not in self.OPTIMAL_PARAMS:
            available_types = ", ".join(self.OPTIMAL_PARAMS.keys())
            raise ValueError(f"generator_type должен быть одним из: {available_types}")
        
        params = self.OPTIMAL_PARAMS[generator_type]
        super().__init__(
            seed=seed,
            a=params["a"],
            c=params["c"],
            m=params["m"]
        )
        
        self.generator_type = generator_type
        self.theoretical_period = params["period"]
    
    def get_theoretical_period(self) -> int:
        """Возвращает теоретический максимальный период для данных параметров"""
        return self.theoretical_period