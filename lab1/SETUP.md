# Инструкция по запуску лабораторной работы

## 1. Создание виртуального окружения

```bash
# Переходим в папку проекта
cd /Users/mashkakoser/Documents/саиммод/lab1

# Создаем виртуальное окружение
python3 -m venv venv

# Активируем окружение
source venv/bin/activate
```

## 2. Установка библиотек

```bash
# Устанавливаем все необходимые библиотеки
pip install numpy matplotlib scipy jupyter

# Или используем файл requirements.txt
pip install -r requirements.txt
```

## 3. Запуск Jupyter Notebook

```bash
# Запускаем Jupyter
jupyter notebook

# Или альтернативно
jupyter lab
```

## 4. Открытие notebook

В браузере откроется Jupyter, выберите файл `lab1_analysis.ipynb`

## 5. Запуск анализа

Выполните ячейки по порядку:
1. Импорты и настройки
2. Подбор оптимальных параметров  
3. Генерация выборки и гистограмма
4. Критерий хи-квадрат
5. Критерий Колмогорова-Смирнова
6. Графическая визуализация
7. Анализ согласованности

## Возможные проблемы

**Если ошибка импорта:**
```bash
# Убедитесь что вы в папке lab1
pwd
# Должно показать: /Users/mashkakoser/Documents/саиммод/lab1
```

**Если не работает виртуальное окружение:**
```bash
# Деактивируем текущее
deactivate

# Удаляем старое
rm -rf venv

# Создаем новое
python3 -m venv venv
source venv/bin/activate
pip install numpy matplotlib scipy jupyter
```
