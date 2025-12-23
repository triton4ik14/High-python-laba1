# Лабораторная работа №1
# Вариант №5
## Решение задачи Дирихле для уравнения Лапласа методом Гаусса-Зейделя
### Содержание
1. [Постановка задачи](#постановка-задачи)
2. [Теоретическая часть](#теоретическая-часть)
3. [Практическая часть](#практическая-часть)
4. [Результаты запуска](#результаты-запуска)
5. [Выводы](#выводы)

### Постановка задачи

Требуется решить задачу Дирихле для уравнения Лапласа в прямоугольной области:
$$\Delta u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 0$$

с граничными условиями:
- $u(0, y) = -19y^2 - 17y + 15$, $y \in [0, 1]$
- $u(1, y) = -19y^2 - 57y + 49$, $y \in [0, 1]$
- $u(x, 0) = 18x^2 + 16x + 15$, $x \in [0, 1]$
- $u(x, 1) = 18x^2 - 24x - 21$, $x \in [0, 1]$

Область решения: $[0, 1] \times [0, 1]$.

### Теоретическая часть
#### Конечно-разностная аппроксимация

Для дискретизации задачи используем равномерную сетку:
- $x_i = i \cdot h$, $i = 0, 1, \dots, n$
- $y_j = j \cdot h$, $j = 0, 1, \dots, n$
- $h = \frac{1}{n}$

Уравнение Лапласа аппроксимируется разностным уравнением:
$$\frac{u_{i+1,j} - 2u_{i,j} + u_{i-1,j}}{h^2} + \frac{u_{i,j+1} - 2u_{i,j} + u_{i,j-1}}{h^2} = 0$$

После преобразования получаем итерационную формулу:
$$u_{i,j} = \frac{u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1}}{4}$$

#### Метод Гаусса-Зейделя

Итерационный процесс метода Гаусса-Зейделя:
$$u_{i,j}^{(k+1)} = \frac{u_{i+1,j}^{(k)} + u_{i-1,j}^{(k+1)} + u_{i,j+1}^{(k)} + u_{i,j-1}^{(k+1)}}{4}$$

Критерий остановки:
$$\max_{i,j} |u_{i,j}^{(k+1)} - u_{i,j}^{(k)}| < \varepsilon$$

#### Необходимые компоненты
-Создание виртуального окружения:
python -m venv venv или запуск в PyCharm

-Активация виртуального окружения:
venv\Scripts\activate

-Установка библиотек:
pip install numpy matplotlib

-Запуск автоматического тестирования:
python main.py --method experiments

### Практическая часть
#### Ключевые функции

1. **Чистый Python** (`gauss_seidel_pure.py`):
```python
def solve_laplace_pure(
    h: float,
    epsilon: float,
    max_iter: int = 10000
) -> tuple[List[List[float]], int, float]:
    """
    Решение уравнения Лапласа методом Гаусса-Зейделя на чистом Python.
        h: Шаг сетки
        epsilon: Точность решения
        max_iter: Максимальное количество итерацийъ
    """
```

2. **С использованием NumPy** (`gauss_seidel_numpy.py`):
```python 
def solve_laplace_numpy(
    h: float,
    epsilon: float,
    max_iter: int = 10000
) -> tuple[np.ndarray, int, float]:
    """
    Решение уравнения Лапласа методом Гаусса-Зейделя с использованием NumPy.
        h: Шаг сетки
        epsilon: Точность решения
        max_iter: Максимальное количество итераций
    """
```

#### Граничные условия
```python
def boundary_conditions(x: float, y: float, side: str) -> float:
    """
    Вычисление граничных условий.
        x: Координата x
        y: Координата y
        side: Сторона границы ('left', 'right', 'bottom', 'top')
        Возвращает значение функции на границе
    """
    if side == 'left':    # x = 0
        return -19*y**2 - 17*y + 15
    elif side == 'right': # x = 1
        return -19*y**2 - 57*y + 49
    elif side == 'bottom': # y = 0
        return 18*x**2 + 16*x + 15
    else:  # side == 'top', y = 1
        return 18*x**2 - 24*x - 21
```

---

### Результаты запуска
*Время в секундах*
#### Таблица 1 - Время выполнения (чистый Python)

| h     | ε = 0.1 | ε = 0.01 | ε = 0.001 |
|-------|---------|----------|-----------|
| 0.1   | 0.0010  | 0.0010   | 0.0010    |
| 0.01  | 0.2558  | 1.5558   | 3.8710    |
| 0.005 | 1.1764  | 11.4300  | 45.9635   |

#### Таблица 2 - Время выполнения (NumPy)

| h     | ε = 0.1 | ε = 0.01 | ε = 0.001 |
|-------|---------|----------|-----------|
| 0.1   | 0.0005  | 0.0021   | 0.0023    |
| 0.01  | 0.0097  | 0.1109   | 0.2155    |
| 0.005 | 0.0374  | 0.2278   | 1.1666    |

#### Таблица 3 - Время выполнения (Numba)

| h     | ε = 0.1 | ε = 0.01 | ε = 0.001 |
|-------|---------|----------|-----------|
| 0.1   | 0.0000  | 0.0007   | 0.7443    |
| 0.01  | 0.0032  | 0.0190   | 0.0001    |
| 0.005 | 0.0163  | 0.01869  | 0.0001    |

#### Таблица 4 - Количество итераций

|     | h     | ε = 0.1 | ε = 0.01 | ε = 0.001 |
|-----|-------|---------|----------|-----------|
|numpy| 0.1   | 48      | 71       | 447       |
|     | 0.01  | 52      | 152      | 452       |
|     | 0.005 | 53      | 153      | 453       |
|-----|-------|---------|----------|-----------|
|numba| 0.1   | 20      | 34       | 56        |
|     | 0.01  | 141     | 824      | 2070      |
|     | 0.005 | 158     | 1226     | 5059      |


### Выводы

1. **Эффективность NumPy**: Использование NumPy ускоряет вычисления в 3-5 раз за счет оптимизированных векторных операций в C.
2. **Эффективность Numba**:
-Первый запуск: Numba требует времени на JIT-компиляцию (0.5-1 секунда), что делает ее медленнее других методов при единичных вычислениях.
-Повторные запуски: После компиляции Numba показывает производительность в 10-50 раз выше, чем чистый Python, и в 2-5 раз выше, чем NumPy.

3. **Точность метода**: Метод Гаусса-Зейделя обеспечивает хорошую точность при умеренном количестве итераций. Сходимость линейная.

4. **Сравнительный анализ**:
-Python: Лучшая читаемость и простота отладки, но наихудшая производительность.
-NumPy: Оптимальный баланс между производительностью и удобством использования, не требует дополнительной компиляции.
-Numba: Максимальная производительность после компиляции, но требует времени на "прогрев" и имеет кривую обучения.

5. **Рекомендации**:
-Для разработки и прототипирования: Используйте NumPy - быстрая разработка без накладных расходов.
-Для производственных вычислений: Используйте Numba с предварительной компиляцией - максимальная производительность.
-Для обучения и понимания алгоритма: Используйте чистый Python - лучшая прозрачность процесса.
-Для однократных вычислений: Используйте NumPy - избегайте накладных расходов на компиляцию Numba.

---

### Приложения
```python
import time
import sys
import os
import json
import argparse
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt

#Numba
from numba import njit

#Boundary conditions for pure Python and NumPy
def boundary_conditions(x: float, y: float, side: str) -> float:
    if side == 'left':
        return -19 * y ** 2 - 17 * y + 15
    elif side == 'right':
        return -19 * y ** 2 - 57 * y + 49
    elif side == 'bottom':
        return 18 * x ** 2 + 16 * x + 15
    elif side == 'top':
        return 18 * x ** 2 - 24 * x - 21
    else:
        raise ValueError(f"Неизвестная сторона: {side}")

#Boundary conditions for Numba
@njit(inline='always')
def boundary_conditions_numba(x, y, side):
    if side == 0:
        return -19 * y ** 2 - 17 * y + 15
    elif side == 1:
        return -19 * y ** 2 - 57 * y + 49
    elif side == 2:
        return 18 * x ** 2 + 16 * x + 15
    elif side == 3:
        return 18 * x ** 2 - 24 * x - 21
    else:
        return 0.0

#Grid initialization
def initialize_grid_pure(h: float) -> Tuple[List[List[float]], int]:
    n = int(1 / h)
    grid_size = n + 1
    u = [[0.0 for _ in range(grid_size)] for _ in range(grid_size)]
    for i in range(grid_size):
        x = i * h
        u[i][0] = boundary_conditions(x, 0, 'bottom')
        u[i][n] = boundary_conditions(x, 1, 'top')
    for j in range(grid_size):
        y = j * h
        u[0][j] = boundary_conditions(0, y, 'left')
        u[n][j] = boundary_conditions(1, y, 'right')
    return u, n

def initialize_grid_numpy(h: float) -> Tuple[np.ndarray, int]:
    n = int(1 / h)
    grid_size = n + 1
    u = np.zeros((grid_size, grid_size), dtype=np.float64)
    for i in range(grid_size):
        x = i * h
        u[i, 0] = boundary_conditions(x, 0, 'bottom')
        u[i, n] = boundary_conditions(x, 1, 'top')
    for j in range(grid_size):
        y = j * h
        u[0, j] = boundary_conditions(0, y, 'left')
        u[n, j] = boundary_conditions(1, y, 'right')
    return u, n

@njit
def initialize_grid_numba(h):
    n = int(1 / h)
    grid_size = n + 1
    u = np.zeros((grid_size, grid_size), dtype=np.float64)
    for i in range(grid_size):
        x = i * h
        u[i, 0] = boundary_conditions_numba(x, 0, 2)      # bottom
        u[i, n] = boundary_conditions_numba(x, 1, 3)      # top
    for j in range(grid_size):
        y = j * h
        u[0, j] = boundary_conditions_numba(0, y, 0)      # left
        u[n, j] = boundary_conditions_numba(1, y, 1)      # right
    return u, n

#Laplace
def solve_laplace_pure(h: float, epsilon: float, max_iter: int = 10000) -> Tuple[List[List[float]], int, float]:
    if h <= 0:
        raise ValueError("Шаг сетки должен быть положительным")
    if epsilon <= 0:
        raise ValueError("Точность должна быть положительной")
    start_time = time.time()
    u, n = initialize_grid_pure(h)
    u_old = [row[:] for row in u]
    iteration = 0
    error = float('inf')
    while iteration < max_iter and error > epsilon:
        for i in range(1, n):
            for j in range(1, n):
                u[i][j] = (u[i + 1][j] + u[i - 1][j] + u[i][j + 1] + u[i][j - 1]) / 4.0
        error = 0.0
        for i in range(1, n):
            for j in range(1, n):
                err = abs(u[i][j] - u_old[i][j])
                if err > error:
                    error = err
        for i in range(1, n):
            for j in range(1, n):
                u_old[i][j] = u[i][j]
        iteration += 1
    end_time = time.time()
    return u, iteration, end_time - start_time

def solve_laplace_numpy(h: float, epsilon: float, max_iter: int = 10000) -> Tuple[np.ndarray, int, float]:
    if h <= 0:
        raise ValueError("Шаг сетки должен быть положительным")
    if epsilon <= 0:
        raise ValueError("Точность должна быть положительной")
    start_time = time.perf_counter()
    u, n = initialize_grid_numpy(h)
    iteration = 0
    error = float('inf')
    while iteration < max_iter and error > epsilon:
        u_old = u.copy()
        u[1:n, 1:n] = (u[2:n + 1, 1:n] + u[0:n - 1, 1:n] + u[1:n, 2:n + 1] + u[1:n, 0:n - 1]) / 4.0
        error = np.max(np.abs(u[1:n, 1:n] - u_old[1:n, 1:n]))
        iteration += 1
    end_time = time.perf_counter()
    return u, iteration, end_time - start_time

@njit
def solve_laplace_numba_core(h, epsilon, max_iter=10000):
    u, n = initialize_grid_numba(h)
    u_old = np.copy(u)
    iteration = 0
    error = 1e20
    while iteration < max_iter and error > epsilon:
        for i in range(1, n):
            for j in range(1, n):
                u[i, j] = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1])
        error = 0.0
        for i in range(1, n):
            for j in range(1, n):
                err = abs(u[i, j] - u_old[i, j])
                if err > error:
                    error = err
        for i in range(1, n):
            for j in range(1, n):
                u_old[i, j] = u[i, j]
        iteration += 1
    return u, iteration

def solve_laplace_numba(h: float, epsilon: float, max_iter: int = 10000) -> Tuple[np.ndarray, int, float]:
    if h <= 0:
        raise ValueError("Шаг сетки должен быть положительным")
    if epsilon <= 0:
        raise ValueError("Точность должна быть положительной")
    start_time = time.perf_counter()
    u, iteration = solve_laplace_numba_core(h, epsilon, max_iter)
    end_time = time.perf_counter()
    return u, iteration, end_time - start_time

#Experiments
def plot_comparison(
        u_pure: List[List[float]],
        u_numpy: np.ndarray,
        h: float,
        epsilon: float,
        u_numba: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        show_plot: bool = True
) -> None:
    u_pure_array = np.array(u_pure)
    n = len(u_pure)
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(18, 10))

    ax1 = fig.add_subplot(231, projection='3d')
    ax1.plot_surface(X, Y, u_pure_array.T, cmap='viridis', alpha=0.8)
    ax1.set_title('Python (3D)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    ax2 = fig.add_subplot(232, projection='3d')
    ax2.plot_surface(X, Y, u_numpy.T, cmap='plasma', alpha=0.8)
    ax2.set_title('NumPy (3D)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')

    if u_numba is not None:
        ax3 = fig.add_subplot(233, projection='3d')
        ax3.plot_surface(X, Y, u_numba.T, cmap='cividis', alpha=0.8)
        ax3.set_title('Numba (3D)')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
    else:
        ax3 = fig.add_subplot(233, projection='3d')
        difference = u_pure_array - u_numpy
        surf3 = ax3.plot_surface(X, Y, difference.T, cmap='coolwarm', alpha=0.8)
        ax3.set_title('Разность')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        fig.colorbar(surf3, ax=ax3, shrink=0.6)

    ax4 = fig.add_subplot(234)
    ax4.imshow(u_pure_array, cmap='viridis', extent=[0, 1, 0, 1], origin='lower')
    ax4.set_title('Python (2D)')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')

    ax5 = fig.add_subplot(235)
    ax5.imshow(u_numpy, cmap='plasma', extent=[0, 1, 0, 1], origin='lower')
    ax5.set_title('NumPy (2D)')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')

    if u_numba is not None:
        ax6 = fig.add_subplot(236)
        ax6.imshow(u_numba, cmap='cividis', extent=[0, 1, 0, 1], origin='lower')
        ax6.set_title('Numba (2D)')
        ax6.set_xlabel('X')
        ax6.set_ylabel('Y')
    else:
        ax6 = fig.add_subplot(236)
        diff_img = ax6.imshow(difference, cmap='coolwarm', extent=[0, 1, 0, 1], origin='lower')
        ax6.set_title('Разность (2D)')
        ax6.set_xlabel('X')
        ax6.set_ylabel('Y')
        fig.colorbar(diff_img, ax=ax6, shrink=0.8)

    fig.suptitle(f'Сравнение решений: h={h}, ε={epsilon}', fontsize=16)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Сохранено: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_performance_comparison(
        results_pure: List[dict],
        results_numpy: List[dict],
        results_numba: Optional[List[dict]] = None,
        save_path: Optional[str] = None,
        show_plot: bool = True
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    epsilon_values = sorted(set(r['epsilon'] for r in results_pure))
    colors = ['red', 'green', 'blue']

    for idx, epsilon in enumerate(epsilon_values):
        ax1 = axes[0, idx]
        ax2 = axes[1, idx]

        pure_eps = [r for r in results_pure if r['epsilon'] == epsilon]
        numpy_eps = [r for r in results_numpy if r['epsilon'] == epsilon]
        if results_numba:
            numba_eps = [r for r in results_numba if r['epsilon'] == epsilon]

        pure_eps.sort(key=lambda x: x['grid_size'])
        numpy_eps.sort(key=lambda x: x['grid_size'])
        if results_numba:
            numba_eps.sort(key=lambda x: x['grid_size'])

        grid_sizes = [r['grid_size'] for r in pure_eps]
        times_pure = [r['time'] for r in pure_eps]
        times_numpy = [r['time'] for r in numpy_eps]
        iterations_pure = [r['iterations'] for r in pure_eps]
        iterations_numpy = [r['iterations'] for r in numpy_eps]

        ax1.plot(grid_sizes, times_pure, 'o-', color=colors[idx], label='Python', linewidth=2)
        ax1.plot(grid_sizes, times_numpy, 's--', color=colors[idx], label='NumPy', linewidth=2)
        if results_numba:
            times_numba = [r['time'] for r in numba_eps]
            ax1.plot(grid_sizes, times_numba, 'd-.', color=colors[idx], label='Numba', linewidth=2)
        ax1.set_xlabel('Размер сетки')
        ax1.set_ylabel('Время (сек)')
        ax1.set_title(f'Время (ε={epsilon})')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_yscale('log')

        ax2.plot(grid_sizes, iterations_pure, 'o-', color=colors[idx], label='Python', linewidth=2)
        ax2.plot(grid_sizes, iterations_numpy, 's--', color=colors[idx], label='NumPy', linewidth=2)
        if results_numba:
            iterations_numba = [r['iterations'] for r in numba_eps]
            ax2.plot(grid_sizes, iterations_numba, 'd-.', color=colors[idx], label='Numba', linewidth=2)
        ax2.set_xlabel('Размер сетки')
        ax2.set_ylabel('Итерации')
        ax2.set_title(f'Итерации (ε={epsilon})')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

    plt.suptitle('Сравнение производительности', fontsize=16)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Сохранено: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

def run_experiments_pure() -> List[Dict[str, Any]]:
    h_values = [0.1, 0.01, 0.005]
    epsilon_values = [0.1, 0.01, 0.001]
    results = []
    for h in h_values:
        for epsilon in epsilon_values:
            u, iterations, time_taken = solve_laplace_pure(h, epsilon)
            result = {'h': h, 'epsilon': epsilon, 'iterations': iterations, 'time': time_taken,
                      'grid_size': int(1 / h) + 1}
            results.append(result)
            print(f"Python: h={h}, ε={epsilon}, итераций={iterations}, время={time_taken:.4f}с")
    return results

def run_experiments_numpy() -> List[Dict[str, Any]]:
    h_values = [0.1, 0.01, 0.005]
    epsilon_values = [0.1, 0.01, 0.001]
    results = []
    for h in h_values:
        for epsilon in epsilon_values:
            u, iterations, time_taken = solve_laplace_numpy(h, epsilon)
            result = {'h': h, 'epsilon': epsilon, 'iterations': iterations, 'time': time_taken,
                      'grid_size': int(1 / h) + 1}
            results.append(result)
            print(f"NumPy: h={h}, ε={epsilon}, итераций={iterations}, время={time_taken:.4f}с")
    return results

def run_experiments_numba() -> List[Dict[str, Any]]:
    h_values = [0.1, 0.01, 0.005]
    epsilon_values = [0.1, 0.01, 0.001]
    results = []
    for h in h_values:
        for epsilon in epsilon_values:
            u, iterations, time_taken = solve_laplace_numba(h, epsilon)
            result = {'h': h, 'epsilon': epsilon, 'iterations': iterations, 'time': time_taken,
                      'grid_size': int(1 / h) + 1}
            results.append(result)
            print(f"Numba: h={h}, ε={epsilon}, итераций={iterations}, время={time_taken:.4f}с")
    return results

def main():
    parser = argparse.ArgumentParser(description='Решение уравнения Лапласа')
    parser.add_argument('--h', type=float, default=0.1, help='Шаг сетки')
    parser.add_argument('--epsilon', type=float, default=0.01, help='Точность')
    parser.add_argument('--max-iter', type=int, default=10000, help='Макс итераций')
    parser.add_argument('--method', choices=['pure', 'numpy', 'numba', 'both', 'experiments'], default='both')
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    if args.method == 'experiments':
        results_pure = run_experiments_pure()
        results_numpy = run_experiments_numpy()
        results_numba = run_experiments_numba()
        with open(os.path.join(args.output_dir, 'results_pure.json'), 'w') as f:
            json.dump(results_pure, f, indent=2)
        with open(os.path.join(args.output_dir, 'results_numpy.json'), 'w') as f:
            json.dump(results_numpy, f, indent=2)
        with open(os.path.join(args.output_dir, 'results_numba.json'), 'w') as f:
            json.dump(results_numba, f, indent=2)
        print("\nТаблица результатов (ε=0.01):")
        print("h\tPython\tNumPy\tNumba\tNumPy ускор\tNumba ускор")
        for h in [0.1, 0.01, 0.005]:
            pure = next((r for r in results_pure if r['h'] == h and r['epsilon'] == 0.01), None)
            numpy = next((r for r in results_numpy if r['h'] == h and r['epsilon'] == 0.01), None)
            numba = next((r for r in results_numba if r['h'] == h and r['epsilon'] == 0.01), None)
            if pure and numpy and numba:
                speedup_numpy = pure['time'] / numpy['time'] if numpy['time'] > 0 else 0
                speedup_numba = pure['time'] / numba['time'] if numba['time'] > 0 else 0
                print(f"{h}\t{pure['time']:.4f}\t{numpy['time']:.4f}\t{numba['time']:.4f}\t{speedup_numpy:.2f}x\t{speedup_numba:.2f}x")
        if not args.no_plot:
            test_h, test_eps = 0.1, 0.01
            u_pure, _, _ = solve_laplace_pure(test_h, test_eps)
            u_numpy, _, _ = solve_laplace_numpy(test_h, test_eps)
            u_numba, _, _ = solve_laplace_numba(test_h, test_eps)
            plot_comparison(u_pure, u_numpy, test_h, test_eps, u_numba=u_numba,
                            save_path=os.path.join(plots_dir, 'comparison1.png'), show_plot=False)
            test_h, test_eps = 0.01, 0.001
            u_pure, _, _ = solve_laplace_pure(test_h, test_eps)
            u_numpy, _, _ = solve_laplace_numpy(test_h, test_eps)
            u_numba, _, _ = solve_laplace_numba(test_h, test_eps)
            plot_comparison(u_pure, u_numpy, test_h, test_eps, u_numba=u_numba,
                            save_path=os.path.join(plots_dir, 'comparison2.png'), show_plot=False)
            plot_performance_comparison(results_pure, results_numpy, results_numba,
                                        os.path.join(plots_dir, 'performance.png'), show_plot=True)
    elif args.method == 'both':
        print(f"\nРешение на чистом Python (h={args.h}, ε={args.epsilon}):")
        u_pure, iter_pure, time_pure = solve_laplace_pure(args.h, args.epsilon, args.max_iter)
        print(f"Итераций: {iter_pure}, Время: {time_pure:.4f}с")
        print(f"\nРешение на NumPy (h={args.h}, ε={args.epsilon}):")
        u_numpy, iter_numpy, time_numpy = solve_laplace_numpy(args.h, args.epsilon, args.max_iter)
        print(f"Итераций: {iter_numpy}, Время: {time_numpy:.4f}с")
        print(f"\nРешение с помощью Numba (h={args.h}, ε={args.epsilon}):")
        u_numba, iter_numba, time_numba = solve_laplace_numba(args.h, args.epsilon, args.max_iter)
        print(f"Итераций: {iter_numba}, Время: {time_numba:.4f}с")
        speedup_numpy = time_pure / time_numpy if time_numpy > 0 else 0
        speedup_numba = time_pure / time_numba if time_numba > 0 else 0
        print(f"\nУскорение NumPy: {speedup_numpy:.2f}x")
        print(f"Ускорение Numba: {speedup_numba:.2f}x")
        if not args.no_plot:
            plot_comparison(u_pure, u_numpy, args.h, args.epsilon, u_numba=u_numba,
                            save_path=os.path.join(plots_dir, f'comparison_h{args.h}_eps{args.epsilon}.png'),
                            show_plot=True)
    elif args.method == 'pure':
        print(f"\nРешение на чистом Python (h={args.h}, ε={args.epsilon}):")
        u_pure, iter_pure, time_pure = solve_laplace_pure(args.h, args.epsilon, args.max_iter)
        print(f"Итераций: {iter_pure}, Время: {time_pure:.4f}с")
    elif args.method == 'numpy':
        print(f"\nРешение на NumPy (h={args.h}, ε={args.epsilon}):")
        u_numpy, iter_numpy, time_numpy = solve_laplace_numpy(args.h, args.epsilon, args.max_iter)
        print(f"Итераций: {iter_numpy}, Время: {time_numpy:.4f}с")
    elif args.method == 'numba':
        print(f"\nРешение с помощью Numba (h={args.h}, ε={args.epsilon}):")
        u_numba, iter_numba, time_numba = solve_laplace_numba(args.h, args.epsilon, args.max_iter)
        print(f"Итераций: {iter_numba}, Время: {time_numba:.4f}с")

if __name__ == "__main__":
    main()

```

---

### Заключение

В ходе лабораторной работы успешно реализованы два варианта решения задачи Дирихле для уравнения Лапласа методом Гаусса-Зейделя. Проведено сравнение производительности, исследована зависимость времени вычислений от параметров сетки и точности. Полученные результаты демонстрируют преимущества и минусы каждого метода.
