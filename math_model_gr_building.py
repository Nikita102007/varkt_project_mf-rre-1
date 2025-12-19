import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from pictures.jpg import functions_tools
import pandas as pd
import pathlib

# Постоянные переменные
g = 9.81  # ускорение свободного падения (м/с²)
rho_0 = 1.225  # плотность воздуха на уровне моря (кг/м³)
G = 6.67430e-11  # Гравитационная постоянная
M_kerbin = 5.292e22  # Масса Кербина в кг
R_kerbin = 600000  # Радиус Кербина в метрах
C_d = 6.0  # 3.3  # Коэффициент аэродинамического сопротивления

# Масса и характеристики ступеней
stages = [
    {"wet_mass": 173392, "fuel_mass": 150500, "thrust": 4_872_000, "burn_time": 80, "ejection_force": 100, "area": 84},
    {"wet_mass": 120892, "fuel_mass": 35500, "thrust": 1_960_000, "burn_time": 110, "ejection_force": 250, "area": 78},
]


# Функция для расчета плотности воздуха в зависимости от высоты
def air_density(h):
    return rho_0 * np.exp(-h / 4500)


# Функция для расчета угла наклона в зависимости от высота
def calculate_pitch(altitude):
    if altitude < 70000:
        return 90 * (1 - altitude / 70000)  # Чем выше высота, тем меньше наклон
    return 0


# Функция для расчета гравитационного ускорения
def gravitational_acceleration(height):
    r = R_kerbin + height
    return G * M_kerbin / r ** 2


# Функция для системы уравнений
def rocket_equations(y, t, stage_index, initial_mass, fuel_mass, burn_time):
    # Распаковка состояния
    x_coord, horizontal_velocity, y_coord, vertical_velocity = y

    # Получаем данные по текущей ступени
    stage = stages[stage_index]
    thrust = stage["thrust"]
    ejection_force = stage["ejection_force"]
    area = stage["area"]

    # Расчет текущей массы с учетом израсходованного топлива
    if t <= burn_time:
        current_mass = initial_mass - (fuel_mass / burn_time) * t
    else:
        # После выгорания топлива масса постоянна
        current_mass = initial_mass - fuel_mass

    # Расчет скорости (модуль)
    velocity = np.sqrt(horizontal_velocity ** 2 + vertical_velocity ** 2)

    # Угол наклона
    pitch = calculate_pitch(y_coord)
    pitch_rad = np.radians(pitch)

    # Расчет гравитационного ускорения и сопротивления
    force_gravity = current_mass * gravitational_acceleration(y_coord)
    air_density_value = air_density(y_coord)
    drag_force = 0.5 * C_d * air_density_value * velocity ** 2 * area

    # Расчет ускорений
    radius = R_kerbin + y_coord
    centrifugal_force = (current_mass * horizontal_velocity ** 2) / radius

    # Суммарная сила тяги минус сопротивление
    net_force = thrust - drag_force

    # Ускорения с учетом угла наклона
    acceleration_vertical = (net_force * np.sin(pitch_rad) + centrifugal_force - force_gravity) / current_mass
    acceleration_horizontal = (net_force * np.cos(pitch_rad)) / current_mass

    # Обновление значений
    dxcoord = horizontal_velocity
    dhorizontal_velocity = acceleration_horizontal
    dycoord = vertical_velocity
    dvertical_velocity = acceleration_vertical

    return [dxcoord, dhorizontal_velocity, dycoord, dvertical_velocity]


# Функция для отделения ступени
def apply_ejection(y, stage_index):
    x_coord, horizontal_velocity, y_coord, vertical_velocity = y

    stage = stages[stage_index]
    ejection_force = stage["ejection_force"]

    # Расчет сухой массы ступени после выгорания топлива
    if stage_index == 0:
        # Для первой ступени: начальная масса минус топливо первой ступени
        dry_mass = stages[0]["wet_mass"] - stages[0]["fuel_mass"]
        # Вычитаем массу первой ступени и добавляем массу второй
        current_mass = dry_mass + stages[1]["wet_mass"]
    else:
        # Для второй ступени
        current_mass = stages[1]["wet_mass"] - stages[1]["fuel_mass"]

    # Угол наклона в момент отделения
    pitch = calculate_pitch(y_coord)
    pitch_rad = np.radians(pitch)

    # Применяем импульс от выброса
    dv = ejection_force / current_mass

    new_horizontal_velocity = horizontal_velocity + dv * np.cos(pitch_rad)
    new_vertical_velocity = vertical_velocity + dv * np.sin(pitch_rad)

    return [x_coord, new_horizontal_velocity, y_coord, new_vertical_velocity]


# Функция для интегрирования одной ступени с обработкой ошибок
def integrate_stage(initial_conditions, stage_index, initial_mass, fuel_mass, burn_time):
    try:
        # Используем метод для жестких систем с увеличенным количеством шагов
        time_eval = np.linspace(0, burn_time, max(2000, int(burn_time * 10)))

        result = odeint(
            rocket_equations,
            initial_conditions,
            time_eval,
            args=(stage_index, initial_mass, fuel_mass, burn_time),
            atol=1e-6,  # Абсолютная точность
            rtol=1e-6,  # Относительная точность
            mxstep=10000,  # Увеличенное максимальное количество шагов
            hmax=1.0,  # Максимальный шаг
            full_output=0
        )

        return time_eval, result

    except Exception as e:
        print(f"Ошибка при интегрировании ступени {stage_index}: {e}")
        # Возвращаем пустые массивы в случае ошибки
        return np.array([]), np.array([])


# Начальные условия
initial_conditions = [0, 0, 0, 0]  # x_coord, horizontal_velocity, y_coord, vertical_velocity

# Интегрирование первой ступени
print("Интегрирование первой ступени...")
time_first_stage, result_first_stage = integrate_stage(
    initial_conditions,
    0,
    stages[0]["wet_mass"],
    stages[0]["fuel_mass"],
    stages[0]["burn_time"]
)

if len(result_first_stage) > 0:
    print(f"Первая ступень: успешно, {len(result_first_stage)} точек")

    # Применяем выброс первой ступени
    state_after_first = result_first_stage[-1, :]
    state_before_second = apply_ejection(state_after_first, 0)

    # Интегрирование второй ступени
    print("Интегрирование второй ступени...")
    time_second_stage, result_second_stage = integrate_stage(
        state_before_second,
        1,
        stages[1]["wet_mass"],
        stages[1]["fuel_mass"],
        stages[1]["burn_time"]
    )

    if len(result_second_stage) > 0:
        print(f"Вторая ступень: успешно, {len(result_second_stage)} точек")

        # Применяем выброс второй ступени
        state_after_second = result_second_stage[-1, :]
        state_final = apply_ejection(state_after_second, 1)
        result_second_stage[-1, :] = state_final

        # Объединение результатов
        time = np.concatenate([time_first_stage, time_first_stage[-1] + time_second_stage])
        x_coords = np.concatenate([result_first_stage[:, 0], result_second_stage[:, 0]])
        x_velocities = np.concatenate([result_first_stage[:, 1], result_second_stage[:, 1]])
        y_coords = np.concatenate([result_first_stage[:, 2], result_second_stage[:, 2]])
        y_velocities = np.concatenate([result_first_stage[:, 3], result_second_stage[:, 3]])

        # Получение данных из симуляции KSP
        PATH = str(pathlib.Path(__file__).parent.joinpath("ksp_flight_data.csv"))
        data = pd.read_csv(PATH)

        time_ksp = data['Time']
        x_coords_ksp = data['Displacement']
        x_velocities_ksp = data['Horizontal Velocity']
        y_coords_ksp = data['Altitude']
        y_velocities_ksp = data['Vertical Velocity']

        # Синхронизация временных интервалов
        i = 0
        while i < len(time) and time_ksp[0] > time[i]:
            i += 1

        if i < len(time):
            time = time[i:]
            x_coords = x_coords[i:]
            x_velocities = x_velocities[i:]
            y_coords = y_coords[i:]
            y_velocities = y_velocities[i:]
        else:
            print("Внимание: Не удалось синхронизировать временные интервалы")


        # Утилиты для интерполяции
        def remap(v, x, y, a, b):
            return (v - x) / (y - x) * (b - a) + a


        def lerp(t, a, b):
            return a + (b - a) * t


        # Интерполяция данных KSP для сравнения
        time_remap = []
        x_coords_remap = []
        x_velocities_remap = []
        y_coords_remap = []
        y_velocities_remap = []
        y_velocities_remap_ = []

        idx_ksp = 0
        for idx in range(0, len(time) - 1):
            if idx_ksp >= len(time_ksp) - 1:
                break

            # Находим соответствующий интервал в данных KSP
            while idx_ksp < len(time_ksp) - 2 and time_ksp[idx_ksp + 1] < time[idx]:
                idx_ksp += 1

            if idx_ksp >= len(time_ksp) - 1:
                break

            # Интерполяция
            dt = remap(time[idx], time_ksp[idx_ksp], time_ksp[idx_ksp + 1], 0, 1)

            x_coord = lerp(dt, x_coords_ksp[idx_ksp], x_coords_ksp[idx_ksp + 1])
            x_velocity = lerp(dt, x_velocities_ksp[idx_ksp], x_velocities_ksp[idx_ksp + 1])
            y_coord = lerp(dt, y_coords_ksp[idx_ksp], y_coords_ksp[idx_ksp + 1])
            y_velocity = lerp(dt, y_velocities_ksp[idx_ksp], y_velocities_ksp[idx_ksp + 1])

            time_remap.append(time[idx])
            x_coords_remap.append(x_coord)
            x_velocities_remap.append(x_velocity)
            y_coords_remap.append(y_coord)
            y_velocities_remap.append(y_velocity)

        y_velocities_remap_.extend(functions_tools.apply_velocity_(y_velocities_remap))
        y_coords_remap_ = functions_tools.apply_height_(y_coords_remap)
        x_coords_remap_ = functions_tools.apply_horizontal_(x_coords_remap)
        x_velocities_remap_ = functions_tools.apply_velocity_x(x_velocities_remap)


        # Вычисление погрешностей
        def abs_error(values):
            return abs(values[1] - values[0])


        def rel_error(values):
            if abs(values[0]) < 1e-6:  # Избегаем деления на ноль
                return 0
            return abs(values[1] - values[0]) / abs(values[0]) * 100


        # Расчет погрешностей
        y_velocities_abs_error = list(map(abs_error, zip(y_velocities_remap_, y_velocities_remap)))
        y_coords_abs_error = list(map(abs_error, zip(y_coords_remap_, y_coords_remap)))
        x_velocities_abs_error = list(map(abs_error, zip(x_velocities_remap_, x_velocities_remap)))
        x_coords_abs_error = list(map(abs_error, zip(x_coords_remap_, x_coords_remap)))

        y_velocities_rel_error = list(map(rel_error, zip(y_velocities_remap_, y_velocities_remap)))
        y_coords_rel_error = list(map(rel_error, zip(y_coords_remap_, y_coords_remap)))
        x_velocities_rel_error = list(map(rel_error, zip(x_velocities_remap_, x_velocities_remap)))
        x_coords_rel_error = list(map(rel_error, zip(x_coords_remap_, x_coords_remap)))

        # Построение графиков с новой компоновкой
        fig = plt.figure(figsize=(16, 10))

        # Создаем сетку 2x2 для графиков справа
        gs = fig.add_gridspec(2, 2, wspace=0.3, hspace=0.4, left=0.4, right=0.95)

        # График высоты (вверху слева)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(time_remap, y_coords_abs_error, label='Абс. погрешность', color='red', alpha=0.7)
        ax1.plot(time_remap, y_coords_remap_, label='Модель', color='blue')
        ax1.plot(time_remap, y_coords_remap, label='KSP', color='orange', linestyle='--')

        # Добавляем вертикальную линию на 80 секундах (отделение ступени)
        ax1.axvline(x=80, color='green', linestyle='-', linewidth=2, alpha=0.5, label='Отделение 1-й ступени (80 с)')

        ax1.set_title('Высота от времени')
        ax1.set_xlabel('Время (с)')
        ax1.set_ylabel('Высота (м)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # График вертикальной скорости (вверху справа)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(time_remap, y_velocities_abs_error, label='Абс. погрешность', color='red', alpha=0.7)
        ax2.plot(time_remap, y_velocities_remap_, label='Модель', color='blue')
        ax2.plot(time_remap, y_velocities_remap, label='KSP', color='orange', linestyle='--')

        # Добавляем вертикальную линию на 80 секундах (отделение ступени)
        ax2.axvline(x=80, color='green', linestyle='-', linewidth=2, alpha=0.5)

        ax2.set_title('Вертикальная скорость от времени')
        ax2.set_xlabel('Время (с)')
        ax2.set_ylabel('Скорость (м/с)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # График смещения (внизу слева)
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(time_remap, x_coords_abs_error, label='Абс. погрешность', color='red', alpha=0.7)
        ax3.plot(time_remap, x_coords_remap_, label='Модель', color='blue')
        ax3.plot(time_remap, x_coords_remap, label='KSP', color='orange', linestyle='--')

        # Добавляем вертикальную линию на 80 секундах (отделение ступени)
        ax3.axvline(x=80, color='green', linestyle='-', linewidth=2, alpha=0.5)

        ax3.set_title('Горизонтальное смещение')
        ax3.set_xlabel('Время (с)')
        ax3.set_ylabel('Смещение (м)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # График горизонтальной скорости (внизу справа)
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(time_remap, x_velocities_abs_error, label='Абс. погрешность', color='red', alpha=0.7)
        ax4.plot(time_remap, x_velocities_remap_, label='Модель', color='blue')
        ax4.plot(time_remap, x_velocities_remap, label='KSP', color='orange', linestyle='--')

        # Добавляем вертикальную линию на 80 секундах (отделение ступени)
        ax4.axvline(x=80, color='green', linestyle='-', linewidth=2, alpha=0.5)

        ax4.set_title('Горизонтальная скорость')
        ax4.set_xlabel('Время (с)')
        ax4.set_ylabel('Скорость (м/с)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Информация о полете (слева)
        ax_text = fig.add_axes([0.05, 0.1, 0.3, 0.8])
        ax_text.axis('off')
        flight_info = f"""
        КОМАНДА BIG CUCUMBER COMPANY

        ИНФОРМАЦИЯ О ПОЛЕТЕ MF-RPE-1
        {'=' * 30}

        ПЕРВАЯ СТУПЕНЬ:
            Время работы: {stages[0]['burn_time']} с
            Тяга: {stages[0]['thrust'] / 1000:.0f} кН
            Масса топлива: {stages[0]['fuel_mass']:,} кг
            Площадь сечения: {stages[0]['area']} м²

        ВТОРАЯ СТУПЕНЬ:
            Время работы: {stages[1]['burn_time']} с
            Тяга: {stages[1]['thrust'] / 1000:.0f} кН
            Масса топлива: {stages[1]['fuel_mass']:,} кг
            Площадь сечения: {stages[1]['area']} м²

        {'=' * 30}

        РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ:

        Максимальная высота: {max(y_coords_remap_) / 1000:.1f} км
        Максимальная скорость: {max([(x ** 2 + y ** 2) ** 0.5 for x in x_velocities_remap_ for y in y_velocities_remap_]):.1f} м/с
        Горизонтальная скорость: {x_velocities_remap_[-1]:.1f} м/с
        Вертикальная скорость: {y_velocities_remap_[-1]:.1f} м/с
        Общее время: {time_remap[-1] + 0.1:.1f} с

        {'=' * 30}
        """
        ax_text.text(0.05, 0.95, flight_info, fontsize=11, fontfamily='monospace',
                     verticalalignment='top', horizontalalignment='left',
                     transform=ax_text.transAxes, linespacing=1.5)

        plt.savefig("final_corrected.png", dpi=150, bbox_inches='tight')
        plt.show()

        # Построение графика относительных погрешностей
        plt.figure(figsize=(12, 6))

        plt.plot(time_remap, y_coords_rel_error, label='Высота', color='blue', linewidth=2)
        plt.plot(time_remap, y_velocities_rel_error, label='Вертикальная скорость', color='orange', linewidth=2)
        plt.plot(time_remap, x_coords_rel_error, label='Горизонтальное смещение', color='green', linewidth=2)
        plt.plot(time_remap, x_velocities_rel_error, label='Горизонтальная скорость', color='red', linewidth=2)

        plt.axvline(x=80, color='purple', linestyle='--', linewidth=2, alpha=0.7, label='Отделение 1-й ступени (80 с)')

        plt.title('Относительные погрешности (%)')
        plt.xlabel('Время (с)')
        plt.ylabel('Погрешность (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("relative_errors.png", dpi=150, bbox_inches='tight')
        plt.show()

        # Вывод статистики
        print("\nСтатистика моделирования:")
        print(f"  Максимальная высота: {max(y_coords) / 1000:.2f} км")
        print(f"  Максимальная горизонтальная скорость: {max(x_velocities):.1f} м/с")
        print(f"  Максимальная вертикальная скорость: {max(y_velocities):.1f} м/с")
        print(f"  Общее время моделирования: {time[-1]:.1f} с")
        print(f"  Момент отделения первой ступени: 80 с")

    else:
        print("Ошибка: Не удалось интегрировать вторую ступень")
else:
    print("Ошибка: Не удалось интегрировать первую ступень")