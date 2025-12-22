import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
import pathlib

# Постоянные переменные
rho_0 = 1.13257  # плотность воздуха на уровне моря (кг/м³)
G = 6.67430e-11  # Гравитационная постоянная
M_kerbin = 5.292e22  # Масса Кербина в кг
R_kerbin = 600000  # Радиус Кербина в метрах
C_d = 6.35  # Коэффициент аэродинамического сопротивления (вынесен в константу)

# Масса и характеристики ступеней
stages = [
    {"wet_mass": 173392, "fuel_mass": 150500, "thrust": 4_872_000, "burn_time": 80, "ejection_force": 100, "area": 84},
    {"wet_mass": 120892, "fuel_mass": 35500, "thrust": 1_960_000, "burn_time": 110, "ejection_force": 250, "area": 78},
]


# Функция для расчета реальной высоты над сферической поверхностью
def calculate_real_height(x, y):
    # Расстояние от центра Кербина
    r = np.sqrt(x ** 2 + (R_kerbin + y) ** 2)
    # Высота над поверхностью
    h = r - R_kerbin
    return h


# Функция для расчета плотности воздуха в зависимости от высота
def air_density(h):
    return rho_0 * np.exp(-h / 4500)


# Функция для расчета угла тангажа (между тягой и локальным горизонтом) в зависимости от высоты
def calculate_pitch(altitude):
    if altitude < 70000:
        return 90 * (1 - altitude / 70000)  # Чем выше высота, тем меньше наклон
    return 0


# Функция для расчета угла между осью X и силой тяги
def calculate_thrust_angle_to_x(x, y):
    """
    Рассчитывает угол между силой тяги и осью X.
    Учитывает кривизну Кербина.
    """
    # Вычисляем реальную высоту
    h = calculate_real_height(x, y)

    # Угол тангажа (между тягой и локальным горизонтом) в градусах
    pitch_deg = calculate_pitch(h)

    # Угол между радиальным направлением и вертикалью (осью Y)
    # Для малых углов это примерно x / R_kerbin
    phi_rad = x / R_kerbin  # угол в радианах

    # Угол между тягой и осью X
    # pitch - это угол к локальному горизонту, который повернут на угол phi
    angle_to_x_deg = pitch_deg - np.degrees(phi_rad)

    return np.radians(angle_to_x_deg)


# Функция для расчета гравитационного ускорения на текущей высоте
def gravitational_acceleration(x, y):
    """
    Рассчитывает гравитационное ускорение на текущей позиции.
    Возвращает компоненты g_x и g_y.
    """
    # Расстояние от центра Кербина
    r = np.sqrt(x ** 2 + (R_kerbin + y) ** 2)

    # Гравитационное ускорение (модуль)
    g_magnitude = G * M_kerbin / r ** 2

    # Направление к центру Кербина (отрицательное по оси Y)
    # Единичный вектор к центру Кербина
    if r > 0:
        g_dir_x = -x / r
        g_dir_y = -(R_kerbin + y) / r
    else:
        g_dir_x = 0
        g_dir_y = -1

    # Компоненты гравитационного ускорения
    g_x = g_magnitude * g_dir_x
    g_y = g_magnitude * g_dir_y

    return g_x, g_y


# Улучшенная функция для системы уравнений с исправлением начального сопротивления
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

    # Угол наклона тяги относительно оси X (в радианах)
    thrust_angle_to_x = calculate_thrust_angle_to_x(x_coord, y_coord)

    # Расчет реальной высоты
    h = calculate_real_height(x_coord, y_coord)

    # Расчет плотности воздуха
    air_density_value = air_density(h)

    # Проекции силы тяги на оси
    thrust_x = thrust * np.cos(thrust_angle_to_x)
    thrust_y = thrust * np.sin(thrust_angle_to_x)

    # Гравитационные компоненты
    g_x, g_y = gravitational_acceleration(x_coord, y_coord)

    # Сила сопротивления должна быть нулевой при нулевой скорости
    if velocity > 0.001:  # Уменьшен порог для более точного расчета
        # Вычисляем направление скорости (единичный вектор)
        direction_x = horizontal_velocity / velocity
        direction_y = vertical_velocity / velocity

        # Вычисляем силу сопротивления (модуль)
        drag_force_magnitude = 0.5 * C_d * air_density_value * velocity ** 2 * area

        # Проекции силы сопротивления на оси (направлены против скорости)
        drag_force_x = -drag_force_magnitude * direction_x
        drag_force_y = -drag_force_magnitude * direction_y
    else:
        # При нулевой или очень малой скорости сопротивление отсутствует
        drag_force_x = 0
        drag_force_y = 0

    # Суммарные силы
    total_force_x = thrust_x + drag_force_x + current_mass * g_x
    total_force_y = thrust_y + drag_force_y + current_mass * g_y

    # Ускорения
    acceleration_horizontal = total_force_x / current_mass
    acceleration_vertical = total_force_y / current_mass

    return [horizontal_velocity, acceleration_horizontal, vertical_velocity, acceleration_vertical]


# Функция для применения выброса (отделения ступени)
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

    # Угол наклона тяги в момент отделения
    thrust_angle_to_x = calculate_thrust_angle_to_x(x_coord, y_coord)

    # Применяем импульс от выброса
    dv = ejection_force / current_mass

    new_horizontal_velocity = horizontal_velocity + dv * np.cos(thrust_angle_to_x)
    new_vertical_velocity = vertical_velocity + dv * np.sin(thrust_angle_to_x)

    return [x_coord, new_horizontal_velocity, y_coord, new_vertical_velocity]


# Улучшенная функция для интегрирования с плотной сеткой в начале
def integrate_stage(initial_conditions, stage_index, initial_mass, fuel_mass, burn_time):
    try:
        # Создаем временную сетку с более высокой плотностью на первых секундах
        # Это поможет уменьшить погрешность в начале полета

        # Первые 2 секунды с очень мелким шагом
        if burn_time > 2:
            time_first_part = np.linspace(0, 2, 200)  # 200 точек за первые 2 секунды
            time_second_part = np.linspace(2, burn_time, max(1800, int((burn_time - 2) * 9)))
            time_eval = np.unique(np.concatenate([time_first_part, time_second_part]))
        else:
            # Если время горения меньше 2 секунд
            time_eval = np.linspace(0, burn_time, max(2000, int(burn_time * 100)))

        result = odeint(
            rocket_equations,
            initial_conditions,
            time_eval,
            args=(stage_index, initial_mass, fuel_mass, burn_time),
            atol=1e-7,  # Увеличена точность
            rtol=1e-7,  # Увеличена точность
            mxstep=20000,  # Увеличенное максимальное количество шагов
            hmax=0.5,  # Уменьшен максимальный шаг для лучшей точности
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

        # Расчет реальных высот
        real_heights = []
        for i in range(len(x_coords)):
            h = calculate_real_height(x_coords[i], y_coords[i])
            real_heights.append(h)
        real_heights = np.array(real_heights)

        # Расчет модуля скорости
        total_velocities = np.sqrt(x_velocities ** 2 + y_velocities ** 2)

        # Получение данных из симуляции KSP
        PATH = str(pathlib.Path(__file__).parent.joinpath("ksp_flight_data.csv"))
        data = pd.read_csv(PATH)

        time_ksp = data['Time']
        x_coords_ksp = data['Displacement']
        x_velocities_ksp = data['Horizontal Velocity']
        y_coords_ksp = data['Altitude']
        y_velocities_ksp = data['Vertical Velocity']

        # Расчет модуля скорости для KSP данных
        total_velocities_ksp = np.sqrt(x_velocities_ksp ** 2 + y_velocities_ksp ** 2)

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
            real_heights = real_heights[i:]
            total_velocities = total_velocities[i:]
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
        total_velocities_remap = []

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
            total_velocity = lerp(dt, total_velocities_ksp[idx_ksp], total_velocities_ksp[idx_ksp + 1])

            time_remap.append(time[idx])
            x_coords_remap.append(x_coord)
            x_velocities_remap.append(x_velocity)
            y_coords_remap.append(y_coord)
            y_velocities_remap.append(y_velocity)
            total_velocities_remap.append(total_velocity)


        # Вычисление погрешностей
        def abs_error(values):
            return abs(values[1] - values[0])


        def rel_error(values):
            if abs(values[0]) < 1e-6:  # Избегаем деления на ноль
                return 0
            return abs(values[1] - values[0]) / abs(values[0]) * 100


        # Расчет погрешностей для реальных высот
        height_abs_error = list(map(abs_error, zip(real_heights[:len(y_coords_remap)], y_coords_remap)))
        x_coords_abs_error = list(map(abs_error, zip(x_coords[:len(x_coords_remap)], x_coords_remap)))
        total_velocities_abs_error = list(
            map(abs_error, zip(total_velocities[:len(total_velocities_remap)], total_velocities_remap)))

        height_rel_error = list(map(rel_error, zip(real_heights[:len(y_coords_remap)], y_coords_remap)))
        x_coords_rel_error = list(map(rel_error, zip(x_coords[:len(x_coords_remap)], x_coords_remap)))
        total_velocities_rel_error = list(
            map(rel_error, zip(total_velocities[:len(total_velocities_remap)], total_velocities_remap)))

        # Построение графиков с новой компоновкой
        fig = plt.figure(figsize=(16, 12))

        # Создаем сетку 2x2 для основных графиков и 1x1 для объединенного графика погрешностей
        gs = fig.add_gridspec(2, 2, wspace=0.25, hspace=0.35, left=0.4, right=0.95)

        # График реальной высоты (вверху слева)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(time[:len(time_remap)], real_heights[:len(time_remap)], label='Модель', color='blue', linewidth=2)
        ax1.plot(time_remap, y_coords_remap, label='KSP', color='orange', linestyle='--', linewidth=2)

        # Добавляем вертикальную линию на 80 секундах (отделение ступени)
        ax1.axvline(x=80, color='green', linestyle='-', linewidth=2, alpha=0.5, label='Отделение 1-й ступени (80 с)')

        ax1.set_title('Высота', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Время (с)', fontsize=10)
        ax1.set_ylabel('Высота (м)', fontsize=10)
        # Перемещаем легенду в верхний левый угол, чтобы она не перекрывала график
        ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax1.grid(True, alpha=0.3)

        # График модуля скорости (вверху справа)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(time[:len(time_remap)], total_velocities[:len(time_remap)], label='Модель', color='blue', linewidth=2)
        ax2.plot(time_remap, total_velocities_remap, label='KSP', color='orange', linestyle='--', linewidth=2)

        # Добавляем вертикальную линию на 80 секундах (отделение ступени)
        ax2.axvline(x=80, color='green', linestyle='-', linewidth=2, alpha=0.5)

        ax2.set_title('Модуль скорости от времени', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Время (с)', fontsize=10)
        ax2.set_ylabel('Скорость (м/с)', fontsize=10)
        ax2.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax2.grid(True, alpha=0.3)

        # График смещения (внизу слева)
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(time[:len(time_remap)], x_coords[:len(time_remap)], label='Модель', color='blue', linewidth=2)
        ax3.plot(time_remap, x_coords_remap, label='KSP', color='orange', linestyle='--', linewidth=2)

        # Добавляем вертикальную линию на 80 секундах (отделение ступени)
        ax3.axvline(x=80, color='green', linestyle='-', linewidth=2, alpha=0.5)

        ax3.set_title('Горизонтальное смещение', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Время (с)', fontsize=10)
        ax3.set_ylabel('Смещение (м)', fontsize=10)
        ax3.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax3.grid(True, alpha=0.3)

        # ОБЪЕДИНЕННЫЙ график относительных погрешностей (внизу справа)
        ax4 = fig.add_subplot(gs[1, 1])

        # Строим все три погрешности на одном графике
        ax4.plot(time_remap, height_rel_error, label='Высота', color='red', linewidth=2, alpha=0.8)
        ax4.plot(time_remap, total_velocities_rel_error, label='Скорость', color='purple', linewidth=2, alpha=0.8,
                 linestyle='-')
        ax4.plot(time_remap, x_coords_rel_error, label='Смещение', color='darkorange', linewidth=2, alpha=0.8,
                 linestyle='-')

        # Добавляем вертикальную линию на 80 секундах (отделение ступени)
        ax4.axvline(x=80, color='green', linestyle='-', linewidth=2, alpha=0.5, label='Отделение ступени')

        ax4.set_title('Относительные погрешности', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Время (с)', fontsize=10)
        ax4.set_ylabel('Погрешность (%)', fontsize=10)
        ax4.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax4.grid(True, alpha=0.3)

        # Настраиваем пределы оси Y для лучшей читаемости
        all_errors = height_rel_error + total_velocities_rel_error + x_coords_rel_error
        max_error = max(all_errors) if all_errors else 10
        ax4.set_ylim(0, max_error * 1.1)

        # Информация о полете (слева)
        ax_text = fig.add_axes([0.05, 0.1, 0.3, 0.8])
        ax_text.axis('off')

        # Расчет средних погрешностей
        avg_height_error = np.mean(height_rel_error) if height_rel_error else 0
        avg_speed_error = np.mean(total_velocities_rel_error) if total_velocities_rel_error else 0
        avg_displacement_error = np.mean(x_coords_rel_error) if x_coords_rel_error else 0

        # Добавляем информацию об исправлениях
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

        Максимальная высота: {max(real_heights) / 1000:.1f} км
        Максимальная скорость: {max(total_velocities):.1f} м/с
        Горизонтальная скорость: {x_velocities[-1]:.1f} м/с
        Вертикальная скорость: {y_velocities[-1]:.1f} м/с
        Модуль скорости: {total_velocities[-1]:.1f} м/с
        Общее время: {time[-1]:.1f} с

        {'=' * 30}

        СТАТИСТИКА ПОГРЕШНОСТЕЙ:
        Средняя погрешность высоты: {avg_height_error:.2f} %
        Средняя погрешность скорости: {avg_speed_error:.2f} %
        Средняя погрешность смещения: {avg_displacement_error:.2f} %
        Макс. погрешность высоты: {max(height_rel_error):.2f} %
        Макс. погрешность скорости: {max(total_velocities_rel_error):.2f} %
        Макс. погрешность смещения: {max(x_coords_rel_error):.2f} %
        {'=' * 30}
        """
        ax_text.text(0.05, 0.95, flight_info, fontsize=10, fontfamily='monospace',
                     verticalalignment='top', horizontalalignment='left',
                     transform=ax_text.transAxes, linespacing=1.5)

        plt.savefig("final_corrected_with_real_height.png", dpi=150, bbox_inches='tight')
        plt.show()

        # Вывод статистики
        print("\nСтатистика моделирования:")
        print(f"  Максимальная реальная высота: {max(real_heights) / 1000:.2f} км")
        print(f"  Максимальная скорость: {max(total_velocities):.1f} м/с")
        print(f"  Конечная скорость: {total_velocities[-1]:.1f} м/с")
        print(f"  Общее время моделирования: {time[-1]:.1f} с")
        print(f"  Момент отделения первой ступени: 80 с")
        print("\nСтатистика погрешностей:")
        print(f"  Средняя погрешность высоты: {avg_height_error:.2f} %")
        print(f"  Средняя погрешность скорости: {avg_speed_error:.2f} %")
        print(f"  Средняя погрешность смещения: {avg_displacement_error:.2f} %")

    else:
        print("Ошибка: Не удалось интегрировать вторую ступень")
else:
    print("Ошибка: Не удалось интегрировать первую ступень")
