import krpc
import time
import csv
from math import sqrt
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation

def update_plots(axes, time_data, altitude_data, vertical_velocity_data, horizontal_velocity_data, stage_num):

    """Обновление графиков в реальном времени"""
    
    # Очищаем оси перед перерисовкой
    for ax in axes:
        ax.clear()
    
    # График 1: Высота от времени
    axes[0].plot(time_data, altitude_data, 'b-', linewidth=2)
    axes[0].set_title(f'Высота ракеты от времени (Ступень {stage_num})', fontsize=12)
    axes[0].set_xlabel('Время (с)')
    axes[0].set_ylabel('Высота (м)')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(left=0)
    
    # Отмечаем отделение ступени на 80 секунде только для первой ступени
    if stage_num == 1 and time_data and time_data[-1] >= 80:
        axes[0].axvline(x=80, color='r', linestyle='--', alpha=0.7, label='Отделение ступени')
        axes[0].legend()
    
    # График 2: Вертикальная скорость от времени
    axes[1].plot(time_data, vertical_velocity_data, 'g-', linewidth=2)
    axes[1].set_title(f'Вертикальная скорость от времени (Ступень {stage_num})', fontsize=12)
    axes[1].set_xlabel('Время (с)')
    axes[1].set_ylabel('Вертикальная скорость (м/с)')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(left=0)
    
    # Отмечаем отделение ступени на 80 секунде только для первой ступени
    if stage_num == 1 and time_data and time_data[-1] >= 80:
        axes[1].axvline(x=80, color='r', linestyle='--', alpha=0.7)
    
    # График 3: Горизонтальная скорость от времени
    axes[2].plot(time_data, horizontal_velocity_data, 'r-', linewidth=2)
    axes[2].set_title(f'Горизонтальная скорость от времени (Ступень {stage_num})', fontsize=12)
    axes[2].set_xlabel('Время (с)')
    axes[2].set_ylabel('Горизонтальная скорость (м/с)')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(left=0)
    
    # Отмечаем отделение ступени на 80 секунде только для первой ступени
    if stage_num == 1 and time_data and time_data[-1] >= 80:
        axes[2].axvline(x=80, color='r', linestyle='--', alpha=0.7)
    
    # Настраиваем расположение графиков
    plt.tight_layout()
    
    # Обновляем график
    plt.draw()
    plt.pause(0.001)

def save_stage_graphs(stage_num, time_data, altitude_data, vertical_velocity_data, horizontal_velocity_data):


    """Сохранение графиков для конкретной ступени"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f'Графики полета ракеты - Ступень {stage_num}', fontsize=16)
    
    # График 1: Высота от времени
    axes[0].plot(time_data, altitude_data, 'b-', linewidth=2.5)
    axes[0].set_title('Высота ракеты от времени', fontsize=14)
    axes[0].set_xlabel('Время (с)', fontsize=12)
    axes[0].set_ylabel('Высота (м)', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(left=0)
    
    # Отмечаем отделение ступени на 80 секунде только для первой ступени
    if stage_num == 1 and time_data and time_data[-1] >= 80:
        axes[0].axvline(x=80, color='r', linestyle='--', linewidth=2, alpha=0.7, label='Отделение ступени (80с)')
        axes[0].legend(fontsize=11)
    
    # График 2: Вертикальная скорость от времени
    axes[1].plot(time_data, vertical_velocity_data, 'g-', linewidth=2.5)
    axes[1].set_title('Вертикальная скорость от времени', fontsize=14)
    axes[1].set_xlabel('Время (с)', fontsize=12)
    axes[1].set_ylabel('Вертикальная скорость (м/с)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(left=0)
    
    # Отмечаем отделение ступени на 80 секунде только для первой ступени
    if stage_num == 1 and time_data and time_data[-1] >= 80:
        axes[1].axvline(x=80, color='r', linestyle='--', linewidth=2, alpha=0.7)
    
    # График 3: Горизонтальная скорость от времени
    axes[2].plot(time_data, horizontal_velocity_data, 'r-', linewidth=2.5)
    axes[2].set_title('Горизонтальная скорость от времени', fontsize=14)
    axes[2].set_xlabel('Время (с)', fontsize=12)
    axes[2].set_ylabel('Горизонтальная скорость (м/с)', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(left=0)
    
    # Отмечаем отделение ступени на 80 секунде только для первой ступени
    if stage_num == 1 and time_data and time_data[-1] >= 80:
        axes[2].axvline(x=80, color='r', linestyle='--', linewidth=2, alpha=0.7)
    
    # Добавляем подсказки
    if altitude_data:
        max_altitude = max(altitude_data)
        max_vert_speed = max(vertical_velocity_data)
        max_horiz_speed = max(horizontal_velocity_data)
        
        fig.text(0.02, 0.02, 
                f"Максимальные значения (Ступень {stage_num}):\nВысота: {max_altitude:.0f} м\n"
                f"Верт. скорость: {max_vert_speed:.0f} м/с\n"
                f"Гориз. скорость: {max_horiz_speed:.0f} м/с",
                fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    
    plt.tight_layout()
    
    # Сохраняем график
    filename = f'ksp_stage_{stage_num}_graphs_{timestamp}.png'
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Графики для ступени {stage_num} сохранены в файл: {filename}")
    
    plt.close(fig)
    return filename

def plot_combined_graphs(all_stages_data):
    """Построение общего графика всех ступеней"""
    plt.ioff()
    
    if not all_stages_data:
        print("Нет данных для построения общего графика")
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Общие графики полета всех ступеней ракеты', fontsize=16)
    
    # Цвета для разных ступеней
    colors = ['b', 'g', 'r', 'm', 'c', 'y']
    
    for i, stage_data in enumerate(all_stages_data):
        color = colors[i % len(colors)]
        stage_num = i + 1
        
        # График высоты
        axes[0].plot(stage_data['time_data'], stage_data['altitude_data'], 
                    f'{color}-', linewidth=2, label=f'Ступень {stage_num}')
        
        # График вертикальной скорости
        axes[1].plot(stage_data['time_data'], stage_data['vertical_velocity_data'], 
                    f'{color}-', linewidth=2)
        
        # График горизонтальной скорости
        axes[2].plot(stage_data['time_data'], stage_data['horizontal_velocity_data'], 
                    f'{color}-', linewidth=2)
    
    # Настройки осей
    axes[0].set_title('Высота ракеты от времени', fontsize=14)
    axes[0].set_xlabel('Время (с)', fontsize=12)
    axes[0].set_ylabel('Высота (м)', fontsize=12)
    axes[0].grid(True)
    axes[0].legend()
    
    axes[1].set_title('Вертикальная скорость от времени', fontsize=14)
    axes[1].set_xlabel('Время (с)', fontsize=12)
    axes[1].set_ylabel('Вертикальная скорость (м/с)', fontsize=12)
    axes[1].grid(True)
    
    axes[2].set_title('Горизонтальная скорость от времени', fontsize=14)
    axes[2].set_xlabel('Время (с)', fontsize=12)
    axes[2].set_ylabel('Горизонтальная скорость (м/с)', fontsize=12)
    axes[2].grid(True)
    
    plt.tight_layout()
    
    # Сохраняем общий график
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    fig.savefig(f'ksp_all_stages_combined_{timestamp}.png', dpi=300, bbox_inches='tight')
    print(f"Общий график всех ступеней сохранен")
    
    # Показываем график
    plt.show()

# Подключаемся к игре
conn = krpc.connect(name='Автопилот MF-RPE-1')
vessel = conn.space_center.active_vessel

# Создаем файл для записи данных
PATH = str(pathlib.Path(__file__).parent.joinpath("ksp_flight_data.csv"))
with open(PATH, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Time", "Altitude", "Vertical Velocity", "Horizontal Velocity",
                     "Total Velocity", "Drag", "Displacement"])

    # Подготовка к запуску
    vessel.control.sas = False
    vessel.control.rcs = False
    vessel.control.throttle = 1.0
    
    # Создаем фигуру и оси для графиков
    plt.ion()  # Включаем интерактивный режим
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle('Параметры полета ракеты - Ступень 1', fontsize=14)
    
    # Списки для хранения данных текущей ступени
    time_data = []
    altitude_data = []
    vertical_velocity_data = []
    horizontal_velocity_data = []
    
    # Список для хранения данных всех ступеней
    all_stages_data = []
    
    print('Запуск через 3...')
    time.sleep(1)
    print('2...')
    time.sleep(1)
    print('1...')
    time.sleep(1)

    # Счетчик времени
    start_time = conn.space_center.ut

    # Начальная позиция для расчета смещения
    initial_position = vessel.position(vessel.orbit.body.reference_frame)
    initial_position_vec_length = np.linalg.norm(initial_position)
    
    vessel.control.activate_next_stage()  # Запуск двигателей первой ступени
    time.sleep(0.1)
    vessel.control.activate_next_stage()  # Освобождение от стартовых клемм
    time.sleep(0.7)
    
    stage_main_engines = ['', 'Engine Cluster', 'Block Engine', 'Block I']
    stage = 1
    
    printed_high = False

    fuel_start_main = vessel.resources_in_decouple_stage(stage=5, cumulative=False).amount('LiquidFuel')

    print(f"Пуск!\nВремя старта: {start_time:.2f} с")
    
    # Основной цикл полета
    while True:
        # Настоящее время
        ut = conn.space_center.ut
        
        # Прошедшее время с начала
        elapsed_time = ut - start_time
        
        # Сбор данных
        altitude = vessel.flight().mean_altitude
        speed = vessel.flight(vessel.orbit.body.reference_frame).speed
        drag_x, drag_y, drag_z = vessel.flight().drag
        drag = sqrt(drag_x ** 2 + drag_y ** 2 + drag_z ** 2)

        # Текущее положение для расчета смещения
        current_position = vessel.position(vessel.orbit.body.reference_frame)
        
        # Расчет смещения
        current_position = current_position / np.linalg.norm(current_position) * initial_position_vec_length
        horizontal_displacement = np.linalg.norm(current_position - initial_position)
        
        # Получение скоростей
        vertical_speed = vessel.flight(vessel.orbit.body.reference_frame).vertical_speed
        horizontal_speed = vessel.flight(vessel.orbit.body.reference_frame).horizontal_speed

        # Сохраняем данные в списки
        time_data.append(elapsed_time)
        altitude_data.append(altitude)
        vertical_velocity_data.append(vertical_speed)
        horizontal_velocity_data.append(horizontal_speed)        

        # Записываем данные в файл
        writer.writerow([elapsed_time, altitude, vertical_speed, horizontal_speed, speed, drag, horizontal_displacement])

        # Наклон ракеты в зависимости от высоты
        vessel.auto_pilot.target_roll = 0
        vessel.auto_pilot.engage()
        if altitude < 70000:
            target_pitch = 90 * (1 - altitude / 70000)  # Чем выше высота, тем меньше наклон
            vessel.auto_pilot.target_pitch_and_heading(target_pitch, 90)
        else:
            vessel.auto_pilot.target_pitch_and_heading(0, 90)
            if not printed_high:
                print(f"Ракета достигла высоты 70.000 метров. Времени после старта прошло: {(ut - start_time):.2f}")
                printed_high = True

        # Отделение первой ступени через 80 секунд
        if stage == 1 and elapsed_time >= 80:
            # Сохраняем данные первой ступени
            stage_1_data = {
                'time_data': time_data.copy(),
                'altitude_data': altitude_data.copy(),
                'vertical_velocity_data': vertical_velocity_data.copy(),
                'horizontal_velocity_data': horizontal_velocity_data.copy()
            }
            all_stages_data.append(stage_1_data)
            
            # Сохранение графика первой ступени
            save_stage_graphs(1, time_data, altitude_data, vertical_velocity_data, horizontal_velocity_data)
            
            # Очистка данных для второй ступени
            time_data.clear()
            altitude_data.clear()
            vertical_velocity_data.clear()
            horizontal_velocity_data.clear()
            
            fuel_st2 = vessel.resources_in_decouple_stage(stage=5, cumulative=False).amount('LiquidFuel')
            # Отделение
            vessel.control.activate_next_stage()
            stage = 2
            print("Отделение ступени 1. Ускорители отделены.")
            print(f"Времени после старта прошло: {(ut - start_time):.2f}")
            print(f"Топлива главной ступени осталось: {fuel_start_main - fuel_st2}")    
            time_stage_1 = ut - start_time
            # Обновляем заголовок графика для второй ступени
            fig.suptitle(f'Параметры полета ракеты - Ступень {stage}', fontsize=14)
            plt.draw()
            plt.pause(0.1)
            
            time.sleep(1)
        
        # Проверка топлива для второй ступени
        if stage == 2:
            
            fuel_st2 = vessel.resources_in_decouple_stage(stage=5, cumulative=False).amount('LiquidFuel')
            if fuel_st2 < 1.0:
                # Сохраняем данные второй ступени
                stage_2_data = {
                    'time_data': time_data.copy(),
                    'altitude_data': altitude_data.copy(),
                    'vertical_velocity_data': vertical_velocity_data.copy(),
                    'horizontal_velocity_data': horizontal_velocity_data.copy()
                }
                all_stages_data.append(stage_2_data)
                    
                # Сохраняем график второй ступени
                save_stage_graphs(2, time_data, altitude_data, vertical_velocity_data, horizontal_velocity_data)
                    
                print("Отделение ступени 2")
                print(f"Времени после старта прошло: {(ut - start_time):.2f}")
                print(f"Время работы второй ступени: {(ut - start_time - time_stage_1):.2f} ")

                    
                vessel.control.activate_next_stage()
                stage = 3
                    
                time.sleep(3)


        # Завершение программы 
        if stage == 3:
            
             
            print('Конец программы полета')
            vessel.control.throttle = 0.0
                
            # Сохраняем данные третьей ступени
            if time_data:  # Если есть данные
                stage_3_data = {
                    'time_data': time_data.copy(),
                    'altitude_data': altitude_data.copy(),
                    'vertical_velocity_data': vertical_velocity_data.copy(),
                    'horizontal_velocity_data': horizontal_velocity_data.copy()
                }
                all_stages_data.append(stage_3_data)
                    
                # Сохраняем график третьей ступени
                save_stage_graphs(3, time_data, altitude_data, vertical_velocity_data, horizontal_velocity_data)
                
                # Закрываем интерактивный режим графиков
                plt.ioff()
                plt.close('all')
                
                # Строим и сохраняем общий график всех ступеней
                plot_combined_graphs(all_stages_data)
                
                # Сохраняем все данные в файл
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                np.savez(f'all_stages_data_{timestamp}.npz', all_stages_data=all_stages_data)
                print(f"Данные ступеней сохранены в файл: all_stages_data_{timestamp}.npz")
                
                break

        # Обновляем графики каждые 5 итераций
        if len(time_data) % 5 == 0:
            update_plots(axes, time_data, altitude_data, 
                        vertical_velocity_data, horizontal_velocity_data, stage)
            
        time.sleep(0.1)


conn.close()
