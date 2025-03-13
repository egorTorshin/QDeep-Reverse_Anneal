from neal import SimulatedAnnealingSampler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import itertools
import seaborn

# Подключаем симулятор
sampler = SimulatedAnnealingSampler()
print("Connected to sampler", sampler.__class__.__name__)

# Проверяем наличие аппаратных свойств. Для симулятора их нет, поэтому устанавливаем значения по умолчанию.
if "max_anneal_schedule_points" in sampler.properties:
    print("Maximum anneal-schedule points: {}".format(sampler.properties["max_anneal_schedule_points"]))
else:
    print("Maximum anneal-schedule points not available for SimulatedAnnealingSampler")

if "annealing_time_range" in sampler.properties:
    print("Annealing time range: {}".format(sampler.properties["annealing_time_range"]))
    max_slope = 1.0 / sampler.properties["annealing_time_range"][0]
    print("Maximum slope allowed on this solver is {:.2f}.".format(max_slope))
else:
    print("Annealing time range not available for SimulatedAnnealingSampler")
    max_slope = 1.0  # значение по умолчанию

# Создаем расписание для reverse-аннинга с помощью функции-ассистента
from helpers.schedule import make_reverse_anneal_schedule
reverse_schedule = make_reverse_anneal_schedule(s_target=0.45, hold_time=80, ramp_up_slope=max_slope)
time_total = reverse_schedule[-1][0]  # общее время – время последней точки расписания

print("Reverse anneal schedule:")
print(reverse_schedule)
print("Total anneal-schedule time is {} us".format(time_total))

# Рисуем расписание обратного отжига
plt.figure(1, figsize=(3, 3))
plt.plot(*np.array(reverse_schedule).T)
plt.title("Reverse Anneal Schedule")
plt.xlabel("Time [us]")
plt.ylabel("Annealing Parameter s")
plt.ylim([0.0, 1.0])
plt.show()

# Вместо sampler.nodelist и sampler.edgelist задаем число квантов вручную.
num_qubits = 16
h = {v: 0.0 for v in range(num_qubits)}

# Генерируем случайный граф: для каждой пары квантов с вероятностью 0.3 создаём ребро с весом -1 или +1
J = {(i, j): np.random.choice([-1, 1])
     for i, j in itertools.combinations(range(num_qubits), 2)
     if np.random.rand() < 0.3}

print("Bias 0 assigned to", len(h), "qubits.")
print("Randomly assigned strengths -1/+1 to", len(J), "couplers.")

# Выполним стандартный (forward) отжиг.
runs = 1000
# Используем общее число проходов, равное time_total (приводим к целому числу)
forward_answer = sampler.sample_ising(h, J, num_reads=runs, num_sweeps=int(time_total))
forward_solutions, forward_energies = forward_answer.record.sample, forward_answer.record.energy
# Определим индекс примерно в 5% от выборки с наименьшей энергией
i5 = int(5.0 / 95 * len(forward_answer.record.energy))
initial = dict(zip(forward_answer.variables, forward_answer.record[i5].sample))

print("Lowest energy found in forward annealing: {}".format(forward_answer.record.energy[0]))
print("Average energy is {:.2f} with standard deviation {:.2f}"
      .format(forward_energies.mean(), forward_energies.std()))
print("\nSetting the initial state to a sample with energy: {}".format(forward_answer.record.energy[i5]))

# Reverse-аннинг с использованием полученного начального состояния.
reverse_answer = sampler.sample_ising(h, J, num_reads=runs, num_sweeps=int(time_total), initial_state=initial)
reverse_solutions, reverse_energies = reverse_answer.record.sample, reverse_answer.record.energy

print("Lowest energy found in reverse annealing: {}".format(reverse_answer.record.energy[0]))
print("Average energy is {:.2f} with standard deviation {:.2f}"
      .format(reverse_energies.mean(), reverse_energies.std()))

# Далее переопределяем задачу Изинга для дальнейших экспериментов.
h = {0: 1.0, 1: -1.0, 2: -1.0, 3: 1.0, 4: 1.0, 5: -1.0, 6: 0.0, 7: 1.0,
     8: 1.0, 9: -1.0, 10: -1.0, 11: 1.0, 12: 1.0, 13: 0.0, 14: -1.0, 15: 1.0}
J = {(9, 13): -1, (2, 6): -1, (8, 13): -1, (9, 14): -1, (9, 15): -1,
     (10, 13): -1, (5, 13): -1, (10, 12): -1, (1, 5): -1, (10, 14): -1,
     (0, 5): -1, (1, 6): -1, (3, 6): -1, (1, 7): -1, (11, 14): -1,
     (2, 5): -1, (2, 4): -1, (6, 14): -1}

# Для симулятора не требуется эмбеддинг – используем sampler напрямую.
sampler_embedded = sampler

from dimod import ising_energy

# Определяем глобальный минимум для сравнения.
ground_state = {s_i: -1 for s_i in range(16)}
ground_energy = ising_energy(ground_state, h, J)

print("Energy of the global minimum: {}".format(ground_energy))
print("Energy of a random sample: {}".format(
    ising_energy({s_i: 2 * random.randint(0, 1) - 1 for s_i in range(16)}, h, J)
))

reads = 1000
# Задаем расписание с паузой (используем для графика и расчета общего числа проходов)
pause_schedule = [[0.0, 0.0],
                  [50.0, 0.5],
                  [time_total - 0.5 / max_slope, 0.5],
                  [time_total, 1.0]]
plt.figure(2, figsize=(3, 3))
plt.plot(*np.array(pause_schedule).T)
plt.title("Anneal Schedule with a Pause")
plt.xlabel("Time [us]")
plt.ylabel("Annealing Parameter s")
plt.ylim([0.0, 1.0])
plt.show()

# Функция для переворачивания битов в выборке
def flip_bits(sample, spin_bits):
    sample.update({bit: -sample[bit] for bit in spin_bits})
    return sample

# Создаем начальное состояние, переворачивая выбранные биты от глобального минимума
initial = flip_bits(dict(ground_state), {1, 4, 7, 10, 12, 15})
print("Energy of initial state: {}".format(ising_energy(initial, h, J)))

# Функция для вычисления Hamming distance
get_hamming_distance = lambda x1, x2: np.sum(x1 != x2)
def get_hamming_distances(sols):
    sols = np.array(sols)
    return np.array([get_hamming_distance(x1, x2) for x1, x2 in zip(sols, sols[1:])])

def analyze(answer):
    solutions, energies = answer.record.sample, answer.record.energy
    energy_best = round(answer.record.energy[0], 2)
    ratio = list(answer.record.energy).count(ground_energy) / float(reads)
    hamming_distances = get_hamming_distances(solutions)
    energy_mean = round(energies.mean(), 2)
    hamming_mean = round(hamming_distances.mean(), 2) if len(hamming_distances) > 0 else 0
    return [solutions, energies, hamming_distances, energy_best, ratio, energy_mean, hamming_mean]

data = []

# Стандартный forward annealing (используем num_sweeps = time_total)
answer = sampler_embedded.sample_ising(h, J, num_reads=reads, num_sweeps=int(time_total))
print("Obtained {} samples from standard forward annealing".format(len(answer.record.energy)))
data.append(analyze(answer))

# Forward annealing с паузой (используем число проходов из последней точки pause_schedule)
pause_sweeps = int(pause_schedule[-1][0])
answer = sampler_embedded.sample_ising(h, J, num_reads=reads, num_sweeps=pause_sweeps)
print("Obtained {} samples from forward annealing with a pause".format(len(answer.record.energy)))
data.append(analyze(answer))

# Reverse annealing (используем начальное состояние)
answer = sampler_embedded.sample_ising(h, J, num_reads=reads, num_sweeps=int(time_total), initial_state=initial)
print("Obtained {} samples from reverse annealing".format(len(answer.record.energy)))
data.append(analyze(answer))

df_columns = ["Solutions", "Energy", "Hamming Distance", "Energy (Lowest)",
              "Global-Minimum Ratio", "Energy (Average)", "Hamming (Average)"]
df_rows = ["Forward", "Pause", "Reverse"]
df = pd.DataFrame(data, index=df_rows, columns=df_columns)

print("Lowest energy found for each method:\n")
print(df["Energy (Lowest)"])
print("\n\nRatio of global minimum to all samples:\n")
print(df["Global-Minimum Ratio"])

from helpers.plot import e_h_plot
e_h_plot(df)

seaborn.set_style('whitegrid')

s_target_vals = [0.3, 0.45, 0.6, 0.75, 0.9]
hold_time_vals = [10, 100]
# Для экспериментов по reverse annealing создаем начальное состояние
initial = flip_bits(dict(ground_state), {0, 7, 15})
data_rows = []
# Здесь QPU_time не используется для симулятора.
for s_target, hold_time in itertools.product(s_target_vals, hold_time_vals):
    print("Running reverse anneals with s_target={} and hold_time={} us".format(s_target, hold_time))
    schedule = make_reverse_anneal_schedule(s_target=s_target, hold_time=hold_time, ramp_up_slope=max_slope)
    num_sweeps_modulating = int(schedule[-1][0])
    answer = sampler_embedded.sample_ising(h, J, num_reads=1000, num_sweeps=num_sweeps_modulating, initial_state=initial)
    modulating_solutions, modulating_energies = answer.record.sample, answer.record.energy
    modulating_distances = get_hamming_distances(modulating_solutions)
    row = dict(
        s_target=s_target,
        hold_time=hold_time,
        energy_stdev=np.std(modulating_energies),
        energy_mean=np.mean(modulating_energies),
        distance_mean=np.mean(modulating_distances) if len(modulating_distances) > 0 else 0,
    )
    data_rows.append(row)

df_exp = pd.DataFrame(data_rows)
print("Reverse annealing experiments complete.")
print(df_exp)

df_long = pd.melt(df_exp, id_vars=['s_target', 'hold_time'])
g = seaborn.FacetGrid(df_long, hue='hold_time', col='variable', sharey=False, legend_out=True)
g.map(plt.plot, 's_target', 'value')
g.add_legend()
plt.suptitle("Statistics for Varying Reverse Annealing Parameters")
plt.subplots_adjust(left=.1, right=.9, top=.8)
plt.show()

# Финальный reverse annealing с выбранными параметрами.
# Выбираем начальное состояние с большей Hamming distance
initial = flip_bits(dict(ground_state), {1, 2, 5, 6, 9, 10, 13, 14, 15})
s_target_vals = [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35]
hold_time_vals = [40, 100]
x, y = 5, 0  # выбираем s_target и hold_time по индексам
s_target, hold_time = s_target_vals[x], hold_time_vals[y]
schedule = make_reverse_anneal_schedule(s_target, hold_time, ramp_up_slope=max_slope)
print("Final reverse anneal schedule:")
print(schedule)
plt.figure()
plt.plot(*np.array(schedule).T)
plt.title("Anneal Schedule")
plt.xlabel("Time [us]")
plt.ylabel("Annealing Parameter s")
plt.ylim([0.0, 1.0])
plt.show()

answer = sampler_embedded.sample_ising(h, J, num_reads=1000, num_sweeps=int(schedule[-1][0]), initial_state=initial)
print("s_target = {} and hold_time = {} us".format(s_target, hold_time))
print("Solution =", answer.record.sample[0], "Energy =", answer.record.energy[0])
ratio = list(answer.record.energy).count(ground_energy) / float(reads)
print("Ratio of global minimum to other samples is", ratio)
