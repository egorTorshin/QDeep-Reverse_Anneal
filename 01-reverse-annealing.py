from qdeepsdk import QDeepHybridSolver
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import itertools
import seaborn
from dimod import BinaryQuadraticModel, ising_energy

# Initialize the QDeepHybridSolver
solver = QDeepHybridSolver()
solver.token = "your-auth-token-here"
print("Connected to solver", solver.__class__.__name__)

# For reverse annealing, we use a helper function (the schedule is only for demonstration and does not affect the solution)
from helpers.schedule import make_reverse_anneal_schedule
max_slope = 1.0  # default value, as our Solver does not provide dynamic details
reverse_schedule = make_reverse_anneal_schedule(s_target=0.45, hold_time=80, ramp_up_slope=max_slope)
time_total = reverse_schedule[-1][0]
print("Reverse anneal schedule:")
print(reverse_schedule)
print("Total anneal-schedule time is {} us".format(time_total))
plt.figure(1, figsize=(3, 3))
plt.plot(*np.array(reverse_schedule).T)
plt.title("Reverse Anneal Schedule")
plt.xlabel("Time [us]")
plt.ylabel("Annealing Parameter s")
plt.ylim([0.0, 1.0])
plt.show()

# Define a random Ising problem
num_qubits = 16
h = {v: 0.0 for v in range(num_qubits)}
J = {(i, j): np.random.choice([-1, 1])
     for i, j in itertools.combinations(range(num_qubits), 2)
     if np.random.rand() < 0.3}

print("Bias 0 assigned to", len(h), "qubits.")
print("Randomly assigned strengths -1/+1 to", len(J), "couplers.")

runs = 1000

def solve_ising(h, J):
    """
    Converts the Ising problem to a QUBO and solves it using QDeepHybridSolver.
    Returns the found solution and its computed energy.
    """
    bqm = BinaryQuadraticModel.from_ising(h, J)
    qubo, offset = bqm.to_qubo()
    size = len(h)
    matrix = np.zeros((size, size))
    for (i, j), val in qubo.items():
        matrix[i, j] = val
    result = solver.solve(matrix)
    sample = result.get('sample')
    energy = ising_energy(sample, h, J)
    return sample, energy

# Solve the problem multiple times to simulate forward annealing
forward_solutions = []
forward_energies = []
for _ in range(runs):
    sample, energy = solve_ising(h, J)
    forward_solutions.append(sample)
    forward_energies.append(energy)

sorted_indices = np.argsort(forward_energies)
lowest_energy = forward_energies[sorted_indices[0]]
print("Lowest energy found in forward annealing: {}".format(lowest_energy))
print("Average energy is {:.2f} with standard deviation {:.2f}"
      .format(np.mean(forward_energies), np.std(forward_energies)))

# Select one of the solutions to simulate an initial state (for reverse annealing)
i5 = int(5.0 / 95 * runs)
initial = forward_solutions[sorted_indices[i5]]
print("\nSetting the initial state to a sample with energy: {}".format(forward_energies[sorted_indices[i5]]))

# Simulate reverse annealing by running the solver repeatedly (initial_state is not supported, so we just repeat solve)
reverse_solutions = []
reverse_energies = []
for _ in range(runs):
    sample, energy = solve_ising(h, J)
    reverse_solutions.append(sample)
    reverse_energies.append(energy)
lowest_energy_reverse = min(reverse_energies)
print("Lowest energy found in reverse annealing: {}".format(lowest_energy_reverse))
print("Average energy is {:.2f} with standard deviation {:.2f}"
      .format(np.mean(reverse_energies), np.std(reverse_energies)))

# Fixed Ising problem for further experiments
h_fixed = {0: 1.0, 1: -1.0, 2: -1.0, 3: 1.0, 4: 1.0, 5: -1.0, 6: 0.0, 7: 1.0,
           8: 1.0, 9: -1.0, 10: -1.0, 11: 1.0, 12: 1.0, 13: 0.0, 14: -1.0, 15: 1.0}
J_fixed = {(9, 13): -1, (2, 6): -1, (8, 13): -1, (9, 14): -1, (9, 15): -1,
           (10, 13): -1, (5, 13): -1, (10, 12): -1, (1, 5): -1, (10, 14): -1,
           (0, 5): -1, (1, 6): -1, (3, 6): -1, (1, 7): -1, (11, 14): -1,
           (2, 5): -1, (2, 4): -1, (6, 14): -1}

ground_state = {s: -1 for s in range(16)}
ground_energy = ising_energy(ground_state, h_fixed, J_fixed)
print("Energy of the global minimum: {}".format(ground_energy))
print("Energy of a random sample: {}".format(
    ising_energy({s: 2 * random.randint(0, 1) - 1 for s in range(16)}, h_fixed, J_fixed)
))

def sample_experiment(h, J, runs):
    sols = []
    energies = []
    for _ in range(runs):
        sample, energy = solve_ising(h, J)
        sols.append(sample)
        energies.append(energy)
    return sols, np.array(energies)

reads = 1000
forward_sols, forward_energy_array = sample_experiment(h_fixed, J_fixed, reads)
print("Obtained {} samples from standard forward annealing".format(len(forward_energy_array)))
pause_sols, pause_energy_array = sample_experiment(h_fixed, J_fixed, reads)
print("Obtained {} samples from forward annealing with a pause".format(len(pause_energy_array)))
reverse_sols, reverse_energy_array = sample_experiment(h_fixed, J_fixed, reads)
print("Obtained {} samples from reverse annealing".format(len(reverse_energy_array)))

data = []
for method, energies in zip(["Forward", "Pause", "Reverse"],
                            [forward_energy_array, pause_energy_array, reverse_energy_array]):
    energy_best = round(min(energies), 2)
    ratio = list(energies).count(ground_energy) / float(reads)
    energy_mean = round(np.mean(energies), 2)
    energy_std = round(np.std(energies), 2)
    data.append([method, energy_best, ratio, energy_mean, energy_std])
df = pd.DataFrame(data, columns=["Method", "Energy (Lowest)", "Global-Minimum Ratio", "Energy (Average)", "Energy StdDev"])
print("Experiment results:\n", df)

# Plot the annealing schedule with a pause (for demonstration)
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

# Experiments varying reverse annealing parameters
s_target_vals = [0.3, 0.45, 0.6, 0.75, 0.9]
hold_time_vals = [10, 100]
data_rows = []
for s_target, hold_time in itertools.product(s_target_vals, hold_time_vals):
    print("Running reverse anneals with s_target={} and hold_time={} us".format(s_target, hold_time))
    schedule = make_reverse_anneal_schedule(s_target=s_target, hold_time=hold_time, ramp_up_slope=max_slope)
    # In our case, the schedule is not used to control the solution - we simply collect statistics
    sols, energies = sample_experiment(h_fixed, J_fixed, 1000)
    row = {
        's_target': s_target,
        'hold_time': hold_time,
        'energy_stdev': np.std(energies),
        'energy_mean': np.mean(energies),
        'energy_best': min(energies)
    }
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

# Final reverse annealing experiment
initial = {s: 2 * random.randint(0, 1) - 1 for s in range(16)}
s_target_vals = [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35]
hold_time_vals = [40, 100]
x, y = 5, 0
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

final_sample, final_energy = solve_ising(h_fixed, J_fixed)
print("s_target = {} and hold_time = {} us".format(s_target, hold_time))
print("Solution =", final_sample, "Energy =", final_energy)
ratio = list(forward_energy_array).count(ground_energy) / float(reads)
print("Ratio of global minimum to other samples is", ratio)
