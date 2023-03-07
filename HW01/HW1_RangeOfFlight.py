#***`Range of flight`***

#construction a 4 histogram of the flight range when its formula is $$L = \frac{v_0^2}{g} \sin(2\alpha)$$ (the formula for the flight range of a bullet fired at an angle to the horizon) and angle can be initial velocity from normal and uniform distirbution and  and initial velocity from normal and uniform distribution.
# 1. generate data arrays for angle and initial velocity from normal and uniform distribution
# 2. calculate the flight range distribution
# 3. construct a histogram of the flight range
# 4. save experiments parameters into json

#Exampl for math part: https://study.com/skill/learn/how-to-calculate-the-range-of-a-projectile-explanation.html#:~:text=25.0%20m%20%2F%20s%20.-,Step%202%3A%20Identify%20the%20angle%20at%20which%20a%20projectile%20is,0%202%20g%20sin%20%E2%81%A1%20.
#референсные значения для проверки на коррекктность логики (если результат будет иметь нормальное распределение):
#30° × π/180 = 0.5236 rad
#flight range =((50*50) / 9.81) * sin (2*0.5236) = 220.699956214

#референсные значения для проверки на коррекктность логики (если разброс близкий к 0):
#45° × π/180 = 0.785398 rad
#flight range =((30*30) / 9.81) * sin (2*0.785398) = 91.7431192661

import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import random
import json


# Constants
g = 9.81  # gravitational acceleration (m/s^2)
sample_size = 100000

# parameters for normal distribution
angle_loc = 45.0  # mean
angle_scale = 2  # 0.0001 #std

v0_loc = 30.0  # mean
v0_scale = 2  # 0.0001  #std

# parameters for uniform distribution

# uniform distribution can be between 0 and 90 degrees
angle_low = angle_loc - (angle_scale / 2.0)
angle_high = angle_loc + (angle_scale / 2.0)
v0_low = v0_loc - (v0_scale / 2.0)
v0_high = v0_loc + (v0_scale / 2.0)


class Experiment:
    def __init__(self, angle, v0, name):
        self.angle = angle
        self.v0 = v0
        self.range = None
        self.name = name

    def generate_data(self, size):
        self.range = self.calculate_flight_range(self.angle, self.v0)

    @staticmethod
    def calculate_flight_range(angle, v0):
        return v0 ** 2 / g * np.sin(2 * np.radians(angle))


def plot_experiment(experiment, ax):
    color = random.choice(sns.color_palette('pastel', 10))
    sns.histplot(data=experiment.range, ax=ax, alpha=0.7, stat='density', kde=True, color=color)
    ax.set_xlabel('Flight Range (m)', fontweight='medium', fontsize=12)
    ax.set_ylabel('Frequency', fontweight='medium', fontsize=12)
    ax.scatter(experiment.range, np.zeros_like(experiment.range), alpha=0.3, color=color)
    ax.margins(x=0.02)


angle_normal_distribution = np.random.normal(loc=angle_loc, scale=angle_scale, size=sample_size)
v0_normal_distribution = np.random.normal(loc=v0_loc, scale=v0_scale, size=sample_size)
angle_uniform_distribution = np.random.uniform(low=angle_low, high=angle_high,
                                               size=sample_size)  # uniform distribution (can set angle between 0 and 90 degrees)
v0_uniform_distribution = np.random.uniform(low=v0_low, high=v0_high, size=sample_size)

# print(angle_normal_distribution.mean())
# print(angle_uniform_distribution.mean())
# print(v0_normal_distribution.mean())
# print(v0_uniform_distribution.mean())


normal_normal = Experiment(
    angle=angle_normal_distribution,
    v0=v0_normal_distribution,
    name='Normal Angle - Normal Velocity'
)

normal_uniform = Experiment(
    angle=angle_normal_distribution,
    v0=v0_uniform_distribution,
    name='Normal Angle - Uniform Velocity'
)

uniform_normal = Experiment(
    angle=angle_uniform_distribution,
    v0=v0_normal_distribution,
    name='Uniform Angle - Normal Velocity'
)

uniform_uniform = Experiment(
    angle=angle_uniform_distribution,
    v0=v0_uniform_distribution,
    name='Uniform Angle - Uniform Velocity'
)

experiments = [normal_normal, normal_uniform, uniform_normal, uniform_uniform]

# Generate data for each experiment
for experiment in experiments:
    experiment.generate_data(size=sample_size)

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(6, 4), sharey=False, dpi=200)
for i, experiment in enumerate(experiments):
    row = i // 2
    col = i % 2
    ax = axs[row][col]
    color = random.choice(sns.color_palette('pastel', 10))
    sns.histplot(data=experiment.range, ax=ax, alpha=0.7, stat='density', kde=True, color=color)
    ax.set_xlabel('Flight Range (m)', fontsize=7)
    ax.set_ylabel('Density', fontsize=7, color=color)
    ax.tick_params(axis='x', labelsize=7)
    ax.tick_params(axis='y', labelsize=7, colors=color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(color)
    ax.yaxis.label.set_color(color)
    ax.tick_params(axis='y', colors=color)
    ax.set_title(experiment.name, fontweight='semibold', fontsize=10, y=1.05)

    # add mean and standard deviation lines
    mean = np.mean(experiment.range)
    std = np.std(experiment.range)

    ax.axvline(mean, color=color, linestyle='--', linewidth=1)
    ax.axvline(mean - std, color=color, linestyle=':', linewidth=1)
    ax.axvline(mean + std, color=color, linestyle=':', linewidth=1)

    # add range
    range = np.ptp(experiment.range)
    ax.annotate(f"Range: {round(range, 2)}, Mean: {round(mean, 2)}", xy=(0.05, 0.98),
                xycoords='axes fraction', fontsize=7)  # calculate the range of flight ranges for a particular experiment и среднее для сравнения с референсными значениями

fig.tight_layout()

sns.despine()
plt.subplots_adjust(wspace=0.3, hspace=0.5)  # increase spacing between subplots
plt.show()


data = []
for experiment in experiments:
    experiment_data = {
        'name': experiment.name,
        'angle_distribution': {
            'params': {
                'loc': angle_loc,
                'scale': angle_scale,
                'low': angle_low,
                'high': angle_high,
            },
            'distribution': experiment.angle.tolist()
        },
        'v0_distribution': {
            'params': {
                'loc': v0_loc,
                'scale': v0_scale,
                'low': v0_low,
                'high': v0_high
            },
            'distribution':  experiment.v0.tolist()
        },
        'L': {
            'distribution': experiment.range.tolist(),  # Convert array to list before saving
            'mean': float(np.mean(experiment.range)),
            'std': float(np.std(experiment.range)),
            'max': float(np.max(experiment.range)),
            'min': float(np.min(experiment.range))
        }
    }
    data.append(experiment_data)

with open('experiments.json', 'w') as f:
    json.dump(data, f, indent=4)

