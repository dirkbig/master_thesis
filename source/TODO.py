"""
TODO:

    Depletion Event comparison with and w/o EnergyBazaar sharing

    Parameter survey/sweep on battery size wrt depletion events

    Constraint to bidding prices for agents with a looming depletion event

    Play with battery prediction horizon: currently only around 70 steps: a hour.
        increase to 700 steps: 10 hours
        show prediction has a positive effect; if not in obvious data, what functionality is enabled by it?

    Show E_demand (mean/std) with or w/o EnergyBazaar: show that E_demand is bended
        expected: w/o EBZ; E_demand is only high when no production. with EBZ, E_demand should
        be higher during production (of consumers? or also of prosumers?)
        Maybe show as well with continuous load? Makes it more clear

    w_mean stabilizes at 0.8: show that when distortion or sudden higher/lower load, w_mean restabilizes to an other value
    probably run parameter sweep over lambda?

    X-axis instead of steps make it minutes (or time for that matter)

    Express EBZ performance in the amount of kwh imported from the grid:
    both with and w/o EBZ, depletion events arise, so in either case, MG has to import extra energy. With EBZ, this amount
    should be way lower. SHOW this and the welfare of a typical prosumers in both cases.
    Since zero-sum game, there are always winners and losers within the community. but the total welfare of the community can also be expressed

    Topology of Sergio Grammatico:  Network Aggregative Games and Distributed Mean Field Control via Consensus Theory

    Time-stamp state in the smart contract for async optimzation/ random possibility where agents do not update the blockchain
    creating asycn. then time-stamp can be used to either ignore that agent or optimize with it anyways

    show plots for consumers or prosumers:: averages cannot be taken between the two: seperate them



DONE:

    Extend number of agent to 100
    Shuffling agents at each optimization game (only microgrid_model for now)
    Max discharge rate constraint of batteries; implement this // ESSENTIAL!
    Utility function Sellers without prediction
    update w_sharing_factor of agents to their actual supplied energy: oversupply is a weird thing





"""


import glob
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from smartstart.agents.qlearning import QLearning
from smartstart.agents.smartstart import SmartStart

plt.rc('text', usetex=True)
sns.set()

from smartstart.utilities import Summary, calc_average_policy_in_training_steps, \
    calc_average_reward_training_steps


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import *

plt.rc('text', usetex=True)


def get_summaries(data):
    for strategy in data['data']:
        fp_data = os.path.join(BASE_DIR, 'smartstart*performance_stochastic', strategy['date'], data['env'])
        fps = glob.glob('%s/*%s*.json' % (fp_data, strategy['method']))
        if not fps:
            raise Exception('No summaries found for fp: %s/*%s*.json' % (fp_data, strategy['method']))
        summaries = [Summary.load(fp) for fp in fps]

        strategy['data_smart_start'] = []
        strategy['data'] = []
        for summary in summaries:
            if summary.agent.__class__ == SmartStart:
                strategy['data_smart_start'].append(summary)
            else:
                strategy['data'].append(summary)


def make_plot(plot_type, all_data, title, ylabel, scale, xlims, ylims, xticks=None, yticks=None, baselines=None):
    if xticks is None:
        xticks = [[None]]*len(all_data)
    if yticks is None:
        yticks = [[None]]*len(all_data)

    fig, axess = plt.subplots(len(all_data[0]['data']), len(all_data), figsize=(FIG_WIDTH, 1.3*len(all_data[0]['data'])))
    fig.set_label('%s_%s' % (os.path.basename(__file__).split('.')[0], title))
    fig.tight_layout(w_pad=0.2, h_pad=0.4)
    for data_env, axes, xlim, ylim, xtick, ytick in zip(all_data, axess.T, xlims, ylims, xticks, yticks):
        if type(axes) is not np.ndarray:
            axes = np.asarray([axes])

        xlim = np.asarray(xlim) / scale
        xtick = np.asarray(xtick) / scale
        for data, ax in zip(data_env['data'], axes):
            # SMART START
            x, mean_y, std = plot_type(data['data_smart_start'])
            x = np.insert(x, 0, 0)
            mean_y = np.insert(mean_y, 0, 0)
            std = np.insert(std, 0, 0)

            # Normalize average reward
            if baselines is not None:
                mean_y *= baselines[data_env['env']]
                std *= baselines[data_env['env']]

            # Scale training steps
            x_scaled = x / scale

            upper = mean_y + std
            lower = mean_y - std
            ax.fill_between(x_scaled, lower, upper, alpha=0.3, color=COLORS['smart_start'])
            ax.plot(x_scaled, mean_y, label='Smart Start', color=COLORS['smart_start'])

            # NORMAL
            x, mean_y, std = plot_type(data['data'])
            x = np.insert(x, 0, 0)
            mean_y = np.insert(mean_y, 0, 0)
            std = np.insert(std, 0, 0)

            # Normalize average reward
            if baselines is not None:
                mean_y *= baselines[data_env['env']]
                std *= baselines[data_env['env']]

            # Scale training steps
            x_scaled = x / scale

            upper = mean_y + std
            lower = mean_y - std
            ax.fill_between(x_scaled, lower, upper, alpha=0.3, color=COLORS['normal'])
            ax.plot(x_scaled, mean_y, linestyle='dashed', label='Normal', color=COLORS['normal'])

            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            if xtick is not None:
                ax.set_xticks(xtick)
            if ytick is not None:
                ax.set_yticks(ytick)

            if baselines is not None:
                ax.hlines(y=1., xmin=xlim[0], xmax=xlim[-1], color="black",
                          linestyle="dotted", label='Optimal Solution')

        axes[0].set_title(r'\textbf{%s}' % data_env['env'])

    label = r'Training Steps'
    if scale != 1:
        label += r' ($\times %d$)' % scale
    for ax in axess[-1, :]:
        ax.set_xlabel(label)

    if type(axess[0]) is np.ndarray:
        ax = axess[0][-1]
    else:
        ax = axess[-1]
    ax.legend(loc='lower right', bbox_to_anchor=(1, 1.1), ncol=3)

    for ax, data in zip(axess[:, 0], all_data[0]['data']):
        ax.set_ylabel(ylabel)
        y_pos = (ax.get_position().y0 + ax.get_position().y1) / 2
        x_pos = ax.get_position().x0 - 0.08
        fig.text(x_pos, y_pos, r'\textbf{%s}' % data['label'], rotation=90, va='center', ha='right')


def main(envs, baselines):
    all_data = []
    data_qlearning = []
    for env in envs:
        data = {
            'env': env,
            'data': [{
                'method': QLearning.E_GREEDY,
                'date': '31-01-2018',
                'title': r'QLearning + $\epsilon$-greedy',
                'label': r'$\epsilon$-greedy'
            }, {
                'method': QLearning.BOLTZMANN,
                'date': '31-01-2018',
                'title': r'QLearning + Boltzmann',
                'label': r'Boltzmann'
            }, {
                'method': QLearning.UCB1,
                'date': '31-01-2018',
                'title': r'QLearning + UCB1',
                'label': r'UCB1'
            }]
        }
        get_summaries(data)
        data_qlearning.append(data)
    all_data += data_qlearning

    scale = 1000
    xlims = [[0, 50000], [0, 100000], [0, 200000]]
    xticks = [range(0, 50001, 10000), range(0, 100001, 20000), range(0, 200001, 40000)]

    ylims = [[-0.05, 1.0]]*3
    yticks = [[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]]*3
    make_plot(calc_average_policy_in_training_steps, data_qlearning, 'qlearning_policy', r'$\pi/\pi^*$',
                        scale, xlims, ylims, xticks, yticks)

    # ylims = [[-0.005, 0.05], [-0.005, 0.025], [-0.005, 0.025]]
    # yticks = [np.arange(0.00, 0.06, 0.01), np.arange(0.00, 0.03, 0.01), np.arange(0.00, 0.03, 0.01)]
    ylims = [[-0.2, 1.2]] * 3
    yticks = [np.arange(0., 1.05, 0.2)] * 3
    make_plot(calc_average_reward_training_steps, data_qlearning, 'qlearning_reward', r'Average Reward',
                        scale, xlims, ylims, xticks, yticks, baselines)

    data_rmax = []
    for data_env in all_data:
        data = {
            'env': data_env['env'],
            'data': [{
                'method': 'MBRL',
                'date': '11-02-2018',
                'title': r'MBRL',
                'label': r'MBRL'
            }, {
                'method': 'RMax',
                'date': '08-02-2018',
                'title': r'\textsc{R-max}',
                'label': r'\textsc{R-max}'
            }]
        }
        get_summaries(data)
        data_rmax.append(data)
        data_env['data'] += data['data']

    scale = 1000
    xlims = [[0, 10000], [0, 15000], [0, 25000]]
    xticks = [range(0, 10001, 2000), range(0, 15001, 3000), range(0, 25001, 5000)]

    ylims = [[-0.05, 1.0]] * 3
    yticks = [[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]] * 3
    make_plot(calc_average_policy_in_training_steps, data_rmax, 'rmax_policy', r'$\pi / \pi^*$',
              scale, xlims, ylims, xticks, yticks)

    # ylims = [[-0.005, 0.05], [-0.005, 0.025], [-0.005, 0.025]]
    # yticks = [np.arange(0.00, 0.06, 0.01), np.arange(0.00, 0.03, 0.01), np.arange(0.00, 0.03, 0.01)]
    ylims = [[-0.2, 1.2]] * 3
    yticks = [np.arange(0., 1.05, 0.2)] * 3
    make_plot(calc_average_reward_training_steps, data_rmax, 'rmax_reward', r'Average Reward',
              scale, xlims, ylims, xticks, yticks, baselines)


if __name__ == '__main__':
    envs = ['Easy', 'Medium', 'Maze']

    baselines = {
        'Easy': 19,
        'Medium': 35,
        'Maze': 40
    }

    main(envs, baselines)

    plt.show()