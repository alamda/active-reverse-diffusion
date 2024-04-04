import sys

sys.path.insert(0, '../src/')

import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

import argparse

from plotter import Plotter


def sample_reverse_time(sim_object=None, time=None):
    sim_object.sample_from_diffusion_passive(time=time)
    sim_object.calculate_passive_diff_list()

    sim_object.sample_from_diffusion_active(time=time)
    sim_object.calculate_active_diff_list()

    t_passive = sim_object.passive_reverse_time_arr
    d_passive = sim_object.passive_diff_list

    t_active = sim_object.active_reverse_time_arr
    d_active = sim_object.active_diff_list

    ymax = 0.5

    time_str = str(time).ljust(4, '0')

    plotter = Plotter(diffusion_object=sim_object)
    plotter.plot_sample_hist_pre_diffusion(png_fname=f"hist_pre_{time_str}.png",
                                           ymax=ymax)

    plotter.plot_sample_hist_post_diffusion(png_fname=f"hist_post_{time_str}.png",
                                            title=f"sample after reverse diffusion from t={time_str}",
                                            ymax=ymax)

    # plotter.plot_hist_animation(mp4_fname=f"hist_{time}.mp4",
    #                             ymax=ymax)

    return (t_passive, d_passive, t_active, d_active)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', required=True)
    parser.add_argument('--xmin', type=float)
    parser.add_argument('--xmax', type=float)
    parser.add_argument('--num_hist_bins', type=int)
    parser.add_argument('--log_kl_min', type=int)
    parser.add_argument('--log_kl_max', type=int)

    args = parser.parse_args()

    if os.path.isfile(args.filename):
        with open(args.filename, 'rb') as f:
            mydiff = pickle.load(f)

        if args.xmin is not None:
            mydiff.data_proc.xmin = args.xmin

        if args.xmax is not None:
            mydiff.data_proc.xmax = args.xmax

        if args.num_hist_bins is not None:
            mydiff.data_proc.num_hist_bins = args.num_hist_bins

        diff_dict = {}

        time_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ] #, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

        # time_list = [0.25, 0.30, 0.35, 0.40, 0.45,
        #              0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

        for time in time_list:
            tp, dp, ta, da = sample_reverse_time(sim_object=mydiff, time=time)

            time_dict = {'time_passive': tp,
                         'diff_passive': dp,
                         'time_active': ta,
                         'diff_active': da}

            diff_dict[time] = time_dict

            print(f"{time}s reverse diffusion done")

        fig, ax = plt.subplots()

        times_to_plot = [0.2, 0.4, 0.6, 0.8] #, 1.0, 1.5, 2.0]

        for time in times_to_plot:  # diff_dict.keys():
            if time in time_list:
                time_dict = diff_dict[time]

                color = next(ax._get_lines.prop_cycler)['color']
                label = f"time = {time}"

                ax.plot(-1*time_dict['time_passive'], np.log(time_dict['diff_passive']),
                        color=color, linestyle='dashed',
                        label=f"{label} (passive)")

                ax.plot(-1*time_dict['time_active'], np.log(time_dict['diff_active']),
                        color=color,
                        label=f"{label} (active)")

        ax.legend()
        ax.set_xlabel("(reverse) time")
        ax.set_ylabel("Log(KL-Divergence)")

        ax.set_xlim((-0.5, 0))

        plt.savefig('reverse_sampling.png')

        plt.close(fig)

        fig, ax = plt.subplots()

        passive_data_list = []
        active_data_list = []

        for time in diff_dict.keys():
            time_dict = diff_dict[time]

            passive_data_list.append([time, time_dict['diff_passive'][-1]])
            active_data_list.append([time, time_dict['diff_active'][-1]])

        passive_data_arr = np.array(passive_data_list)
        active_data_arr = np.array(active_data_list)

        ax.scatter(passive_data_arr[:, 0], np.log(passive_data_arr[:, 1]),
                   facecolors='none', edgecolors='blue',
                   label="passive")

        ax.scatter(active_data_arr[:, 0], np.log(active_data_arr[:, 1]),
                   color='orange',
                   label="active")

        ax.legend()
        ax.set_xlabel("(reverse) time")
        ax.set_ylabel("Log(KL-Divergence)")
        ax.set_ylim((args.log_kl_min, args.log_kl_max))

        plt.savefig('reverse_kl.png')

        plt.close(fig)
