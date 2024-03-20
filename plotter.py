import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import functools


class Plotter:
    def __init__(self, diffusion_object=None):
        self.diffusion_object = diffusion_object

    def plot_loss(self, ymin=0, ymax=None, png_fname='loss.png'):
        fig, ax = plt.subplots()

        time_arr_passive = np.array([
            i*self.diffusion_object.dt for i in range(len(self.diffusion_object.passive_loss_history))])

        time_arr_active_x = np.array([
            i*self.diffusion_object.dt for i in range(len(self.diffusion_object.active_loss_history_x))])

        time_arr_active_eta = np.array(
            [i*self.diffusion_object.dt for i in range(len(self.diffusion_object.active_loss_history_eta))])

        ax.scatter(time_arr_passive,
                   self.diffusion_object.passive_loss_history, label="Passive Loss")
        ax.scatter(time_arr_active_x,
                   self.diffusion_object.active_loss_history_x, label="Active Loss, x")
        ax.scatter(time_arr_active_eta,
                   self.diffusion_object.active_loss_history_eta, label="Active Loss, eta")

        ax.legend()

        if ymax is not None:
            ax.set_ylim((ymin, ymax))

        plt.savefig(png_fname)

        plt.close(fig)

    def plot_sample_hist(self, idx=None, png_fname=None, ymax=None, title=None):

        if png_fname is None:
            png_fname = f'hist{str(idx).zfill(3)}'

        fig, ax = plt.subplots()

        if title is None:
            time = float(self.diffusion_object.active_reverse_time_arr[idx])
            ax.set_title("time = %.2f" % round(time, 2))
        else:
            ax.set_title(title)

        target_sample = self.diffusion_object.target.sample.flatten()
        passive_sample = self.diffusion_object.passive_reverse_samples[idx].flatten(
        )
        active_sample = self.diffusion_object.active_reverse_samples_x[idx].flatten(
        )

        num_hist_bins = self.diffusion_object.data_proc.num_hist_bins

        xmin = self.diffusion_object.data_proc.xmin
        xmax = self.diffusion_object.data_proc.xmax

        hist, bins, _ = ax.hist((passive_sample, active_sample, target_sample),
                                bins=num_hist_bins,
                                density=True,
                                label=['passive', 'active', 'target'],
                                histtype='step',
                                fill=False,
                                alpha=1,
                                range=(xmin, xmax))

        new_bins = (bins[1:] + bins[:-1])/2

        ax.legend()
        if ymax is not None:
            ax.set_ylim((0, ymax))

        plt.savefig(png_fname)
        plt.close(fig)

    def plot_sample_hist_post_diffusion(self, png_fname="hist_post.png", ymax=None, title=None):
        self.plot_sample_hist(idx=-1, png_fname=png_fname,
                              ymax=ymax, title=title)

    def plot_sample_hist_pre_diffusion(self, png_fname="hist_pre.png", ymax=None, title=None):
        self.plot_sample_hist(idx=0, png_fname=png_fname,
                              ymax=ymax, title=title)

    def plot_KL_diffusion(self, png_fname="KL_learning.png"):
        fig, ax = plt.subplots()

        ax.set_title("KL Divergence of Diffusion")

        passive_diff_list = self.diffusion_object.passive_diff_list
        active_diff_list = self.diffusion_object.active_diff_list

        t_list_passive = np.arange(
            0, len(passive_diff_list))*self.diffusion_object.dt
        t_list_active = np.arange(
            0, len(active_diff_list))*self.diffusion_object.dt

        ax.set_xlabel("Time")
        ax.set_ylabel("Log(KL-Divergence)")

        ax.plot(t_list_passive, np.log(passive_diff_list), label="passive")
        ax.plot(t_list_active, np.log(active_diff_list), label="active")

        ax.legend()

        plt.savefig(png_fname)

        plt.close(fig)

    def update_hist(self, frame_idx, bar_container=None, ax=None, passive_data=None, active_data=None, num_bins=None, label=None):

        xmin = self.diffusion_object.data_proc.xmin
        xmax = self.diffusion_object.data_proc.xmax

        time = frame_idx * self.diffusion_object.dt

        ax.set_title("time = %.2f" % round(time, 2))

        passive_data = passive_data[frame_idx].reshape(
            passive_data[frame_idx].shape[0])
        active_data = active_data[frame_idx].reshape(
            active_data[frame_idx].shape[0])

        lfig, lax = plt.subplots()

        _, _, bar_container_passive = lax.hist(passive_data,
                                               bins=num_bins,
                                               density=True,
                                               histtype='step',
                                               fill=False,
                                               range=(xmin, xmax))

        _, _, bar_container_active = lax.hist(active_data,
                                              bins=num_bins,
                                              density=True,
                                              histtype='step',
                                              fill=False,
                                              range=(xmin, xmax))

        bar_container[0][0].set_xy(bar_container_passive[0].get_xy())
        bar_container[1][0].set_xy(bar_container_active[0].get_xy())

        plt.close(lfig)

        return bar_container

    def plot_hist_animation(self, num_bins=None, ymax=None, mp4_fname='hist.mp4'):
        if num_bins is None:
            num_bins = self.diffusion_object.data_proc.num_hist_bins

        fig, ax = plt.subplots()

        if ymax is not None:
            ax.set_ylim((0, ymax))

        target_data = self.diffusion_object.target.sample
        passive_data = self.diffusion_object.passive_reverse_samples
        active_data = self.diffusion_object.active_reverse_samples_x

        len_data = len(passive_data)

        xmin = self.diffusion_object.data_proc.xmin
        xmax = self.diffusion_object.data_proc.xmax

        # _, _, bar_container_passive = ax.hist((passive_data[0].reshape(passive_data[0].shape[0]),
        #                                        active_data[0].reshape(active_data[0].shape[0]),
        #                                        target_data.reshape(target_data.shape[0])),
        #                                       bins=num_bins,
        #                                       label=['passive', 'active', 'target'])

        _, _, bar_container = ax.hist((passive_data[0].reshape(passive_data[0].shape[0]),
                                       active_data[0].reshape(
            active_data[0].shape[0]),
            target_data.reshape(target_data.shape[0])),
            bins=num_bins,
            label=['passive',
                   'active', 'target'],
            density=True,
            histtype='step',
            fill=False,
            alpha=1,
            range=(xmin, xmax))

        anim_passive = functools.partial(self.update_hist,
                                         ax=ax,
                                         bar_container=bar_container,
                                         passive_data=passive_data,
                                         active_data=active_data,
                                         num_bins=num_bins,
                                         label='passive')

        ani = animation.FuncAnimation(fig,
                                      anim_passive,
                                      len_data,
                                      repeat=False)

        FFwriter = animation.FFMpegWriter(fps=10)
        ani.save(mp4_fname, writer=FFwriter)
