import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from matplotlib.pyplot import cm
from matplotlib.widgets import RangeSlider, Slider

import torch
import math
class Metrics:
    def __init__(self, fsamp, tol=6) -> None:
        """
        Args:
            fsamp (float): Sampling frequenze in hz.
            L_waveform (int): Lenth of the waveform in unit.
            tol (int, optional): Tolerance for discharge time in ms. Defaults to 6.
        """
        self.fsamp = fsamp
        self.tol = tol
        pass

    def evaluate(self, sts_eval, sts_ref, max_shift=100):
        """Caculate the cross efficience for each spike train among the reference spike trains.
        The output rho is the highest cross corelation efficience, j is the index of the referece
        spike train and t is the time shift.

        Args:
            sts_eval (torch.tensor): spike trains to be evaluated.
            sts_ref (torch.tensor): Groups of reference spike trains, where raws are variable, columns
            are samples.
            max_shift (int, optional): Max time shift in unit. Defaults to 100.
        """
        pad = max_shift
        window = int(self.tol / 1e3 * self.fsamp)
        sts_ref = sts_ref.double()
        sts_ref = F.max_pool1d(sts_ref, window,stride=window)
        sts_ref = sts_ref / (sts_ref.std(dim=-1,keepdim=True))
        rhos, js, ts = [], [], []
        for st in sts_eval:
            st = F.max_pool1d(st.unsqueeze(0), window,stride=window)
            st = st / (st.std(dim=-1,keepdim=True))
            cross_cors =  F.conv1d(sts_ref, st.expand(sts_ref.shape[-2], -1).view(-1,1,st.shape[-1]), 
                                padding=pad, groups=sts_ref.shape[-2])
            rho, jt = cross_cors.max(dim=-1)
            rho ,j = rho.max(dim=-1)
            rho = rho / st.shape[-1]
            t = jt[j]-pad+1
            rhos.append(rho)
            js.append(j)
            ts.append(t)
        return torch.stack(rhos), torch.stack(js), torch.stack(ts)
                

    @staticmethod
    def plot_spike_raster(spike_trains, start=0, end=None):
        """Gernerate raster plot for the input spike trains.

        Args:
            spike_trains (torch.tensor): input spike trains. Rows are variables, column are
            samples.
            start (in, optional): start position of the plot. Defaults to 0.
            end (int, optional): end position of the plot. Defaults to None.
        """
        fig,ax =plt.subplots()
        color = cm.rainbow(np.linspace(0, 1, spike_trains.shape[0]))
        plt.eventplot([s.nonzero().flatten() for s in spike_trains],colors=color)
        fig.subplots_adjust(bottom=0.2)
        slider_ax = fig.add_axes([0.20, 0.1, 0.60, 0.03])
        if end == None:
            end = ax.get_xlim()[-1]
        slider = RangeSlider(slider_ax, "Time", valmin=start, valmax=end, valstep=1.)

        def update(val):
            ax.set_xlim(int(slider.val[0]), int(slider.val[1] + 10))
            # Redraw the figure to ensure it updates
            fig.canvas.draw_idle()


        slider.on_changed(update)
        plt.show()

    def plot_muap_map(self, X, L_waveform, ind):
        """Plot signals of all channels for the multi-channels signal X.

        Args:
            X (torch.tensor): Multi-channels signal to plot. Rows are channels, column are
            samples.
            L_waveform (int): Lenth of the waveform in unit.
            ind (int): Index of the waveform to plot.
        """
        nrow = math.isqrt(X.shape[0])
        ncol = nrow
        y_min, y_max = 1.2 * torch.min(X), 1.2 * torch.max(X)
        fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=(5*nrow, 5*ncol), sharex=True, sharey=True)
        fig.subplots_adjust(bottom=0.2)
        lines=[]
        for ax, signal in zip(axs.reshape(-1),X):
            ax.set_ylim(y_min, y_max)
            line, = ax.plot(signal[int(ind*L_waveform):int((ind+1) * L_waveform)])
            lines.append(line)
        slider_ax = fig.add_axes([0.20, 0.1, 0.60, 0.03])
        slider = Slider(
                ax=slider_ax,
                label="position",
                valmin=0,
                valmax=X.shape[-1],
                valinit=ind,
                orientation="horizontal"
                )

        def update(val):
            pos = slider.val
            # axs.cla()
            for line, signal in zip(lines,X):
                line.set_ydata(signal[int(pos):int(pos + self.L_waveform)])
                # ax.set_xlim(pos,pos + self.L_waveform)
            fig.canvas.draw_idle()
        
        slider.on_changed(update)
        plt.show()

    def plot_time_serise(self, st_ref, sig_fica, st_fica, sig_cica, st_cica,thh):
        """Plot siganls and stack them vertically.

        Args:
            ref_ST (torch.tensor): Reference spike train.
            sig_fICA (torch.tensor): signal of fast ICA.
            fICA_ST (torch.tensor): Spike train of fast ICA.
            sig_cICA (torch.tensor): signal of constrained ICA
            ST_cICA (troch.tensor): spike train of constrained ICA.
            thh (float): threshold to extract spike trains.
        """
        fig,axs = plt.subplots(5,1, figsize=(8, 10), sharex=True)
        axs[0].eventplot(st_ref.nonzero().flatten())
        axs[1].plot(sig_fica)
        axs[1].axhline(y=thh, color='r', linestyle='-')
        axs[2].eventplot(st_fica.nonzero().flatten())
        axs[3].plot(sig_cica)
        axs[3].axhline(y=thh, color='r', linestyle='-')
        axs[4].eventplot(st_cica.nonzero().flatten())

        axs[0].set_title("Reference spike train")
        axs[1].set_title("Fast ICA signal")
        axs[2].set_title("Fast ICA spike train")
        axs[3].set_title("Constrained ICA signal")
        axs[4].set_title("Constrained ICA spike train")
        
        plt.tight_layout()
        plt.show()
    
    def f1_score(self, st_eval, sts_ref, max_shift=100):
        """
        Compute the F1 score between a spike train among a set of reference spike trains.
        Args:
            st_eval (torch.tensor): spike train to evaluate.
            sts_ref (torch.tensor): reference spike trains.
            max_shift (int, optional): maximum shift to consider. Defaults to 100.
        Returns:
            torch.tensor: F1 score.
        """
        pad = max_shift
        window = int(self.tol / 1e3 * self.fsamp)
        st_eval = F.max_pool1d(st_eval.unsqueeze(0), window,stride=window)
        sts_ref = sts_ref.double()
        sts_ref = F.max_pool1d(sts_ref, window,stride=window)
        cross_cors =  F.conv1d(sts_ref, st_eval.expand(sts_ref.shape[-2], -1).view(-1,1,st_eval.shape[-1]), 
                            padding=pad, groups=sts_ref.shape[-2])
        rho, jt = cross_cors.max(dim=-1)
        rho ,j = rho.max(dim=-1)
        # rho = rho / st_eval.shape[-1]
        # t = jt[j]-pad+1
        tp = torch.max(cross_cors)
        # j, _ = torch.argmax(cross_cors)
        f1 = 2* tp /(sts_ref[j].count_nonzero() + st_eval.count_nonzero())
        return f1
    
    def roa(self, st_eval, sts_ref, max_shift=100):
        """Compute the rate of agreement (ROA) between a spike train to be evaluate
        st_eval among all the reference spike trains sts_ref
        
        Args:
            st_eval (torch.tensor): Spike train to be evaluated.
            sts_ref (torch.tensor): Reference spike trains.
            max_shift (int, optional): Maximum shift for the evaluation. Defaults to 100.

        Returns:
            torch.tensor: Rate of agreement.
        """
        pad = max_shift
        window = int(self.tol / 1e3 * self.fsamp)
        st_eval = F.max_pool1d(st_eval.unsqueeze(0), window,stride=window)
        sts_ref = sts_ref.double()
        sts_ref = F.max_pool1d(sts_ref, window,stride=window)
        cross_cors =  F.conv1d(sts_ref, st_eval.expand(sts_ref.shape[-2], -1).view(-1,1,st_eval.shape[-1]), 
                            padding=pad, groups=sts_ref.shape[-2])
        rho, jt = cross_cors.max(dim=-1)
        rho ,j = rho.max(dim=-1)
        # rho = rho / st_eval.shape[-1]
        # t = jt[j]-pad+1
        tp = torch.max(cross_cors)
        # j, _ = torch.argmax(cross_cors)
        f1 = 2* tp /(sts_ref[j].count_nonzero() + st_eval.count_nonzero())
        return f1


def dataloader(path):
    df = pd.read_csv(path)
    data = df.iloc[:, 1:].values
    data = torch.tensor(data).T
    return data



