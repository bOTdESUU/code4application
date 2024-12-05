"""
This file contains the base class of the EMG decomposition using PFP framework.
"""
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import pickle
import time
from typing import Union



class fastica_decomposer:
    
    def __init__(self, emg_signal:torch.tensor, n_components=10, fsamp=2000, L=25, K=15, thh=4., max_iter=200, tol=1e-4, sym=False) -> None:
        """
        Args:
            emg_signal (torch.tenor): raw EMG signal to be decomposed
            n_components (int, optional): Number of independent components. Defaults to 10.
            fsamp (int, optional): Sampling rate in hz. Defaults to 2000.
            L (int, optional): Length of waveform in ms. Defaults to 25.
            K (int, optional): Delay used in the data model. Defaults to 15.
            thh (float, optional): Threshold used to extract spikes for the signal. Defaults to 4.
            max_iter (int, optional): Maximum iteration in fast ICA. Defaults to 200.
            tol (float, optional): Tolerant of convergence in fastICA. Defaults to 1e-4.
            sym (bool, optional): determine if using the symetric fast ICA.
        """
        global DEVICE 
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.raw_sig = emg_signal.to(DEVICE)
        self.fsamp = fsamp
        self.K = int(K * fsamp / 1000)
        self.L = int(L * fsamp / 1000)
        self.n_components_init = n_components
        self.n_components = n_components
        self.n_channel = emg_signal.shape[0]
        self.n_samples = emg_signal.shape[1]
        self.max_iter = max_iter
        self.tol = tol
        self.thh = thh
        self.max_iter_fICA = max_iter
        self.max_iter_cICA = 20
        self.w_cros_cor = int(K  * fsamp / 1000)
        self.sym = sym
        torch.set_grad_enabled(False)

    def decompose(self) -> dict:
        """This function decompose the raw EMG signal into independent components using the PFP framework.

        Returns:
            dict: A dictionary containing the decomposed spike trains and other information.
        """
        self.start_time = time.time()
        raw_signal_ext = self._extend(self.raw_sig.to(DEVICE), self.K-1)
        raw_signal_ext_w = self._whiten(raw_signal_ext)
        del raw_signal_ext
        self.no_update = 0
        self.sts_fICA = []
        self.sig_fICA = []
        self.sig_cICA = []
        res_signal_ext_w = self._whiten(self._extend(self.raw_sig.to(DEVICE), self.K - 1),sym=self.sym)

        while True:
            valid_st = 0
            if self.sym ==True:
                sig_fICA = self._fastICA(res_signal_ext_w[0:self.n_components], sym=self.sym)
            else:
                sig_fICA = self._fastICA(res_signal_ext_w, sym=self.sym)
            sts_fICA = self._extract_spikes(sig_fICA)
            if sts_fICA.count_nonzero() == 0: #number of non zero
                break
            elif self.no_update > 3 :
                if hasattr(self, 'phi'):
                    break
                elif self.no_update > 5:
                    break 
            else:
                for st_fICA, s_fICA in zip(sts_fICA, sig_fICA):
                    if st_fICA.count_nonzero() == 0:
                        continue
                    ref = self._regularize_ref(st_fICA)
                    sig_cICA = self._cICA(raw_signal_ext_w, ref)
                    if sig_cICA != None:
                        st_cICA = self._extract_spikes(sig_cICA)
                        if self._test_duplicate(st_cICA):
                            self._update_st(st_cICA)
                            # self.sel_ic.append(st_cICA.cpu())
                            self.sig_cICA.append(sig_cICA.cpu())
                            self.sig_fICA.append(s_fICA.cpu())
                            self.sts_fICA.append(st_fICA.to_sparse().cpu())
                            valid_st += 1
            if valid_st != 0:
                self._muap_estimate()
                self.no_update = 0
                res_signal_ext_w = self._whiten(self._extend(self.res_sig, self.K - 1),sym=self.sym)
                del self.res_sig
            else:
                self.no_update += 1
            self._update_params()
        return self._generate_output()
    
    def _fastICA(self, X, sym=False) -> torch.tensor:
        """This Function apply the parallel version of fastICA on the given inputs and return the independent
        components decomposed by the fastICA.

        Args:
            X (torch.tensor): 2-dimensional tensor of the signal to decompose, where the raws are variable and
            columns are samples.

        Returns:
            torch.tensor: 2-dimensional tensor of decomposed independent components with shape (n_componets,
            n_samples).
        """
        if sym == True:
            W = self._decorrelation(torch.rand(self.n_components, self.n_components, dtype=torch.float64,device=DEVICE).T).T
        else:
            W = self._decorrelation(torch.rand(X.shape[0], self.n_components, dtype=torch.float64,device=DEVICE).T).T
        for _ in range(self.max_iter_fICA):
            G = torch.tanh(W.T @ X) #(components, samples)
            G_grad = (1. - G ** 2).mean(axis=-1) #n_component
            W1 =  (X @ G.T) / self.n_samples - (G_grad.unsqueeze(0) * W)
            W1 = self._decorrelation(W1.T).T
            lim = max(abs(abs(torch.einsum("ij,ij->i", W1.T, W.T)) - 1))
            W = W1
            if lim < self.tol:
                break
        return W.T @ X
    
    def _cICA(self, X, ref) -> Union[torch.tensor, None]:
        """This function apply constrained ICA algorithm on the input signal with the given reference signal.

        Args:
            X (torch.tensor):  2-dimensional tensor of the signal to decompose, where the raws are variable and
            columns are samples.
            ref (torch.tensor): reference signal to constrain on X.

        Returns:
            Union[torch.tensor, None]: 2-dimensional tensor of decomposed independent components with shape (n_componets
            , n_samples) or None if rejected.
        """
        #ref - (1, n_samples), X - (n_channel, n_samples)
        mu = 1.
        gamma = 0.1
        xi = 1.
        w = torch.rand(X.shape[0], 1, dtype=torch.float64, device=DEVICE)
        w = w / torch.norm(w)
        ref = ref.unsqueeze(0)
        for _ in range(self.max_iter_cICA):
            y = w.T @ X #(1,samples)
            G = torch.tanh(y)
            cor = (y * ref).mean(-1)
            g = xi - cor # y.T@r? average by the last axis, scalar
            g_grad = - ref # shape(1,n_samples)
            w1 = (X @ G.T) / self.n_samples - mu * (X * g_grad).mean(axis=-1).unsqueeze(-1) # ()
            w1 = w1 / torch.norm(w1)
            mu = max(0, mu + gamma * g)
            lim = abs(1. - torch.norm(w.T @ w1))
            if lim < 1e-6 and g < 0:

                return y.squeeze()
            w = w1
            xi *= 0.97
            gamma *= 4
        return None

    def _muap_estimate(self) -> None:
        """This function estimate the MUAPs from the selected independent components
        by constructing S_hat. S_hat have a shape of (n_smaples, q*L), where q is the number of
        reliable components and L is the length of the Wavelength.
        Each spike in phi.T result in a diag 1 in S_hat with length L.
        """

        S_hat = self._extend(self.phi, self.L).T #shape(n_samples, q*L)
        self.waveForms = torch.inverse((S_hat.T @ S_hat)) @ S_hat.T @ self.raw_sig.T # shape(q*L, n_channels)
        recov_signal = (S_hat @ self.waveForms).T
        self.res_sig = self.raw_sig - recov_signal
        pass

    def _whiten(self, X, sym=False) -> torch.tensor:
        """This function whiten the input signal.

        Args:
            X (torch.tensor): 2-dimensional tensor of the signal to whiten, where the rows are variable and
            columns are samples.

        Returns:
            torch.tensor: 2-dimensional tensor of whitened signal with shape (n_componets, n_samples).
        """
        D = X - torch.mean(X, -1, True)
        del X
        cov = torch.cov(D,correction=0)
        L, V = torch.linalg.eigh(cov)
        # sorted_indices = torch.argsort(L)
        # L = L[sorted_indices[0:self.n_components]]
        # V = V[sorted_indices[:, 0:self.n_components]]
        L = L.flip(-1)
        V = V.flip(-1)
        L = torch.clip(L, min=torch.finfo(D.dtype).tiny, max=None)
        diagL = torch.diag(torch.pow(L, -0.5)).to_sparse()
        Z = diagL @ V.T @ D  #PCA
        return Z
    
    def _decorrelation(self, W) -> torch.tensor:
        """This function decorrelate the weight matrix in FastICA.

        Args:
            W (torch.tensor): weight matrix to decorrelate.

        Returns:
            torch.tensor: decorelated weight matrix.
        """
        L, V = torch.linalg.eigh(W @ W.T)
        L = torch.clip(L, min=torch.finfo(W.dtype).tiny, max=None)
        diagL = 1. / L.sqrt()
        W = V @ torch.diag(diagL) @ V.T @ W
        return W

    def _extract_spikes(self, X) -> torch.tensor:
        """Extract binary spike train from input signal. Use a max pooling mask to avoid duplicate spike in short time.

        Args:
            X (torch.tensor): Input signal to extract spike train.

        Raises:
            Exception: no spike can be extracted

        Returns:
            torch.tensor: binary spike train.
        """
        if X.dim() != 2:
            X = X.view(-1,X.shape[-1])
        # X = X.abs()
        mask = F.max_pool1d(X, self.L // 4 * 2 + 1, stride=1, padding=self.L // 4)
        mask = torch.clamp(mask, min=self.thh)
        sts = torch.where(X < mask, 0., 1).double()

        # if all(sts.count_nonzero(dim=-1)) == 0:
        #     raise Exception
        return sts

    def _regularize_ref(self, ref) -> torch.tensor:
        #regularize r so that E[r^2] = 1
        amp = self.n_samples / (ref.count_nonzero())
        return ref * torch.sqrt(amp)

    def _update_st(self, st) -> None:
        """Save and update the reliable spike train.

        Args:
            st (torch.tensor): spike train to update.
        """
        #update retrived reliable spike in self.phi
        # spike = self._extract_spikes(spike)
        if st.dim() != 2 :
            st = st.unsqueeze(0)
        if st.count_nonzero() == 0:
            raise Exception
        try:
            self.phi = torch.cat((self.phi, st))
        except:
            self.phi = st


    def _test_duplicate(self, X) -> bool:
        """Test the duplication of the extracted spike train.

        Args:
            X (torch.tensor): Input spike train to test.

        Returns:
            bool: True if the spike train is duplicate, False otherwise.
        """
        try:
            self.phi
        except:
            return True
        cross_cor = self._cross_cor(X)
        if torch.any(cross_cor > 0.5):
            return False
        else:
            return True

    def _update_params(self) -> None:
        """Update the parameters of the algorithm dynamically.
        """
        #update params dynamically e.g. n_components
        try:
            n_sts = self.phi.shape[0]
        except:
            n_sts = 0
        if self.sym == True:
            self.n_components = max(1, self.n_components_init - n_sts + 5*self.no_update**1.2)
        else:
            self.n_components = max(1, self.n_components_init - n_sts + self.no_update)

        self.n_components = int(min(self.n_components, self.n_channel))

    def _generate_output(self) -> dict:
        """Generate decomposition results as a dictionary.

        Returns:
            dict: dictionary of decomposition results.
        """
        STs = self.phi.cpu()
        output={}
        output['sample_rate'] = self.fsamp
        output['spike_trains'] = STs.to_sparse()
        output['spike_times'] = self._spike_time(STs)
        output['spike_trains_fICA'] = torch.stack(self.sts_fICA)
        output['signal_fICA'] = torch.stack(self.sig_fICA)
        output['signal_cICA'] = torch.stack(self.sig_cICA)
        output['delay'] = self.K
        output['waveform_length'] = self.L
        output['threshold'] = self.thh
        output['waveforms'] = self.waveForms.cpu()
        output['run_time'] = time.time() - self.start_time
        return output

    def _spike_time(self, STs:torch.tensor) -> torch.tensor:
        """Convert spike train to spike time.

        Args:
            STs (torch.tensor): spike train.

        Returns:
            torch.tensor: spike time.
        """
        return STs.to_sparse().indices().T

    @staticmethod
    def _extend(X, n) -> torch.tensor:
        #extend x by n position
        zero_pad = torch.nn.ZeroPad2d((n - 1, 0, 0, 0))
        X = zero_pad(X)
        filters = torch.eye(n, device=DEVICE).flip(-1).unsqueeze(0).expand(X.shape[0], -1, -1).double().view(-1, 1, n)
        return F.conv1d(X, filters, groups=X.shape[-2])

    def _cross_cor(self, X) -> torch.tensor:
        """Caculate the cross correaltion coefficient of the input X and all other retrived reliable spike trains.

        Args:
            X (torch.tensor): Input spike train.

        Returns:
            torch.tensor: cross correalation coeffient with shape(n_rST, 2 * shift_windows)
        """
        X = X / X.std()
        cross_cors =  F.conv1d(self.phi, X.expand(self.phi.shape[-2], -1).view(-1,1,X.shape[-1]),
                            padding=self.w_cros_cor, groups=self.phi.shape[-2])
        cross_cors /=  (self.phi.std(dim=-1,keepdim=True))
        cross_cors /= X.shape[-1]
        return cross_cors



def dataloader(path) -> torch.tensor:
    #dummy function to load data
    df = pd.read_csv(path)
    data = df.iloc[:, 1:].values
    data = torch.tensor(data).T
    return data

import pickle
def save_result(output, name) -> None:
    with open('result/' + name, 'wb') as f:
        pickle.dump(output, f)