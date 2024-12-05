# FastICA Peel-Off Algorithm for EMG Decomposition (PyTorch implementation)

## Overview

This project is an implementation of FastICA Peel-Off algorithm for EMG decomposition. The algorithm is based on the paper [A Novel Framework Based on FastICA for High Density Surface EMG Decomposition](https://pubmed.ncbi.nlm.nih.gov/25775496/). The algorithm is implemented in Python 3.8 and Pytorch. GPU acceleration is automatically enabled if available.

## Dependencies

* Python >= 3.8
* Numpy
* Scipy
* Matplotlib
* Pytorch >= 1.8.1

## Usage

### Load sEMG data
```python
path = '../Data/ramp_10mu_20s/emg_data.csv'
raw_signal = base.dataloader(path)
```

### Decompose
```python
dec = base.fastica_decomposer(raw_signal, n_components=10, sym=True)
out = dec.decompose()
```
### Plot
```python
from emg_dec_fastica.metrics import Metrics

mtr = Metrics(out['sample_rate'])
mtr.plot_muap_map(out['waveforms'].T, out['waveform_length'], 0)
mtr.plot_spike_raster(out['spike_trains'].to_dense().cpu())
mtr.plot_time_serise(ref_signal,
                     out['signal_fICA'][0].cpu(), 
                     out['spike_trains_fICA'][0].to_dense().cpu(),
                     out['signal_cICA'][0].cpu(),
                     out['spike_trains'][0].to_dense().cpu()
                     )
```

## Parameters Configuration

There are several parameters that can be configured when initializing the decomposer.
- `n_components`: Number of independent components. Defaults to 10 (int, optional).
- `fsamp`: Sampling rate in hz. Defaults to 2000 (int, optional).
- `L` : Length of waveform in ms. Defaults to 25 (int, optional).
- `K`: Delay used in ms. Defaults to 15 (int, optional).
- `thh`: Threshold used to extract spikes for the signal. Defaults to 4 (float, optional).
- `max_iter`: Maximum iteration in fast ICA. Defaults to 200 (int, optional).
- `tol`: Tolerant of convergence in fastICA. Defaults to 1e- (float, optional).
- `sym`: determine if using the symetric fast ICA (bool, optional).

## Contributors
- [Jiayu Zhong], [jiayu.zhong@tum.de](jiayu.zhong@tum.de)

## References

* [FastICA](https://europepmc.org/article/med/10946390), Aapo Hyvärinen and Erkki Oja
* [A Novel Framework Based on FastICA for High Density Surface EMG Decomposition](https://pubmed.ncbi.nlm.nih.gov/25775496/), Maoqi Chen and Ping Zhou

