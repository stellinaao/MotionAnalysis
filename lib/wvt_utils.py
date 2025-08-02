import numpy as np
import pywt
import matplotlib.pyplot as plt
from matplotlib import colors
from lib import data, constants


def wavelet_decomp_pca(dlc_redux, freqs_in=None):
    print("running wavelet_decom_pca")
    dlc_redux = dlc_redux.T

    time = get_time(len(dlc_redux[0]), data.fs)
    n_freqs = len(get_freqs()) if any(freqs_in == None) else len(freqs_in)

    cwtmatrs = np.zeros((len(dlc_redux), n_freqs-1, len(time)-1))
    freqs = np.zeros((len(dlc_redux), n_freqs))

    for i, pc in enumerate(dlc_redux):
        cwtmatrs[i], freqs[i] = wavelet_decomp(pc, freqs_in)
    
    return cwtmatrs, freqs

def wavelet_decomp(time_series, freqs=None, fs=30, wavelet="cmor2-1.0", w0=2*np.pi,f_min=1, f_max=50, f_N=25, doPlot=False):
    widths = get_corr_scales(f_min, f_max, f_N, w0) if any(freqs==None) else pywt.frequency2scale(wavelet, freqs)

    cwtmatr, freqs = pywt.cwt(time_series, widths*fs, wavelet, sampling_period=1/fs) # (signal, 1/freq, wavelet, 1/fs)
    cwtmatr = np.abs(cwtmatr[:-1, :-1])

    if doPlot:
        time = get_time(len(time_series), fs)
        plot_wavelet_decomp(cwtmatr, freqs, time)
        print(f"{cwtmatr.shape}, {freqs.shape}, {time.shape}")
    
    return cwtmatr, freqs

def get_scales(f_min=1, f_max=50, f_N=25, w0=2*np.pi):
    freqs = get_freqs(f_min, f_max, f_N)
    scales = [(w0+(2+w0**2))/(4*np.pi*f) for f in freqs]
    return scales

def get_corr_scales(f_min=1, f_max=50, f_N=25, w0=2*np.pi):
    scales = get_scales(f_min, f_max, f_N, w0)
    corr_scales = [np.pi**(-1/4)/((2*s)**0.5)*np.e**((1/4)*((w0-(w0**2+2)**0.5)**2)) for s in scales]
    return corr_scales

    
def get_freqs(f_min=1, f_max=50, f_N=25):
    return [(f_max*2**((i-1)/(f_N-1)*np.log2(f_max/f_min))) for i in range(f_N)]

def get_time(n_points, fs, doTest=False):
    return np.linspace(start=0, stop=n_points/fs, num=n_points)

def test_fn():
    time = get_time(30, 30)
    assert min(time) == 0 and max(time) == 1 and len(time) == 30

def plot_wavelet_decomp(cwtmatr, freqs, time, ax=None, vmin=None, vmax=None, norm=None, title=None, subtitle=None):
    if ax==None:
        fig, ax = plt.subplots()
    if not(vmin == None and vmax == None): 
        pcm = ax.pcolormesh(time, freqs, cwtmatr, vmin=vmin, vmax=vmax)
    elif ~(norm == None):
        pcm = ax.pcolormesh(time, freqs, cwtmatr, norm=norm)
    else: 
        pcm = ax.pcolormesh(time, freqs, cwtmatr)
    ax.set_yscale("log")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    
    if title==None: 
        title = "Continuous Wavelet Transform (Scaleogram)"
        if not subtitle == None:
            title += f" - {subtitle}"
    ax.set_title(title)
    
    if ax==None:
        fig.colorbar(pcm, ax=ax)

    return pcm

def plot_wavelet_decomp_pca(cwtmatrs, freqs, n_rows=None, n_cols=None, title=None, subtitle=None):
    if n_rows == None or n_cols == None:
        raise ValueError("ERROR: no value for n_rows and n_cols specified!")
    
    time = get_time(np.shape(cwtmatrs)[2]+1, data.fs)
    all_vals = np.ravel(cwtmatrs); min_val = min(all_vals); max_val = max(all_vals)
    shared_norm = colors.Normalize(vmin=min_val, vmax=8)
    pcms = []


    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*constants.SUBPLOT_SQUARE_SIDELEN, n_rows*constants.SUBPLOT_SQUARE_SIDELEN))
    for i, ax in enumerate(axes.flat):
        if i < len(cwtmatrs):
            pcm = plot_wavelet_decomp(cwtmatrs[i], freqs[i], time, ax, norm=shared_norm)
            pcms.append(pcm)

    assert all([pcm.norm == pcms[0].norm for pcm in pcms])

    if title==None: 
        title = "Wavelet Decomposition on Each PC"
        if not subtitle == None:
            title += f" - {subtitle}"
    fig.suptitle(title)
    fig.tight_layout()
    fig.colorbar(pcms[0], ax=axes)

    fig.show()
    return pcms, shared_norm