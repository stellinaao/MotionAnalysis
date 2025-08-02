import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from lib import constants, data

def get_pca_evr(dlc_coords, n_components, doPlot=False, n_rows=None, n_cols=None, title=None, subtitle=None):
    evrs = [PCA(n_components).fit(dlc_coord).explained_variance_ratio_ for dlc_coord in dlc_coords]
    
    if doPlot:
        if n_rows == None or n_cols == None:
            raise Exception("ERROR: no value for n_rows and n_cols specified!")
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*constants.SUBPLOT_SQUARE_SIDELEN, n_rows*constants.SUBPLOT_SQUARE_SIDELEN))
        for i, ax in enumerate(axes.flat):
            ax.bar(np.arange(n_components), 100*evrs[i])
            ax.set_xlabel("PC Index"); ax.set_ylabel("\% of Variance Explained")
            ax.set_title(f"{data.subject_ids[i]}")

        if title==None: 
            title = "Variance Captured by Each PC"
            if not subtitle == None:
                title += f" - {subtitle}"
        fig.suptitle(title)

        fig.tight_layout()
        plt.show()

    return evrs

def get_num_pcs(evrs, p_ev):
    num_pcs = np.zeros(len(evrs), dtype=int)
    
    for i, evr in enumerate(evrs):
        pc_idx = 0
        while np.sum(evr[:pc_idx]) < p_ev:
            pc_idx+=1
        num_pcs[i] = pc_idx
    
    return num_pcs, max(num_pcs)

def plot_cum_var(evrs, thresh, n_pcs, n_rows=None, n_cols=None, title=None, subtitle=None):
    if n_rows == None or n_cols == None:
        raise Exception("ERROR: no value for n_rows and n_cols specified!")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*constants.SUBPLOT_SQUARE_SIDELEN, n_rows*constants.SUBPLOT_SQUARE_SIDELEN))
    for i, ax in enumerate(axes.flat):
        ax.plot(np.arange(len(evrs[i])), [100*np.sum(evrs[i][:j]) for j in range(len(evrs[i]))])
        ax.axhline(thresh, linestyle='--')
        ax.set_xticks(np.arange(len(evrs[i])), np.arange(len(evrs[i])))
        ax.set_xlabel("Components"); ax.set_ylabel("\% Explained Variance")
        ax.set_title(f"{data.subject_ids[i]} ({n_pcs[i]} PCs to Explain {thresh}\% Variance)")
    
    if title==None: 
        title = "Cumulative Variance Captured by PCs"
        if not subtitle == None:
            title += f" - {subtitle}"
    fig.suptitle(title)
    
    fig.tight_layout()
    plt.show()

def plot_pca(dlc_reduxs, pc_a=0, pc_b=1,  n_rows=None, n_cols=None, title=None, subtitle=None):
    if n_rows == None or n_cols == None:
        raise Exception("ERROR: no value for n_rows and n_cols specified!")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*constants.SUBPLOT_SQUARE_SIDELEN, n_rows*constants.SUBPLOT_SQUARE_SIDELEN))
    for i, ax in enumerate(axes.flat):
        ax.scatter(dlc_reduxs[i][:, pc_a], dlc_reduxs[i][:, pc_b], s=0.5, alpha=0.5)
        ax.set_xlabel(f"PC {pc_a+1}"); ax.set_ylabel(f"PC {pc_b+1}")
        ax.set_title(f"{data.subject_ids[i]}")

    if title==None: 
        title = "Scores"
        if not subtitle == None:
            title += f" - {subtitle}"
    fig.suptitle(title)

    fig.tight_layout()
    plt.show()