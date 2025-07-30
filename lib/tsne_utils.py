import numpy as np
import matplotlib.pyplot as plt

from lib import data, constants

def plot_density_all(tsne_embeds, n_box=100):
    fig, axes = plt.subplots(nrows=data.n_subj, ncols=3, figsize=(3*constants.SUBPLOT_SQUARE_SIDELEN, data.n_subj*constants.SUBPLOT_SQUARE_SIDELEN))

    for i in range(data.n_subj):
        axes_ = axes[i]
        fig, _ = plot_density(tsne_embeds[i], n_box=n_box, axes=axes_)
        axes_[1].set_title(f"{data.subject_ids[i]}")

    fig.show()

def plot_density(tsne_embed, n_box=100, axes=None, doShow=False):
    if any(axes==None):
        fig, axes = plt.subplots(nrows=1, ncols=3)
    else:
        fig = plt.gcf()

    idx = [(i,j) for i in range(3) for j in range(3) if i < j]

    edges = np.linspace(start=np.min(tsne_embed), stop=np.max(tsne_embed), num=n_box)

    for i, ax in enumerate(axes):
        plane_i = tsne_embed[:,idx[i][0]]
        plane_j = tsne_embed[:,idx[i][1]]

        H, _, _ = np.histogram2d(plane_i, plane_j, bins=(edges, edges))
        
        im = ax.imshow(H.T) # because the x-values are by default plotted along the ordinate
        ax.set_xlabel(f"Axis {idx[i][0]}"); ax.set_ylabel(f"Axis {idx[i][1]}")
    
    fig.colorbar(im, ax=axes, orientation='vertical')

    if doShow: fig.show()

    return fig, axes