import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.ndimage import gaussian_filter

from lib import data, constants

def plot_density_all(tsne_embeds, n_box=100):
    fig, axes = plt.subplots(nrows=data.n_subj, ncols=3, figsize=(3*constants.SUBPLOT_SQUARE_SIDELEN, data.n_subj*constants.SUBPLOT_SQUARE_SIDELEN))

    for i in range(data.n_subj):
        axes_ = axes[i]
        fig, _ = plot_density_2d(tsne_embeds[i], n_box=n_box, axes=axes_)
        axes_[1].set_title(f"{data.subject_ids[i]}")

    fig.show()

def plot_density_2d(tsne_embed, n_box=100, axes=None, doShow=False, title=None, subtitle=None):
    if axes==None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    edges = np.linspace(start=np.min(tsne_embed), stop=np.max(tsne_embed), num=n_box+1)

    plane_i = tsne_embed[:,0]
    plane_j = tsne_embed[:,1]

    H, _, _ = np.histogram2d(plane_i, plane_j, bins=(edges, edges))
    H = gaussian_filter(H.T, sigma=1.5)
    
    im = ax.imshow(H, cmap='jet') # because the x-values are by default plotted along the ordinate
    ax.set_xlabel(f"Axis 0"); ax.set_ylabel(f"Axis 1")

    if title==None: 
        title = "Probability Density Plot"
        if not subtitle == None:
            title += f" - {subtitle}"
    fig.suptitle(title)

    fig.colorbar(im, ax=axes, orientation='vertical')

    if doShow: fig.show()

    return fig, axes, H, edges

def plot_density_3d(tsne_embed, n_box=100, axes=None, doShow=False):
    if axes==None:
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

def get_idx_box(embeds, edges, i, j):
    x_range = (edges[i], edges[i+1])
    y_range = (edges[j], edges[j+1])

    return [i for i in range(len(embeds)) if embeds[i][0] >= x_range[0] and embeds[i][0] < x_range[1] and embeds[i][1] >= y_range[0] and embeds[i][1] < y_range[1]]

def plot_embed_temporal(embed, title=None, subtitle=None):
    if title==None: 
        title = "TSNE Embedding Space"
        if not subtitle == None:
            title += f" - {subtitle}" 

    fig = px.scatter(x=embed[:,0], y=embed[:,1], color=np.arange(len(embed)), title=title)
    fig.update_traces(marker=dict(size=1))
    fig.update_layout(coloraxis_colorbar=dict(title="index"), width=constants.SUBPLOT_SQUARE_SIDELEN*200, height=constants.SUBPLOT_SQUARE_SIDELEN*200)
    fig.show()
    
def plot_embed_power(embed, cwtmatr_fvec, title=None, subtitle=None):
    if title==None: 
        title = "TSNE Embedding Space"
        if not subtitle == None:
            title += f" - {subtitle}"
            
    fig = px.scatter(x=embed[:,0], y=embed[:,1], color=np.log([np.sum(vec) for vec in cwtmatr_fvec]), title=title)
    fig.update_traces(marker=dict(size=1))
    fig.update_layout(coloraxis_colorbar=dict(title="power"), width=constants.SUBPLOT_SQUARE_SIDELEN*200, height=constants.SUBPLOT_SQUARE_SIDELEN*200)
    fig.show()