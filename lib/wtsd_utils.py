import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from lib import data, tsne_utils, vid_utils

def watershed_pd(p_density):
    coords = peak_local_max(p_density, footprint=np.ones((3, 3)))
    mask = np.zeros(p_density.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-p_density, markers, mask=np.invert(p_density == 0), watershed_line=True)

    return labels

def plot_wslabels(labels, p_density, title=None, subtitle=None):
    fig, ax = plt.subplots(1,2)

    im1 = ax[0].imshow(labels, cmap='jet')
    ax[0].set_xlabel(f"Axis 0"); ax[0].set_ylabel(f"Axis 1")
    ax[0].set_title("Watershed Labels")
    fig.colorbar(im1, ax=ax[0], orientation='vertical')

    im2 = ax[1].imshow(p_density, cmap='jet')
    ax[0].set_xlabel(f"Axis 0"); ax[0].set_ylabel(f"Axis 1")
    ax[0].set_title("Probability Density")
    fig.colorbar(im2, ax=ax[1], orientation='vertical')


    if title==None: 
        title = "Watershed Labels and Probability Density"
        if not subtitle == None:
            title += f" - {subtitle}"
    fig.suptitle(title)

def get_idx_labels(embeds, edges, labels, label_id):
    label_idx = np.asarray(np.where(labels == label_id)).T

    idx = []
    for i, j in label_idx:
        idx.append(tsne_utils.get_idx_box(embeds, edges, i, j))
    return [i for idx_list in idx for i in idx_list]

def save_label_frames(embeds, edges, labels, label_id, subj_id):
    idx = get_idx_labels(embeds, edges, labels, label_id)
    vid_utils.save_vidclip_frames(data.subject_ids[subj_id], data.session_ids[subj_id], frames=idx, tag=f"{data.subject_ids[subj_id]}-region{label_id}")