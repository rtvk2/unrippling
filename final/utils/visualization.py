import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def montage(images, path):
    cols = min(4, len(images))
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (name, image) in zip(axes.flat, images.items()):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def motion_figure(template, field, validity, path, tracks=None):
    """First-frame pull vectors in pixels and temporal valid-sampling fraction."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(template, cmap="gray", vmin=0, vmax=1)
    h, w = field.shape[:2]
    step = max(1, min(h, w) // 14)
    import numpy as np

    y, x = np.mgrid[:h:step, :w:step]
    dx, dy = field[::step, ::step].transpose(2, 0, 1)
    axes[0].quiver(
        x, y, dx, dy, angles="xy", scale_units="xy", scale=1, color="tab:orange"
    )
    if tracks is not None:
        centroids = np.nanmean(tracks, axis=0)
        axes[0].scatter(*centroids.T, s=3, color="tab:cyan")
    axes[0].set_title(
        "Frame 0 pull field" + (" and KLT centroids" if tracks is not None else "")
    )
    axes[1].imshow(validity.mean(axis=0), cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title("Fraction of frames sampled in bounds")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=125)
    plt.close(fig)
