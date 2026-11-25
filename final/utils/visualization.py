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
