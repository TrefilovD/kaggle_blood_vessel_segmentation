import matplotlib.pyplot as plt
import numpy as np

def show_images(images, masks = None, titles = None, cmap='gray'):
    n = len(images.shape[0])
    if masks:
        cols = 2
        assert images.shape[0] == masks.shape[0]
    else:
        cols = 1
    assert cols == len(titles)
    fig, axes = plt.subplots(n, cols, figsize=(20, 10))
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    for i in range(n):
        axes[i][0].imshow(images[i], cmap=cmap)
        if titles:
            axes[i][0].set_title(titles[0])
        if masks:
            axes[i][1].imshow(masks[i], cmap=cmap)
            if titles:
                axes[i][1].set_title(titles[1])
        # axes[i].axis('off')
    plt.tight_layout()
    plt.show()