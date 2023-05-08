import matplotlib.pyplot as plt

from src.config import GENERATED_PATH


def show(imgs, ncols=4, nrows=4):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(6, 6))
    for i, img in enumerate(imgs):
        axs[i // ncols, i % ncols].imshow(img)
        axs[i // ncols, i % ncols].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(GENERATED_PATH)


def visualize_collection(collection, n=16):
    imgs = [collection[i].cpu().detach().numpy().transpose((1, 2, 0)) for i in range(n)]
    show(imgs, int(n ** .5), int(n ** .5))
