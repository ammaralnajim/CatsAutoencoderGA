import matplotlib.pyplot as plt

from src.config import ROOT_N_IMAGES


def show(imgs, save_path):
    """
    Output image grid to specified file

    :param imgs: collection of images
    :param save_path:
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=ROOT_N_IMAGES, nrows=ROOT_N_IMAGES, figsize=(ROOT_N_IMAGES, ROOT_N_IMAGES))
    for i, img in enumerate(imgs):
        axs[i // ROOT_N_IMAGES, i % ROOT_N_IMAGES].imshow(img)
        axs[i // ROOT_N_IMAGES, i % ROOT_N_IMAGES].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(save_path)


def visualize_collection(collection, save_path):
    """
    transforms collection of torch Tensors for future output

    :param collection:
    :param save_path:
    :return:
    """
    imgs = [collection[i].cpu().detach().numpy().transpose((1, 2, 0)) for i in range(ROOT_N_IMAGES ** 2)]
    show(imgs, save_path)
