import torch
import numpy as np
import matplotlib.pyplot as plt

def grab(var):
  return var.detach().cpu().numpy()

def show_images(images, title=""):
    """Shows the provided images as sub-pictures in a square"""

    # Converting images to CPU numpy arrays
    if type(images) is torch.Tensor:
        images = grab(images)

    # Defining number of rows and columns
    fig = plt.figure(figsize=(8, 8))
    if len(images)>16:
        rows = 4
        cols = 4
    else:
        rows = int(len(images) ** (1 / 2))
        cols = round(len(images) / rows)

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)
            if idx < len(images):
                im = plt.imshow(images[idx][0], cmap='viridis')#, cmap="gray")
                im.axes.xaxis.set_visible(False)
                im.axes.yaxis.set_visible(False)
                idx += 1
                
    fig.suptitle(title, fontsize=30)
    
    #fig 的位置为[0,1]，设置前面4个子图的占的位置为[0,0.8]
    fig.subplots_adjust(right=0.8)

    #在原fig上添加一个子图句柄为cbar_ax, 设置其位置为[0.85,0.15,0.05,0.7]
    #colorbar 左 下 宽 高 
    l = 0.85
    b = 0.12
    w = 0.05
    h = 1 - 2*b 
    #对应 l,b,w,h；设置colorbar位置；
    rect = [l,b,w,h] 
    cbar_ax = fig.add_axes(rect)

    cb = fig.colorbar(im, cax = cbar_ax) 
    
    # Showing the figure
    plt.show()


def show_samples_t(samples_t):

    # Select slices
    sample_t_np = np.array(samples_t)
    # Select first, two evenly spaced in the middle, and last
    indices = [0, sample_t_np.shape[0] // 4, 2 * sample_t_np.shape[0] // 4, 3 * sample_t_np.shape[0] // 4, sample_t_np.shape[0] - 1]
    slices = sample_t_np[indices, 1:5, 0, :, :]

    fig, axs = plt.subplots(4, 5, figsize=(12, 9), gridspec_kw={'wspace': 0.05, 'hspace': 0.1})

    for i in range(4):  # Loop over the second dimension
        for j in range(5):  # Loop over the first dimension
            # Display image
            im = axs[i, j].imshow(slices[ j, i, :, :], cmap='viridis')
            im.axes.xaxis.set_visible(False)
            im.axes.yaxis.set_visible(False)
            axs[i, j].axis('on')  # Hide axes
            

    # Create an axes for colorbar
    cbar_ax = fig.add_axes([0.92, 0.125, 0.025, 0.75])
    plt.colorbar(im, cax=cbar_ax)

    # plt.savefig('figures/sample_t_k{}.pdf'.format(k), dpi=300, bbox_inches='tight')
    plt.show()