import numpy as np


def color_image(image, num_classes=1):
    import matplotlib as mpl
    import matplotlib.cm
    norm = mpl.colors.Normalize(vmin=0., vmax=num_classes)
    mycm = mpl.cm.get_cmap('Greys_r')
    return mycm(norm(image))
