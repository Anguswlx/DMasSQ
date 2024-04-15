def norm(x):
    y = ((x- x.min())/(x.max() - x.min()) - 0.5)*2
    return y, x.min(), x.max()


def renorm(y,xmin,xmax):
    x = (y/2 + 0.5) *(xmax - xmin) + xmin
    return x