import numpy as np

from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
from sklearn.cluster import KMeans
    
def get_n_subdivsions_by_area(area, l_high, l_low, n_max):
    if area >= l_high:
        return n_max
    elif area <= l_low:
        return 1
    else:
        t = (area - l_low) / (l_high - l_low)
        return int(round(1 + t * (n_max - 1)))

def split_by_watershed(mask, k):
    coords = np.column_stack(np.nonzero(mask))
    km = KMeans(k, n_init='auto').fit(coords)
    markers = np.zeros_like(mask, int)
    for (y, x), lbl in zip(coords, km.labels_):
        markers[y, x] = lbl + 1
    dist = distance_transform_edt(mask)
    ws   = watershed(-dist, markers, mask=mask)
    return [(ws == i) for i in range(1, k+1)]

