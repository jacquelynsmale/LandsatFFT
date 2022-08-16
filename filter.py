import numpy as np
import numpy.fft as fft
import pandas as pd
import scipy.ndimage as nd
from skimage.measure import regionprops_table

valid_domain = np.zeros((10, 10))
valid_domain[3:7, 3:7] = np.ones((4, 4))
valid_domain[9, 9] = 1

m, n = valid_domain.shape

center_m = int(round(m / 2, 0))
center_n = int(round(n / 2, 0))

labeled_img, num_labels = nd.label(valid_domain)
props = regionprops_table(labeled_img, properties=('centroid', 'area', 'bbox'))
props = pd.DataFrame(props)
main_centroid = props.loc[props['area'] == props['area'].max(), ['centroid-0', 'centroid-1']]

true_array = np.full((m, n), True)
true_array[int(round(main_centroid['centroid-0'], 0)), int(round(main_centroid['centroid-1'], 0))] = False
dist_from_centroid = nd.distance_transform_edt(true_array)
dist_from_centroid[~(valid_domain > 0)] = 0

main_bbox = props.loc[props['area'] == props['area'].max(), ['bbox-0', 'bbox-1', 'bbox-2', 'bbox-3']]
min_m = main_bbox['bbox-0'][0]
min_n = main_bbox['bbox-1'][0]
max_m = main_bbox['bbox-2'][0]
max_n = main_bbox['bbox-3'][0]
bbox = [[min_m, min_n], [max_m, min_n], [min_m, max_n], [max_m, max_n]]

'''
Ok, so we are going dimension by dimension to fft the array IX
Can we use the fftn function in python (fft for n number of arrays?) 

fft_ix = fft.fftn(Ix)
'''

fft_ix = fft.fftn(Ix)

for k in range(1,len(Ix[:,:,3])):

    fftIm = Ix[:,:, k]
    fftIm[fftIm > 3] = 3
    fftIm[fftIm < -3] = -3
    fftIm[np.isnan(fftIm)] = 0
    fftIm = fft.fftshift(fft.fft2(fftIm))
    P = abs(fftIm)
    mP = np.mean(P)
    stdP = np.std(P)
    P = (P - mP) > 10 * stdP