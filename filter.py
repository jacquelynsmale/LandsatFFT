import cv2
import sys
import numpy as np
from scipy.ndimage import distance_transform_edt

import numpy as np
import numpy.fft as fft
# import pandas as pd
import scipy.ndimage as nd
from osgeo import gdal
# from skimage.measure import regionprops_table
import cv2


def load_geotiff(infile, band=1):
    ds = gdal.Open(infile, gdal.GA_ReadOnly)

    data = ds.GetRasterBand(band).ReadAsArray()
    nodata = ds.GetRasterBand(band).GetNoDataValue()
    mask = data == nodata
    data = np.ma.array(data, mask=mask, fill_value=-9999)
    projection = ds.GetProjection()
    transform = ds.GetGeoTransform()
    ds = None
    return data, transform, projection


def write_geotiff(outfile, data, transform, projection, nodata):
    driver = gdal.GetDriverByName('GTiff')

    if isinstance(data, np.ma.core.MaskedArray):
        nodata = data.fill_value
        data = data.filled()

    rows, cols = data.shape
    ds = driver.Create(outfile, cols, rows, 1, gdal.GDT_Float64)
    ds.SetGeoTransform(transform)
    ds.SetProjection(projection)

    ds.GetRasterBand(1).SetNoDataValue(nodata)
    ds.GetRasterBand(1).WriteArray(data)
    data_set = None
    return outfile


def find_largest_region(arr):
    binary_arr = np.zeros(arr.shape)
    binary_arr[arr != 0] = 1
    label_arr, nb_labels = nd.label(binary_arr)
    sizes = nd.sum(binary_arr, label_arr, range(nb_labels + 1))
    max_label = sizes.argmax()
    label_arr[label_arr != max_label] = 0
    return label_arr


def wallis_filter(Ix, filter_width):
    kernel = np.ones((filter_width, filter_width), dtype=np.float32)
    n = np.sum(kernel)
    m = cv2.filter2D(Ix, -1, kernel, borderType=cv2.BORDER_CONSTANT) / n

    m2 = cv2.filter2D(Ix ** 2, -1, kernel, borderType=cv2.BORDER_CONSTANT) / n
    std = np.sqrt(m2 - (m ** 2)) * np.sqrt(n / (n - 1))
    filtered = (Ix - m) / std
    return filtered


def fft_filter(Ix, valid_domain, power_threshold):
    m, n = valid_domain.shape
    center_m = int(np.floor(m / 2))
    center_n = int(np.floor(n / 2))

    single_region = find_largest_region(Ix)
    single_region = np.uint8(single_region * 255)
    contours, hierarchy = cv2.findContours(single_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy.shape[1] > 1:
        raise ValueError(f'{hierarchy.shape[1]} external objects founds, only expecting 1.')
    contour = contours[0]
    moment = cv2.moments(contour)

    centroid_m = int(np.floor(moment['m01'] / moment['m00']))
    centroid_n = int(np.floor(moment['m10'] / moment['m00']))
    rectangle = cv2.minAreaRect(contour)
    angle = rectangle[2]

    filter_base = np.full((m, n), False)
    filter_base[center_m - 70:center_m + 70, :] = 1
    filter_base[:, center_n - 100:center_n + 100] = 0

    filter_a = nd.rotate(filter_base, -angle, reshape=False)
    filter_b = nd.rotate(filter_base, 90 - angle, reshape=False)

    ctr_shift = [centroid_m - center_m, centroid_n - center_n]

    translate_matrix = np.array([(1, 0, ctr_shift[0]), (0, 1, ctr_shift[1]), (0, 0, 1)])
    filter_a = nd.affine_transform(filter_a, matrix=translate_matrix)
    filter_b = nd.affine_transform(filter_b, matrix=translate_matrix)

    image = Ix.copy()
    image[image > 3] = 3
    image[image < -3] = -3
    image[np.isnan(image)] = 0

    fft_image = fft.fftshift(fft.fft2(image))
    P = abs(fft_image)
    mP = np.mean(P)
    stdP = np.std(P)
    P = (P - mP) > (10 * stdP)

    sA = np.nansum(P[filter_a == 1])
    sB = np.nansum(P[filter_b == 1])
    print(sA, sB)
    if ((sA / sB >= 2) | (sB / sA >= 2)) & ((sA > power_threshold) | (sB > power_threshold)):
        if sA > sB:
            final_filter = filter_a.copy()
        elif sB > sA:
            final_filter = filter_b.copy()

        filtered_image = np.real(fft.ifft2(fft.ifftshift(fft_image * (1 - (final_filter)))))
        filtered_image[~valid_domain] = 0
    else:
        print(f'Power along flight direction ({max(sB, sA)}) does not exceed banding threshold ({power_threshold}). '
              f'No banding filter applied.')
        return image

    return filtered_image


def main():
    image_dir = './scenes/'

    Ix, transform, projection = load_geotiff(image_dir + 'LT05_L2SP_018013_20060610_20200901_02_T1_SR_B2.TIF')

    valid_domain = np.array(~Ix.mask)
    Ix = np.array(Ix.filled(fill_value=0.0)).astype(float)

    wallis = wallis_filter(Ix, filter_width=5)
    wallis[~valid_domain] = 0
    write_geotiff(image_dir + 'wallis_image.tif', wallis, transform, projection, nodata=0.0)

    ls_fft = fft_filter(wallis, valid_domain, power_threshold=500)
    write_geotiff(image_dir + 'filtered_image.tif', ls_fft, transform, projection, nodata=0.0)


if __name__ == '__main__':
    main()
