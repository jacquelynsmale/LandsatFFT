# WARNING! This is an older (but working) version, and not all of the libraries required are available in the
# environment specified in environment.yml
import logging
import numpy as np
import numpy.fft as fft
import pandas as pd
import scipy.ndimage as nd
from osgeo import gdal
from scipy.spatial.distance import pdist
from skimage.measure import regionprops_table
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


def calculate_slope(point1, point2):
    slope = np.arctan((point1[1] - point2[1]) / (point1[0] - point2[0]))
    return slope


def locate_max_in_subset(regions_params, distance_arr):
    max_location = {}
    for name, bounds in regions_params.items():
        m_min, m_max, n_min, n_max = bounds
        subset = distance_arr[m_min:m_max, n_min:n_max].copy()
        indices = np.where(np.max(subset) == subset)
        m, n = (indices[0][0], indices[1][0])
        m += m_min
        n += n_min
        max_location[name] = (m, n)

    return max_location


def find_centroid(arr):
    labeled_img, num_labels = nd.label(arr)
    props = regionprops_table(labeled_img, properties=('centroid', 'area'))
    props = pd.DataFrame(props)
    main_centroid = props.loc[props['area'] == props['area'].max(), ['centroid-0', 'centroid-1']]
    centroid_m = int(np.floor(main_centroid['centroid-0']))
    centroid_n = int(np.floor(main_centroid['centroid-1']))
    return centroid_m, centroid_n


def find_slopes(arr):
    labeled_img, num_labels = nd.label(arr)
    props = regionprops_table(labeled_img, properties=('area', 'bbox'))

    m_l = props['bbox-0'][0]
    m_r = props['bbox-2'][0]
    n_t = props['bbox-1'][0]
    n_b = props['bbox-3'][0]

    n_l = np.argmax(labeled_img[m_l])
    n_r = np.argmax(labeled_img[m_r - 1])
    m_t = np.argmax(labeled_img[:, n_t])
    m_b = np.argmax(labeled_img[:, n_b - 1])

    slope1 = calculate_slope([m_r, n_r], [m_b, n_b])
    slope2 = calculate_slope([m_l, n_l], [m_t, n_t])
    slope3 = calculate_slope([m_t, n_t], [m_r, n_r])
    slope4 = calculate_slope([m_b, n_b], [m_l, n_l])

    return slope1, slope2, slope3, slope4


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

    centroid_m, centroid_n = find_centroid(valid_domain)
    slope1, slope2, slope3, slope4 = find_slopes(valid_domain)

    filter_base = np.full((m, n), False)
    filter_base[center_m - 70:center_m + 70, :] = 1
    filter_base[:, center_n - 100:center_n + 100] = 0

    filter_a = nd.rotate(filter_base, np.rad2deg(np.nanmax([slope1, slope2])), reshape=False)
    filter_b = nd.rotate(filter_base, np.rad2deg(np.nanmax([slope3, slope4])), reshape=False)

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

    if ((sA / sB >= 2) | (sB / sA >= 2)) & ((sA > power_threshold) | (sB > power_threshold)):
        if sA > sB:
            final_filter = filter_a.copy()
        elif sB > sA:
            final_filter = filter_b.copy()

        filtered_image = np.real(fft.ifft2(fft.ifftshift(fft_image * (1 - (final_filter)))))
        filtered_image[~valid_domain] = 0
    else:
        print('Power along flight direction does not exceed banding threshold. No banding filter applied.')

    return filtered_image


def main():
    image_dir = './scenes/'

    Ix, transform, projection = load_geotiff(image_dir + 'LT05_L2SP_018013_20090330_20200827_02_T1_SR_B2.TIF')

    valid_domain = np.array(~Ix.mask)
    Ix = np.array(Ix.filled(fill_value=0.0)).astype(float)

    wallis = wallis_filter(Ix, filter_width=21)
    wallis[~valid_domain] = 0

    ls_fft = fft_filter(wallis, valid_domain, power_threshold=500)
    write_geotiff(image_dir + 'filtered_image.tif', ls_fft, transform, projection, nodata=0.0)


if __name__ == '__main__':
    main()