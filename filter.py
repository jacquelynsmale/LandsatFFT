import numpy as np
import numpy.fft as fft
import pandas as pd
import scipy.ndimage as nd
from osgeo import gdal
from scipy.spatial.distance import pdist
from skimage.measure import regionprops_table


def calculate_slope(point1, point2):
    slope = np.arctan((point1[0] - point2[0]) / (point1[1] - point2[1]))
    return slope

def find_max(corner):
    indices = np.where(np.max(corner) == corner)
    return indices[0][0], indices[1][0]

def locate_max_in_subset(arr, bounds):
    m_min, m_max, n_min, n_max = bounds
    subset = arr[m_min:m_max, n_min:n_max].copy()
    indices = np.where(np.max(subset) == subset)
    m, n = (indices[0][0], indices[1][0])
    m += m_min
    n += n_min
    return m, n

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

def write_geotiff(outfile, data, transform, projection):
    driver = gdal.GetDriverByName('GTiff')

    nodata = data.fill_value
    unmasked_data = data.filled()

    rows, cols = data.shape
    ds = driver.Create(outfile, cols, rows, 1, gdal.GDT_Float64)
    ds.SetGeoTransform(transform)
    ds.SetProjection(projection)

    ds.GetRasterBand(1).SetNoDataValue(nodata)
    ds.GetRasterBand(1).WriteArray(unmasked_data)
    data_set = None
    return outfile


def fft_filter(Ix, valid_domain):
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

    regions = {'top_left': (0, center_m, 0, center_n),
               'top_right': (0, center_m, center_n + 1, n),
               'bottom_left': (center_m + 1, m, 0, center_n),
               'bottom_right': (center_m + 1, m, center_n + 1, n)}

    max_location = {}
    for name, bounds in regions.items():
        max_location[name] = locate_max_in_subset(dist_from_centroid, bounds)

    corners = [max_location['bottom_left'], max_location['bottom_right'], max_location['top_left'],
               max_location['top_right']]
    sep = pdist(corners, 'euclidean')

    if any(sep < center_m):
        regions = {'top_left': (0, round(center_m / 2), 0, center_n),
                   'top_right': (0, m, 0, round(center_m / 2)),
                   'bottom_left': (round(center_m / 2 * 3), m, 0, n),
                   'bottom_right': (0, m, round(center_n / 2 * 3), n)}

        for name, bounds in regions.items():
            max_location[name] = locate_max_in_subset(dist_from_centroid, bounds)

        corners = [max_location['bottom_left'], max_location['bottom_right'], max_location['top_left'],
                   max_location['top_right']]

        sep = pdist(corners, 'euclidean')
        if any(sep < center_m):
            # print('error encountered but disabled')
            raise ValueError('two or more recovered image corner locations are too close to eachother'
                             '\n add to cleanLandsatDataDir corruptImages list then run cleanLandsatDataDir')

    slope1 = calculate_slope(max_location['bottom_right'], max_location['bottom_left'])
    slope2 = calculate_slope(max_location['top_right'], max_location['top_left'])
    slope3 = calculate_slope(max_location['top_left'], max_location['bottom_left'])
    slope4 = calculate_slope(max_location['bottom_right'], max_location['top_right'])

    filter_base = np.full((m, n), False)
    filter_base[center_m - 70:center_m + 70, :] = 1
    filter_base[:, center_n - 100:center_n + 100] = 0

    filter_a = nd.rotate(filter_base, np.rad2deg(np.nanmax([slope1, slope2])), reshape=False)
    filter_b = nd.rotate(filter_base, np.rad2deg(np.nanmax([slope3, slope4])), reshape=False)

    ctr_shift = (round(main_centroid['centroid-0'] - center_m), round(main_centroid['centroid-1'] - center_n))

    translation_matrix = np.array([(1, 0, ctr_shift[0][0]), (0, 1, ctr_shift[1][0]), (0, 0, 1)])
    filter_a = nd.affine_transform(filter_a, matrix=translation_matrix)
    filter_b = nd.affine_transform(filter_b, matrix=translation_matrix)

    image = Ix.copy()
    image[image > 3] = 3
    image[image < -3] = -3
    image[np.isnan(image)] = 0

    fft_image = fft.fftshift(fft.fft2(image))
    P = abs(fft_image)
    mP = np.mean(P)
    stdP = np.std(P)
    P = (P - mP) > 10 * stdP

    sA = np.nansum(P[filter_a])
    sB = np.nansum(P[filter_b])

    #if (sA / sB >= 2 | sB / sA >= 2) & (sA > 500 | sB > 500):
    if sA > sB:
        final_filter = filter_a.copy()
    elif sB > sA:
        final_filter = filter_b.copy()

    filtered_image = np.real(fft.ifft2(fft.ifftshift(fft_image * (1 - (final_filter)))))
    filtered_image[np.isnan(valid_domain)] = np.nan

    return filtered_image

def main():
    # valid_domain = np.zeros((1000, 1000))
    # valid_domain[300:700, 300:700] = 1
    # valid_domain[900:950, 900:950] = 1

    image_dir = '/Users/jrsmale/projects/LandsatFFT/scenes/'

    Ix, transform, projection = load_geotiff(image_dir + 'LT05_L2SP_062018_20091012_20200825_02_T1_SR_B1.TIF')
    valid_domain = ~Ix.mask
    Ix = Ix.filled(fill_value=0)

    filtered = fft_filter(Ix, valid_domain)

    write_geotiff(image_dir + 'filtered.tif', filtered, transform, projection)


if __name__ == '__main__':
    main()
