import numpy as np
import numpy.fft as fft
import pandas as pd
import scipy.ndimage as nd
from skimage.measure import regionprops_table
from osgeo import gdal


def calculate_slope(point1, point2):
    slope = np.arctan((point1[0] - point2[0]) / (point1[1] - point2[1]))
    return slope


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

    main_bbox = props.loc[props['area'] == props['area'].max(), ['bbox-0', 'bbox-1', 'bbox-2', 'bbox-3']]
    min_m = main_bbox['bbox-0'][0]
    min_n = main_bbox['bbox-1'][0]
    max_m = main_bbox['bbox-2'][0]
    max_n = main_bbox['bbox-3'][0]
    bbox = [[min_m, min_n], [max_m, min_n], [min_m, max_n], [max_m, max_n]]

    slope1 = calculate_slope([max_m, max_n], [min_m, max_m])
    slope2 = calculate_slope([max_m, min_n], [min_m, min_n])
    slope3 = calculate_slope([min_m, min_n], [min_m, max_n])
    slope4 = calculate_slope([max_m, max_n], [max_m, min_n])

    fooA = np.full((m, n), False)
    fooA[center_m - 70:center_m + 70, :] = 1
    fooA[:, center_n - 100:center_n + 100] = 0

    A = nd.rotate(fooA, np.rad2deg(np.nanmax([slope1, slope2])), reshape=False)
    B = nd.rotate(fooA, np.rad2deg(np.nanmax([slope3, slope4])), reshape=False)

    shiftCtr = (round(main_centroid['centroid-0'] - center_m), round(main_centroid['centroid-1'] - center_n))

    matrix = np.array([(1, 0, 0), (0, 1, 0), (shiftCtr[0], shiftCtr[1], 1)])
    A = nd.affine_transform(A, matrix=matrix) # Failing here
    B = nd.affine_transform(B, matrix=matrix)

    fftIm = Ix.copy()
    fftIm[fftIm > 3] = 3
    fftIm[fftIm < -3] = -3
    fftIm[np.isnan(fftIm)] = 0
    fftIm = fft.fftshift(fft.fft2(fftIm))
    P = abs(fftIm)
    mP = np.mean(P)
    stdP = np.std(P)
    P = (P - mP) > 10 * stdP
    sA = np.nansum(P[A])
    sB = np.nansum(P[B])
    if (sA / sB >= 2 | sB / sA >= 2) & (sA > 500 | sB > 500):
        if sA > sB:
            mask = A.copy()
        elif sB > sA:
            mask = B.copy()

        foo1 = np.isnan(valid_domain)
        foo2 = np.real(fft.ifft2(fft.ifftshift(fftIm * (1 - (mask)))))
        foo2[foo1] = np.nan

    return foo2


def main():
    # valid_domain = np.zeros((1000, 1000))
    # valid_domain[300:700, 300:700] = 1
    # valid_domain[900:950, 900:950] = 1

    Ix, transform, projection = load_geotiff('data/LT05_L2SP_062018_20091012_20200825_02_T1_SR_B1.TIF')
    valid_domain = ~Ix.mask
    Ix = Ix.filled(fill_value=0)

    filtered = fft_filter(Ix, valid_domain)

    write_geotiff('data/filtered.tif', filtered, transform, projection)


if __name__ == '__main__':
    main()
