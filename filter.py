import cv2
import numpy as np
import numpy.fft as fft
import scipy.ndimage as nd
from osgeo import gdal
from scipy.spatial import distance as dist


def load_geotiff(infile, band=1):
    ds = gdal.Open(infile, gdal.GA_ReadOnly)

    data = ds.GetRasterBand(band).ReadAsArray()
    nodata = ds.GetRasterBand(band).GetNoDataValue()
    projection = ds.GetProjection()
    transform = ds.GetGeoTransform()
    ds = None
    return data, transform, projection, nodata


def write_geotiff(outfile, data, transform, projection, nodata):
    driver = gdal.GetDriverByName("GTiff")

    if isinstance(data, np.ma.core.MaskedArray):
        nodata = data.fill_value
        data = data.filled()

    rows, cols = data.shape
    ds = driver.Create(outfile, cols, rows, 1, gdal.GDT_Float64)
    ds.SetGeoTransform(transform)
    ds.SetProjection(projection)

    ds.GetRasterBand(1).SetNoDataValue(nodata)
    ds.GetRasterBand(1).WriteArray(data)
    ds = None
    return outfile


def wallis_filter(Ix, filter_width):
    kernel = np.ones((filter_width, filter_width), dtype=np.float32)
    n = np.sum(kernel)
    m = cv2.filter2D(Ix, -1, kernel, borderType=cv2.BORDER_CONSTANT) / n

    m2 = cv2.filter2D(Ix**2, -1, kernel, borderType=cv2.BORDER_CONSTANT) / n
    std = np.sqrt(m2 - (m**2)) * np.sqrt(n / (n - 1))
    filtered = (Ix - m) / std
    return filtered


def find_largest_region(binary_arr):
    n_labels, label_arr = cv2.connectedComponents(binary_arr)
    sizes = nd.sum(binary_arr, label_arr, range(n_labels + 1))
    max_label = sizes.argmax()
    label_arr[label_arr != max_label] = 0
    return label_arr


def calculate_slope(point1, point2):
    slope = np.rad2deg(np.arctan((point1[1] - point2[1]) / (point1[0] - point2[0])))
    return slope


def order_points(pts):
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    (tl, bl) = left_most
    D = dist.cdist(tl[np.newaxis], right_most, "euclidean")[0]
    (br, tr) = right_most[np.argsort(D)[::-1], :]
    return np.array([tl, tr, br, bl], dtype="float32")


def get_slopes(corners):
    tl, tr, br, bl = order_points(corners)
    slope1 = calculate_slope(br, bl)
    slope2 = calculate_slope(tr, tl)
    slope3 = calculate_slope(tr, br)
    slope4 = calculate_slope(tl, bl)
    along_track_angle = -1 * np.nanmax([slope3, slope4])
    cross_track_angle = -1 * np.nanmax([slope1, slope2])
    return along_track_angle, cross_track_angle


def fft_filter(Ix, valid_domain, power_threshold):
    y, x = valid_domain.shape
    center_y = y / 2
    center_y_int = np.floor(y / 2).astype(int)
    center_x = x / 2
    center_x_int = np.floor(x / 2).astype(int)

    regions = (valid_domain != 0).astype("uint8") * 255
    breakpoint()
    single_region = find_largest_region(regions)
    single_region = np.uint8(single_region * 255)
    contours, hierarchy = cv2.findContours(single_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy.shape[1] > 1:
        raise ValueError(f"{hierarchy.shape[1]} external objects founds, only expecting 1.")
    contour = contours[0]

    hull = cv2.convexHull(contour, returnPoints=True)
    hull_image = np.zeros(valid_domain.shape, dtype=np.uint8)
    hull_image = cv2.drawContours(hull_image, [hull], -1, 255)
    corners = cv2.goodFeaturesToTrack(hull_image, 4, 0.33, 1000)[:, 0, :]

    along_track, cross_track = get_slopes(corners)
    print(f"Along track angle is {along_track:.2f} degrees")

    filter_base = np.zeros((y, x))
    filter_base[center_y_int - 70 : center_y_int + 70, :] = 1
    filter_base[:, center_x_int - 100 : center_x_int + 100] = 0

    rotation_a = cv2.getRotationMatrix2D(center=(center_x, center_y), angle=cross_track, scale=1)
    rotation_b = cv2.getRotationMatrix2D(center=(center_x, center_y), angle=along_track, scale=1)
    filter_a = cv2.warpAffine(src=filter_base, M=rotation_a, dsize=(x, y))
    filter_b = cv2.warpAffine(src=filter_base, M=rotation_b, dsize=(x, y))

    # Alex's code appears note to use this shift
    # moment = cv2.moments(contour)
    # centroid_y = moment["m01"] / moment["m00"]
    # centroid_x = moment["m10"] / moment["m00"]
    # y_shift = centroid_y - center_y
    # x_shift = centroid_x - center_x
    # print(f"shift = ({x_shift:.1f},{y_shift:.1f})")

    # translation = np.array([[1, 0, x_shift],
    #                         [0, 1, y_shift]],
    #                        dtype=np.float32)
    # filter_a = cv2.warpAffine(src=filter_a, M=translation, dsize=(x, y))
    # filter_b = cv2.warpAffine(src=filter_b, M=translation, dsize=(x, y))

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
        print(
            f"Power along flight direction ({max(sB, sA)}) does not exceed banding threshold ({power_threshold}). "
            f"No banding filter applied."
        )
        return image

    return filtered_image


def main():
    image_dir = "./scenes/"

    Ix, transform, projection, nodata = load_geotiff(image_dir + "LT05_L2SP_018013_20060610_20200901_02_T1_SR_B2.TIF")
    valid_domain = np.array(Ix != nodata)
    Ix[~valid_domain] = 0
    Ix = Ix.astype(float)

    wallis = wallis_filter(Ix, filter_width=5)
    wallis[~valid_domain] = 0
    write_geotiff(image_dir + "wallis_image.tif", wallis, transform, projection, nodata=0.0)
    ls_fft = fft_filter(wallis, valid_domain, power_threshold=500)
    ls_fft[~valid_domain] = 0
    write_geotiff(image_dir + "filtered_image.tif", ls_fft, transform, projection, nodata=0.0)


if __name__ == "__main__":
    main()
