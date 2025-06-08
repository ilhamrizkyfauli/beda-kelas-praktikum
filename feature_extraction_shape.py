import os
import cv2
import numpy as np
from skimage.measure import regionprops, label
import mahotas
from skimage.morphology import skeletonize

def extract_shape_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Gagal membaca gambar: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    labeled = label(thresh)
    props = regionprops(labeled)
    if not props:
        return np.zeros(52)

    region = max(props, key=lambda r: r.area)
    area = region.area
    perimeter = region.perimeter
    eccentricity = region.eccentricity
    extent = region.extent
    solidity = region.solidity
    minr, minc, maxr, maxc = region.bbox
    aspect_ratio = (maxc - minc) / (maxr - minr) if (maxr - minr) != 0 else 0

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(52)

    cnt = max(contours, key=cv2.contourArea)

    # Hu Moments
    moments = cv2.moments(cnt)
    hu_moments = cv2.HuMoments(moments).flatten()
    hu_moments = [-np.sign(h)*np.log10(abs(h)) if h != 0 else 0 for h in hu_moments]

    # Zernike Moments dengan padding
    cropped = thresh[minr:maxr, minc:maxc]
    radius = max(cropped.shape) // 2
    try:
        zm = mahotas.features.zernike_moments(cropped, radius, degree=8)
        zernike_moments = zm.tolist()
        if len(zernike_moments) < 25:
            zernike_moments += [0] * (25 - len(zernike_moments))
    except:
        zernike_moments = [0] * 25

    # Fourier Descriptors dengan padding
    contour_array = cnt[:, 0, :]
    complex_contour = contour_array[:, 0] + 1j * contour_array[:, 1]
    descriptors = np.fft.fft(complex_contour)
    fd_len = 10
    fd_features = np.abs(descriptors[:fd_len])
    if len(fd_features) < fd_len:
        fd_features = np.pad(fd_features, (0, fd_len - len(fd_features)), 'constant')

    # Convex Hull Features
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    contour_area = cv2.contourArea(cnt)
    ch_features = [hull_area, float(contour_area) / hull_area if hull_area > 0 else 0]

    # Skeleton Features
    binary_bool = (thresh == 255)
    skeleton = skeletonize(~binary_bool)
    skeleton_length = np.sum(skeleton)

    # Fractal Dimension
    Z = (thresh < 128)
    def boxcount(Z, k):
        S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                                           np.arange(0, Z.shape[1], k), axis=1)
        return len(np.where((S > 0) & (S < k*k))[0])
    p = min(Z.shape)
    n = 2**np.floor(np.log2(p)).astype(int)
    sizes = 2**np.arange(int(np.log2(n)), 1, -1)
    counts = [boxcount(Z, size) for size in sizes]
    if np.all(np.array(counts) == 0):
        fractal_feat = [0]
    else:
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        fractal_feat = [coeffs[0]]

    feature_vector = [
        area, perimeter, eccentricity, extent, aspect_ratio, solidity
    ] + hu_moments + zernike_moments + list(fd_features) + ch_features + [skeleton_length] + fractal_feat

    return np.array(feature_vector)


if __name__ == "__main__":
    image_dir = 'images/organik'
    expected_length = 52
    all_color_features = []
    gagal_files = []

    all_files_in_dir = os.listdir(image_dir)
    print(f"Jumlah total file di direktori '{image_dir}': {len(all_files_in_dir)}")

    image_files = [f for f in all_files_in_dir if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Jumlah file citra yang terdeteksi setelah filter: {len(image_files)}")

    for i, img_file in enumerate(image_files):
        img_path = os.path.join(image_dir, img_file)
        try:
            features = extract_shape_features(img_path)
            if features is not None and len(features) == expected_length:
                all_color_features.append(features)
            else:
                print(f"[!] Gambar dilewati: {img_file} (panjang fitur = {len(features)})")
                gagal_files.append(img_file)
        except Exception as e:
            print(f"[!] Error saat ekstraksi {img_file}: {e}")
            gagal_files.append(img_file)

    if len(all_color_features) > 0:
        all_color_features = np.array(all_color_features)
        print("--------------------------------------------")
        print(f"Ekstraksi fitur bentuk selesai. Total fitur diekstrak: {all_color_features.shape}")
        os.makedirs("fitur", exist_ok=True)
        np.savetxt("fitur/extracted_shape_features_organik.csv", all_color_features, delimiter=",")
        print("Fitur bentuk untuk organik berhasil disimpan ke 'fitur/extracted_shape_features_organik.csv'")
    else:
        print("Tidak ada fitur bentuk yang berhasil diekstrak.")

    if gagal_files:
        with open("fitur/gagal_diekstrak.txt", "w") as f:
            for img_file in gagal_files:
                f.write(f"{img_file}\n")
        print(f"[!] {len(gagal_files)} gambar gagal diekstrak. Disimpan di 'fitur/gagal_diekstrak.txt'")
