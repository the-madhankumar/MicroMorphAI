import cv2
import numpy as np
import scipy
import mahotas


class ContourDetector:
    """
    Detect and compute contour geometry + 131 micro-image descriptors.
    """

    def __init__(self, image: np.ndarray, min_area=200):
        self.min_area = min_area

        # ensure BGR format
        if len(image.shape) == 2:
            self.orig_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            self.orig_img = image.copy()

        self.img = cv2.cvtColor(self.orig_img, cv2.COLOR_BGR2GRAY)
        self._ensure_binary()

    def _ensure_binary(self):
        """Convert grayscale -> binary using Otsu only when required."""
        unique_vals = np.unique(self.img)

        if set(unique_vals).issubset({0, 255}):
            return

        _, th = cv2.threshold(
            self.img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        self.img = th

    def find_contours(self):
        contours, hierarchy = cv2.findContours(
            self.img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        self.contours = contours
        self.hierarchy = hierarchy
        return contours

    def compute_properties(self):
        props = []

        for cnt in self.contours:

            # ---------------- AREA ----------------
            area = cv2.contourArea(cnt)
            M = cv2.moments(cnt)
            area_moment = M["m00"]

            # ---------------- CENTROID -------------
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0  
            # ---------------- PERIMETER ----------------
            perimeter = cv2.arcLength(cnt, True)

            # ---------------- CIRCULARITY ----------------
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

            # ---------------- ELLIPSE FIT ----------------
            if len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                (xc, yc), (MA, ma), angle = ellipse
                major_axis = max(MA, ma)
                minor_axis = min(MA, ma)
                a, b = major_axis / 2, minor_axis / 2

                eccentricity = np.sqrt(1 - (b*b)/(a*a)) if a > 0 else 0
                roundness = minor_axis / major_axis if major_axis > 0 else 0
            else:
                major_axis = minor_axis = eccentricity = roundness = 0

            # ---------------- EQUIVALENT DIAMETER ----------------
            equivalent_diameter = np.sqrt(4 * area / np.pi) if area > 0 else 0

            # ---------------- BOUNDING BOX ----------------
            x, y, w, h = cv2.boundingRect(cnt)
            rectangularity = area / (w*h) if w*h > 0 else 0
            shape_factor = (perimeter * perimeter) / area if area > 0 else 0
            width_bb, height_bb = w, h

            # ---------------- CONVEXITY & SOLIDITY ----------------
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            convexity = cv2.arcLength(hull, True) / perimeter if perimeter > 0 else 0
            solidity = area / hull_area if hull_area > 0 else 0
            extent = area / (w*h) if w*h > 0 else 0

            # ---------------- HU MOMENTS ----------------
            hu = cv2.HuMoments(M).flatten()
            hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

            # ---------------- ZERNIKE MOMENTS ----------------
            mask = np.zeros(self.img.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)

            radius = min(mask.shape) // 2
            zernike = mahotas.features.zernike_moments(mask, radius)

            # ---------------- COLOR FEATURES ----------------
            img_bgr = self.orig_img.copy()
            b, g, r = cv2.split(img_bgr)
            eps = 1e-12

            ratio_rg = np.mean(r) / (np.mean(g) + eps)
            ratio_rb = np.mean(r) / (np.mean(b) + eps)
            ratio_bg = np.mean(b) / (np.mean(g) + eps)

            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            gray_mean = np.mean(gray)
            gray_std = np.std(gray)
            gray_skew = scipy.stats.skew(gray.reshape(-1))
            gray_kurt = scipy.stats.kurtosis(gray.reshape(-1))

            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_norm = hist / (hist.sum() + eps)
            entropy = -np.sum(hist_norm * np.log2(hist_norm + eps))

            # ---------------- HARALICK FEATURES ----------------
            gray_har = gray.astype(np.uint8)
            haralick_full = mahotas.features.haralick(gray_har, return_mean=False)
            har = haralick_full.mean(axis=0)

            # ---------------- LBP (54 FEATURES) ----------------
            from skimage.feature import local_binary_pattern
            P, R = 8, 1
            lbp = local_binary_pattern(gray, P, R, method="uniform")

            hist_lbp, _ = np.histogram(lbp.ravel(), bins=59, range=(0, 59))
            hist_lbp = hist_lbp.astype(float) / (hist_lbp.sum() + eps)
            lbp_54 = hist_lbp[:54]

            # ---------------- FOURIER DESCRIPTORS (10) ----------------
            cnt_np = cnt.squeeze()
            if len(cnt_np.shape) == 2 and cnt_np.shape[0] > 20:
                complex_cnt = cnt_np[:, 0] + 1j * cnt_np[:, 1]
                fd = np.fft.fft(complex_cnt)
                fd_mag = np.abs(fd)
                fourier_10 = fd_mag[1:11] / (fd_mag[1] + eps)
            else:
                fourier_10 = np.zeros(10)

            # ---------------- ASSEMBLE FEATURE DICT ----------------
            feature = {
                # Geometric (17)
                "area": area,
                "area_moment": area_moment,
                "perimeter": perimeter,
                "circularity": circularity,
                "major_axis": major_axis,
                "minor_axis": minor_axis,
                "eccentricity": eccentricity,
                "equivalent_diameter": equivalent_diameter,
                "width_bb": width_bb,
                "height_bb": height_bb,
                "rectangularity": rectangularity,
                "roundness": roundness,
                "shape_factor": shape_factor,
                "convexity": convexity,
                "solidity": solidity,
                "extent": extent,
                "centroid": (cx, cy),

                # Hu (7)
                **{f"hu_{i+1}": hu_log[i] for i in range(7)},

                # Zernike (25)
                **{f"z_{i}": zernike[i] for i in range(25)},

                # Color/Gray (8)
                "ratio_rg": ratio_rg,
                "ratio_rb": ratio_rb,
                "ratio_bg": ratio_bg,
                "gray_mean": gray_mean,
                "gray_std": gray_std,
                "gray_skew": gray_skew,
                "gray_kurt": gray_kurt,
                "gray_entropy": entropy,

                # Haralick (13)
                **{f"h_{i}": har[i] for i in range(13)},

                # LBP (54)
                **{f"lbp_{i}": lbp_54[i] for i in range(54)},

                # Fourier (10)
                **{f"fourier_{i}": fourier_10[i] for i in range(10)},

                "contour": cnt
            }

            props.append(feature)

        return props

    def get_big_contours(self):
        return [p for p in self.compute_properties() if p["area"] >= self.min_area]

    def draw_big_contours(self, base_image):
        """Draw only large contours on the image with labels."""
        if len(base_image.shape) == 2:
            base_image = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)

        img_out = base_image.copy()
        filtered = self.get_big_contours()

        for p in filtered:
            cnt = p["contour"]
            cx, cy = p["centroid"]

            cv2.drawContours(img_out, [cnt], -1, (0, 255, 0), 2)

            txt = [
                f"Area:{p['area']:.0f}",
                f"Circ:{p['circularity']:.2f}",
                f"Ecc:{p['eccentricity']:.2f}",
                f"Sol:{p['solidity']:.2f}",
            ]

            ypos = 0
            for t in txt:
                cv2.putText(
                    img_out, t, (cx, cy + ypos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 255), 1, cv2.LINE_AA
                )
                ypos += 15

        return img_out

    def show(self, base_image):
        img = self.draw_big_contours(base_image)
        cv2.imshow("131 Feature Contours", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
