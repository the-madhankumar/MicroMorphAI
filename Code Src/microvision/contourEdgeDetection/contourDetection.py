import cv2
import numpy as np


class ContourDetector:
    """
    Detect and compute contour geometry.
    This version ONLY shows and returns large contours (noise removed).
    """

    def __init__(self, image: np.ndarray, min_area=200):
        self.min_area = min_area

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        self.img = image.copy()
        self._ensure_binary()

    def _ensure_binary(self):
        """Convert image to binary using Otsu only if not already binary."""
        unique_vals = np.unique(self.img)

        if set(unique_vals).issubset({0, 255}):
            return

        _, th = cv2.threshold(
            self.img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        self.img = th

    def find_contours(self):
        """Find all contours."""
        contours, hierarchy = cv2.findContours(
            self.img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        self.contours = contours
        self.hierarchy = hierarchy
        return contours

    def compute_properties(self):
        """Compute area, perimeter, centroid, approx — for all contours."""
        props = []

        for cnt in self.contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)

            # ---- Circularity ----
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

            # ---- Fit ellipse for major/minor axis + eccentricity ----
            if len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                (x_c, y_c), (major_axis, minor_axis), angle = ellipse

                a = max(major_axis, minor_axis) / 2
                b = min(major_axis, minor_axis) / 2
                eccentricity = np.sqrt(1 - (b * b) / (a * a)) if a > 0 else 0
            else:
                major_axis = minor_axis = eccentricity = 0

            # ---- Hu moments (log scale) ----
            hu = cv2.HuMoments(cv2.moments(cnt)).flatten()
            hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

            # ---- Centroid ----
            M = cv2.moments(cnt)
            cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else -1
            cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else -1

            # ---- Polygon approximation ----
            epsilon = 0.01 * perimeter
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            # ---- Aspect Ratio ----
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h if h > 0 else 0

            # ---- Solidity (area/convex hull area) ----
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0

            # ---- Extent (area / bounding box area) ----
            bbox_area = w * h
            extent = area / bbox_area if bbox_area > 0 else 0

            # ---- Compactness (perimeter² / area) ----
            compactness = (perimeter ** 2) / area if area > 0 else 0

            props.append({
                "contour": cnt,
                "area": area,
                "perimeter": perimeter,
                "centroid": (cx, cy),
                "approx": approx,
                "circularity": circularity,
                "eccentricity": eccentricity,
                "major_axis": major_axis,
                "minor_axis": minor_axis,
                "hu_log": hu_log,
                "aspect_ratio": aspect_ratio,
                "solidity": solidity,
                "extent": extent,
                "compactness": compactness
            })

        return props

    def get_big_contours(self):
        """Return contours above min_area only."""
        props = self.compute_properties()
        return [p for p in props if p["area"] >= self.min_area]

    def draw_big_contours(self, base_image):
        """Draw only large contours on the image."""
        if len(base_image.shape) == 2:
            base_image = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)

        big = self.get_big_contours()
        img_out = base_image.copy()

        for p in big:
            cnt = p["contour"]
            area = p["area"]
            perimeter = p["perimeter"]
            cx, cy = p["centroid"]

            cv2.drawContours(img_out, [cnt], -1, (0, 255, 0), 2)

            lines = [
                f"A:{int(area)} P:{int(perimeter)}",
                f"Circ:{p['circularity']:.2f}",
                f"Ecc:{p['eccentricity']:.2f}",
                f"AR:{p['aspect_ratio']:.2f}",
                f"Sol:{p['solidity']:.2f}",
                f"Ext:{p['extent']:.2f}",
                f"Comp:{p['compactness']:.2f}",
                f"Hu1(log):{p['hu_log'][0]:.1f}"
            ]

            y_offset = 0
            for line in lines:
                cv2.putText(
                    img_out,
                    line,
                    (cx - 70, cy + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA
                )
                y_offset += 15

        return img_out

    def show(self, base_image, title="Large Contours Only"):
        """Show only large contours."""
        img = self.draw_big_contours(base_image)
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
