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

            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter ** 2)
            else:
                circularity = 0

            if len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                (x_c, y_c), (major_axis, minor_axis), angle = ellipse

                a = max(major_axis, minor_axis) / 2
                b = min(major_axis, minor_axis) / 2

                if a > 0:
                    eccentricity = np.sqrt(1 - (b * b) / (a * a))
                else:
                    eccentricity = 0
            else:
                major_axis = 0
                minor_axis = 0
                eccentricity = 0

            hu = cv2.HuMoments(cv2.moments(cnt)).flatten()
            hu1 = hu[0]  

            M = cv2.moments(cnt)
            cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else -1
            cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else -1

            epsilon = 0.01 * perimeter
            approx = cv2.approxPolyDP(cnt, epsilon, True)

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
                "hu1": hu1
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

            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter ** 2)
            else:
                circularity = 0

            if len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                (x_c, y_c), (major_axis, minor_axis), angle = ellipse

                # eccentricity = sqrt(1 - (b^2 / a^2))
                a = max(major_axis, minor_axis) / 2
                b = min(major_axis, minor_axis) / 2
                if a > 0:
                    eccentricity = np.sqrt(1 - (b * b) / (a * a))
                else:
                    eccentricity = 0
            else:
                major_axis = minor_axis = eccentricity = 0

            hu = cv2.HuMoments(cv2.moments(cnt)).flatten()
            hu1 = hu[0]  

            lines = [
                f"A:{int(area)} P:{int(perimeter)}",
                f"Circ:{circularity:.2f}",
                f"Ecc:{eccentricity:.2f}",
                f"Maj:{major_axis:.1f} Min:{minor_axis:.1f}",
                f"Hu1:{hu1:.3e}"
            ]

            y_offset = 0
            for line in lines:
                cv2.putText(
                    img_out,
                    line,
                    (cx - 60, cy + y_offset),
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
