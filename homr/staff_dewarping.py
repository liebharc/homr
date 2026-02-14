import cv2
import numpy as np
import PIL.Image

from homr.debug import Debug
from homr.model import Staff
from homr.simple_logging import eprint
from homr.type_definitions import NDArray


class DelaunayTriangulation:
    """OpenCV-based Delaunay triangulation using Subdiv2D"""

    def __init__(self, points: NDArray) -> None:
        self.points = np.array(points, dtype=np.float32)
        self.simplices: NDArray | None = None
        self._triangulate()

    def _triangulate(self) -> None:
        """Perform Delaunay triangulation using OpenCV's Subdiv2D"""
        points = self.points
        n = len(points)

        if n < 3:
            raise ValueError("Need at least 3 points for triangulation")

        # Create subdivision with expanded bounding rectangle
        rect = cv2.boundingRect(points.reshape(-1, 1, 2))
        rect = (rect[0] - 10, rect[1] - 10, rect[2] + 20, rect[3] + 20)
        subdiv = cv2.Subdiv2D(rect)

        # Insert all points
        for point in points:
            subdiv.insert((float(point[0]), float(point[1])))

        # Get triangles and convert from coordinates to point indices
        triangle_list = subdiv.getTriangleList()
        triangles = []

        for t in triangle_list:
            pt1 = np.array([t[0], t[1]], dtype=np.float32)
            pt2 = np.array([t[2], t[3]], dtype=np.float32)
            pt3 = np.array([t[4], t[5]], dtype=np.float32)

            idx1 = self._find_point_index(pt1, points)
            idx2 = self._find_point_index(pt2, points)
            idx3 = self._find_point_index(pt3, points)

            if idx1 is not None and idx2 is not None and idx3 is not None:
                triangles.append([idx1, idx2, idx3])

        self.simplices = np.array(triangles, dtype=np.int32)

    def _find_point_index(
        self, point: NDArray, points: NDArray, tolerance: float = 1e-3
    ) -> int | None:
        """Find the index of a point in the points array"""
        distances = np.linalg.norm(points - point, axis=1)
        min_idx = np.argmin(distances)
        if distances[min_idx] < tolerance:
            return int(min_idx)
        return None

    def find_simplex(self, points: NDArray) -> NDArray:
        """Find which triangle contains each point"""
        if self.simplices is None:
            return np.full(len(points), -1, dtype=np.int32)

        points = np.atleast_2d(points)
        result = np.full(len(points), -1, dtype=np.int32)

        for i, point in enumerate(points):
            for j, simplex in enumerate(self.simplices):
                triangle = self.points[simplex]
                if self._point_in_triangle(point, triangle):
                    result[i] = j
                    break

        return result

    def _point_in_triangle(self, point: NDArray, triangle: NDArray) -> bool:
        """Check if point is inside triangle using barycentric coordinates"""
        x, y = point
        x0, y0 = triangle[0]
        x1, y1 = triangle[1]
        x2, y2 = triangle[2]

        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-10:
            return False

        a = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denom
        b = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denom
        c = 1 - a - b

        return a >= -1e-10 and b >= -1e-10 and c >= -1e-10


class PiecewiseAffineTransform:
    """Piecewise affine transformation using triangulation"""

    def __init__(self) -> None:
        self.src_points: NDArray | None = None
        self.dst_points: NDArray | None = None
        self.triangulation: DelaunayTriangulation | None = None
        self.affine_matrices: list[NDArray | None] = []

    def estimate(self, src: NDArray, dst: NDArray) -> None:
        """Estimate the piecewise affine transformation from source to destination points"""
        self.src_points = np.array(src, dtype=np.float32)
        self.dst_points = np.array(dst, dtype=np.float32)

        self.triangulation = DelaunayTriangulation(self.src_points)

        # Compute affine transform for each triangle
        self.affine_matrices = []
        if self.triangulation.simplices is not None:
            for simplex in self.triangulation.simplices:
                src_tri = self.src_points[simplex]
                dst_tri = self.dst_points[simplex]

                if self._is_degenerate_triangle(src_tri) or self._is_degenerate_triangle(dst_tri):
                    self.affine_matrices.append(None)
                    continue

                try:
                    mat = cv2.getAffineTransform(src_tri, dst_tri)
                    self.affine_matrices.append(mat)
                except cv2.error:
                    self.affine_matrices.append(None)

    def transform_point(self, point: tuple[float, float]) -> tuple[float, float]:
        """Transform a single point using the piecewise affine transformation"""
        if self.triangulation is None:
            return point

        point_array = np.array([[point[0], point[1]]], dtype=np.float32)
        simplex_idx = self.triangulation.find_simplex(point_array)[0]

        if simplex_idx == -1 or simplex_idx >= len(self.affine_matrices):
            return point

        mat = self.affine_matrices[simplex_idx]
        if mat is None:
            return point

        point_homogeneous = np.array([point[0], point[1], 1.0], dtype=np.float32)
        transformed = mat @ point_homogeneous

        return (float(transformed[0]), float(transformed[1]))

    def warp_image(self, image: NDArray, fill_color: int = 1, order: int = 1) -> NDArray:
        """Warp an image using the piecewise affine transformation"""
        if (
            self.triangulation is None
            or self.triangulation.simplices is None
            or self.src_points is None
            or self.dst_points is None
        ):
            return image

        # Assign to local variables to help type checker
        src_points = self.src_points
        dst_points = self.dst_points

        height, width = image.shape[:2]
        output = np.full_like(image, fill_color)

        for simplex_idx, simplex in enumerate(self.triangulation.simplices):
            if self.affine_matrices[simplex_idx] is None:
                continue

            src_tri = src_points[simplex].astype(np.float32)
            dst_tri = dst_points[simplex].astype(np.float32)

            if self._is_degenerate_triangle(src_tri) or self._is_degenerate_triangle(dst_tri):
                continue

            src_rect = cv2.boundingRect(src_tri.reshape(-1, 1, 2))
            dst_rect = cv2.boundingRect(dst_tri.reshape(-1, 1, 2))

            if src_rect[2] <= 0 or src_rect[3] <= 0 or dst_rect[2] <= 0 or dst_rect[3] <= 0:
                continue

            # Offset triangles to bounding rectangle origin
            src_tri_cropped = src_tri - np.array([src_rect[0], src_rect[1]], dtype=np.float32)
            dst_tri_cropped = dst_tri - np.array([dst_rect[0], dst_rect[1]], dtype=np.float32)

            if self._is_degenerate_triangle(src_tri_cropped) or self._is_degenerate_triangle(
                dst_tri_cropped
            ):
                continue

            try:
                warp_mat = cv2.getAffineTransform(src_tri_cropped, dst_tri_cropped)
            except cv2.error:
                continue

            src_cropped = image[
                src_rect[1] : src_rect[1] + src_rect[3], src_rect[0] : src_rect[0] + src_rect[2]
            ]

            if src_cropped.size == 0:
                continue

            interpolation = cv2.INTER_NEAREST if order == 0 else cv2.INTER_LINEAR
            dst_cropped = cv2.warpAffine(
                src_cropped,
                warp_mat,
                (dst_rect[2], dst_rect[3]),
                flags=interpolation,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=fill_color,
            )

            # Create triangle mask
            triangle_mask = np.zeros((dst_rect[3], dst_rect[2]), dtype=np.uint8)
            cv2.fillConvexPoly(triangle_mask, dst_tri_cropped.astype(np.int32), 255)

            # Clip to image bounds
            y1, y2 = dst_rect[1], dst_rect[1] + dst_rect[3]
            x1, x2 = dst_rect[0], dst_rect[0] + dst_rect[2]
            y1_clip, y2_clip = max(0, y1), min(height, y2)
            x1_clip, x2_clip = max(0, x1), min(width, x2)

            if y2_clip <= y1_clip or x2_clip <= x1_clip:
                continue

            dy1, dy2 = y1_clip - y1, y2_clip - y1
            dx1, dx2 = x1_clip - x1, x2_clip - x1

            if len(image.shape) == 3:
                for c in range(image.shape[2]):
                    output[y1_clip:y2_clip, x1_clip:x2_clip, c] = np.where(
                        triangle_mask[dy1:dy2, dx1:dx2] > 0,
                        dst_cropped[dy1:dy2, dx1:dx2, c],
                        output[y1_clip:y2_clip, x1_clip:x2_clip, c],
                    )
            else:
                output[y1_clip:y2_clip, x1_clip:x2_clip] = np.where(
                    triangle_mask[dy1:dy2, dx1:dx2] > 0,
                    dst_cropped[dy1:dy2, dx1:dx2],
                    output[y1_clip:y2_clip, x1_clip:x2_clip],
                )

        return output

    @staticmethod
    def _is_degenerate_triangle(triangle: NDArray) -> bool:
        """Check if triangle has near-zero area"""
        v1 = triangle[1] - triangle[0]
        v2 = triangle[2] - triangle[0]
        area = abs(v1[0] * v2[1] - v1[1] * v2[0]) / 2.0
        return area < 1e-6

    @staticmethod
    def _triangle_area(triangle: NDArray) -> float:
        """Calculate area of a triangle"""
        v1 = triangle[1] - triangle[0]
        v2 = triangle[2] - triangle[0]
        return abs(v1[0] * v2[1] - v1[1] * v2[0]) / 2.0


class StaffDewarping:
    def __init__(self, tform: PiecewiseAffineTransform | None) -> None:
        self.tform = tform

    def dewarp(self, image: NDArray, fill_color: int = 1, order: int = 1) -> NDArray:
        if self.tform is None:
            return image
        return self.tform.warp_image(image, fill_color, order)

    def dewarp_point(self, point: tuple[float, float]) -> tuple[float, float]:
        if self.tform is None:
            return point
        return self.tform.transform_point(point)


def is_point_on_image(pts: tuple[int, int], image: NDArray) -> bool:
    height, width = image.shape[:2]
    margin = 10
    if pts[0] < margin or pts[0] > width - margin or pts[1] < margin or pts[1] > height - margin:
        return False
    return True


def calculate_span_and_optimal_points(
    staff: Staff, image: NDArray
) -> tuple[list[list[tuple[int, int]]], list[list[tuple[int, int]]]]:
    span_points: list[list[tuple[int, int]]] = []
    optimal_points: list[list[tuple[int, int]]] = []
    first_y_offset = None
    number_of_y_intervals = 6

    if int(image.shape[0] / number_of_y_intervals) == 0:
        return span_points, optimal_points

    for y in range(2, image.shape[0] - 2, int(image.shape[0] / number_of_y_intervals)):
        line_points: list[tuple[int, int]] = []
        for x in range(2, image.shape[1], 80):
            y_values = staff.get_at(x)
            if y_values is not None:
                y_offset = y_values.y[2]
                if not first_y_offset:
                    first_y_offset = y_offset
                    y_delta = 0
                else:
                    y_delta = int(y_offset - first_y_offset)
                point = (x, y + y_delta)
                if is_point_on_image(point, image):
                    line_points.append(point)

        minimum_number_of_points = 2
        if len(line_points) > minimum_number_of_points:
            average_y = sum([p[1] for p in line_points]) / len(line_points)
            span_points.append(line_points)
            optimal_points.append([(p[0], int(average_y)) for p in line_points])

    return span_points, optimal_points


def calculate_dewarp_transformation(
    image: NDArray, source: list[list[tuple[int, int]]], destination: list[list[tuple[int, int]]]
) -> StaffDewarping:
    def add_image_edges_to_lines(
        lines: list[list[tuple[int, int]]],
    ) -> list[list[tuple[int, int]]]:
        lines.insert(0, [(0, 0), (image.shape[1], 0)])
        lines.append([(0, image.shape[0]), (image.shape[1], image.shape[0])])
        return lines

    def add_first_and_last_point_to_every_line(
        lines: list[list[tuple[int, int]]],
    ) -> list[list[tuple[int, int]]]:
        for line in lines:
            line.insert(0, (0, line[0][1]))
            line.append((image.shape[1], line[-1][1]))
        return lines

    source = add_first_and_last_point_to_every_line(source)
    destination = add_first_and_last_point_to_every_line(destination)

    source = add_image_edges_to_lines(source)
    destination = add_image_edges_to_lines(destination)

    source_conc = np.concatenate(source)
    destination_conc = np.concatenate(destination)

    tform = PiecewiseAffineTransform()
    tform.estimate(source_conc, destination_conc)
    return StaffDewarping(tform)


def dewarp_staff_image(image: NDArray, staff: Staff, index: int, debug: Debug) -> StaffDewarping:
    try:
        span_points, optimal_points = calculate_span_and_optimal_points(staff, image)

        if debug.debug:
            debug_img = image.copy()
            for line in span_points:
                for point in line:
                    cv2.circle(debug_img, [int(point[0]), int(point[1])], 5, (0, 0, 255), -1)
            for line in optimal_points:
                for point in line:
                    cv2.circle(debug_img, [int(point[0]), int(point[1])], 5, (255, 0, 0), -1)
            debug.write_image_with_fixed_suffix(f"_staff-{index}_debug_span_points.png", debug_img)

        return calculate_dewarp_transformation(image, span_points, optimal_points)
    except Exception as e:
        eprint("Dewarping failed for staff", index, "with error", e)
        return StaffDewarping(None)


def warp_image_randomly(image: PIL.Image.Image) -> PIL.Image.Image:
    array = np.array(image)
    result = warp_image_array_randomly(array)
    return PIL.Image.fromarray(result)


def warp_image_array_randomly(image: NDArray) -> NDArray:
    center = (image.shape[1] // 2, image.shape[0] // 2)
    num_points = 5
    upper = [(i * image.shape[1] // num_points, 0) for i in range(num_points)]
    source = [(i * image.shape[1] // num_points, center[1]) for i in range(num_points)]
    lower = [(i * image.shape[1] // num_points, image.shape[0]) for i in range(num_points)]
    max_random_offset = 20
    destination = [
        (
            i * image.shape[1] // num_points,
            center[1] + np.random.randint(-max_random_offset, max_random_offset),
        )
        for i in range(num_points)
    ]
    result = calculate_dewarp_transformation(
        image, [upper, source, lower], [upper, destination, lower]
    ).dewarp(image, fill_color=255, order=3)

    return result.astype(np.uint8)


def warp_image_array_randomly2(image: NDArray) -> NDArray:
    """
    Apply a smooth random warp to the image to simulate paper folding/bending.
    Uses cv2.remap with a 1D vertical displacement pattern (tiled horizontally)
    to create a realistic wavy look without "hard cuts" or 2D stretching.
    """
    height, width = image.shape[:2]

    # Use a small grid: 5 columns, 2 identical rows for vertical-only wave
    grid_w = 5
    grid_h = 2

    # Generate random vertical displacement (same for both rows)
    # alpha controls the maximum displacement in pixels
    alpha = np.random.uniform(5, 25)
    dy_row = np.random.uniform(-1, 1, (1, grid_w)).astype(np.float32)

    # Anchor left and right edges to zero displacement
    dy_row[0, 0] = 0
    dy_row[0, -1] = 0

    # Tile vertically to create a consistent wave across the entire image height
    dy_small = np.tile(dy_row, (grid_h, 1))
    dx_small = np.zeros((grid_h, grid_w), dtype=np.float32)

    # Upscale to full image size with cubic interpolation for smoothness
    dx = cv2.resize(dx_small, (width, height), interpolation=cv2.INTER_CUBIC) * alpha
    dy = cv2.resize(dy_small, (width, height), interpolation=cv2.INTER_CUBIC) * alpha

    # Create coordinate maps for cv2.remap
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    # Perform the warping
    result = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    return result.astype(np.uint8)


if __name__ == "__main__":
    import sys

    image = cv2.imread(sys.argv[1])
    if image is None:
        raise ValueError("Failed to read " + sys.argv[1])
    cv2.imwrite(sys.argv[2], warp_image_array_randomly(image))
