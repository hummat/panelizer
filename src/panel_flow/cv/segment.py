import math
from typing import List, Optional, Tuple

import numpy as np

Point = Tuple[int, int]


class Segment:
    """Line segment between two integer-coordinate points."""

    def __init__(self, a: Tuple[int, int], b: Tuple[int, int]) -> None:
        self.a: Point = (int(a[0]), int(a[1]))
        self.b: Point = (int(b[0]), int(b[1]))

        for dot in [self.a, self.b]:
            if len(dot) != 2:
                raise ValueError(f"Creating a segment with more or less than two dots: Segment({a}, {b})")
            if not isinstance(dot[0], int) or not isinstance(dot[1], int):
                raise ValueError(f"Creating a segment with non-integer coordinates: Segment({a}, {b})")

    def __str__(self) -> str:
        return f"({self.a}, {self.b})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Segment):
            return False
        return any(
            [
                self.a == other.a and self.b == other.b,
                self.a == other.b and self.b == other.a,
            ]
        )

    def dist(self) -> float:
        """Euclidean distance between endpoints."""
        return math.sqrt(self.dist_x() ** 2 + self.dist_y() ** 2)

    def dist_x(self, keep_sign: bool = False) -> float:
        """Horizontal distance (absolute or signed)."""
        dist = self.b[0] - self.a[0]
        return dist if keep_sign else abs(dist)

    def dist_y(self, keep_sign: bool = False) -> float:
        """Vertical distance (absolute or signed)."""
        dist = self.b[1] - self.a[1]
        return dist if keep_sign else abs(dist)

    def left(self) -> int:
        return min(self.a[0], self.b[0])

    def top(self) -> int:
        return min(self.a[1], self.b[1])

    def right(self) -> int:
        return max(self.a[0], self.b[0])

    def bottom(self) -> int:
        return max(self.a[1], self.b[1])

    def to_xyrb(self) -> List[int]:
        """Return [left, top, right, bottom]."""
        return [self.left(), self.top(), self.right(), self.bottom()]

    def center(self) -> Point:
        """Midpoint of segment."""
        return (
            int(self.left() + self.dist_x() / 2),
            int(self.top() + self.dist_y() / 2),
        )

    def may_contain(self, dot: Point) -> bool:
        """Check if point is within segment's bounding box."""
        return all(
            [
                dot[0] >= self.left(),
                dot[0] <= self.right(),
                dot[1] >= self.top(),
                dot[1] <= self.bottom(),
            ]
        )

    def intersect(self, other: "Segment") -> Optional["Segment"]:
        """
        Return overlapping portion of two segments if they are nearly parallel and close.
        Uses 5% gutter tolerance and 10-degree angle threshold.
        """
        gutter = max(self.dist(), other.dist()) * 5 / 100

        # angle too big ?
        if not self.angle_ok_with(other):
            return None

        # from here, segments are almost parallel

        # segments are apart ?
        if any(
            [
                self.right() < other.left() - gutter,  # self left from other
                self.left() > other.right() + gutter,  # self right from other
                self.bottom() < other.top() - gutter,  # self above other
                self.top() > other.bottom() + gutter,  # self below other
            ]
        ):
            return None

        projected_c = self.projected_point(other.a)
        dist_c_to_ab = Segment(other.a, projected_c).dist()

        projected_d = self.projected_point(other.b)
        dist_d_to_ab = Segment(other.b, projected_d).dist()

        # segments are a bit too far from each other
        if (dist_c_to_ab + dist_d_to_ab) / 2 > gutter:
            return None

        # segments overlap, or one contains the other
        #  A----B
        #     C----D
        # or
        #  A------------B
        #      C----D
        sorted_dots = sorted([self.a, self.b, other.a, other.b], key=sum)
        middle_dots = sorted_dots[1:3]
        b, c = middle_dots

        return Segment(b, c)

    def union(self, other: "Segment") -> Optional["Segment"]:
        """
        Merge two overlapping/parallel segments into one.
        Returns None if they don't intersect.
        """
        intersect = self.intersect(other)
        if intersect is None:
            return None

        dots: List[Point] = [self.a, self.b, other.a, other.b]
        dots.remove(intersect.a)
        dots.remove(intersect.b)
        return Segment(dots[0], dots[1])

    def angle_with(self, other: "Segment") -> float:
        """Angle difference in degrees."""
        return math.degrees(abs(self.angle() - other.angle()))

    def angle_ok_with(self, other: "Segment") -> bool:
        """Check if segments are nearly parallel (< 10 degrees difference)."""
        angle = self.angle_with(other)
        return angle < 10 or abs(angle - 180) < 10

    def angle(self) -> float:
        """Angle in radians."""
        return math.atan(self.dist_y() / self.dist_x()) if self.dist_x() != 0 else math.pi / 2

    def intersect_all(self, segments: List["Segment"]) -> List["Segment"]:
        """Find all segments that intersect with this one, return unified result."""
        segments_match = []
        for segment in segments:
            s3 = self.intersect(segment)
            if s3 is not None:
                segments_match.append(s3)

        return Segment.union_all(segments_match)

    @staticmethod
    def along_polygon(polygon: np.ndarray, i: int, j: int) -> "Segment":
        """
        Extend segment through adjacent collinear polygon edges.
        Walks backward from i and forward from j while edges remain parallel.
        """
        dot1 = polygon[i][0]
        dot2 = polygon[j][0]
        split_segment = Segment(dot1, dot2)

        while True:
            i = (i - 1) % len(polygon)
            add_segment = Segment(polygon[i][0], polygon[(i + 1) % len(polygon)][0])
            if add_segment.angle_ok_with(split_segment):
                split_segment = Segment(add_segment.a, split_segment.b)
            else:
                break

        while True:
            j = (j + 1) % len(polygon)
            add_segment = Segment(polygon[(j - 1) % len(polygon)][0], polygon[j][0])
            if add_segment.angle_ok_with(split_segment):
                split_segment = Segment(split_segment.a, add_segment.b)
            else:
                break

        return split_segment

    @staticmethod
    def union_all(segments: List["Segment"]) -> List["Segment"]:
        """
        Iteratively merge overlapping/parallel segments.
        Continues until no more segments can be merged.
        """
        unioned_segments = True
        while unioned_segments:
            unioned_segments = False
            dedup_segments = []
            used = []
            for i, s1 in enumerate(segments):
                for s2 in segments[i + 1 :]:
                    if s2 in used:
                        continue

                    s3 = s1.union(s2)
                    if s3 is not None:
                        unioned_segments = True
                        dedup_segments += [s3]
                        used.append(s1)
                        used.append(s2)
                        break

                if s1 not in used:
                    dedup_segments += [s1]

            segments = dedup_segments

        return dedup_segments

    def projected_point(self, p: Point) -> Point:
        """Project point p onto the infinite line defined by this segment."""
        a = np.array(self.a)
        b = np.array(self.b)
        p_arr = np.array(p)
        ap = p_arr - a
        ab = b - a
        if ab[0] == 0 and ab[1] == 0:
            return self.a
        result = a + np.dot(ap, ab) / np.dot(ab, ab) * ab
        return (round(result[0]), round(result[1]))
