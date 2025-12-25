import math
from typing import List, Optional, Tuple

import cv2 as cv
import numpy as np

from .segment import Segment


class InternalPanel:
    """
    Internal panel representation with polygon support for detection pipeline.
    Converts to schema.Panel with bbox for output.
    """

    def __init__(
        self,
        img_size: Tuple[int, int],
        small_panel_ratio: float,
        xywh: Optional[Tuple[int, int, int, int]] = None,
        polygon: Optional[np.ndarray] = None,
        splittable: bool = True,
    ) -> None:
        """
        Create panel from either xywh or polygon.
        img_size: (width, height) of source image
        small_panel_ratio: minimum panel size as fraction of image dimensions
        """
        self.img_size = img_size
        self.small_panel_ratio = small_panel_ratio

        if xywh is None and polygon is None:
            raise ValueError("Panel requires either xywh or polygon")

        computed_xywh: Tuple[int, int, int, int]
        if xywh is None:
            if polygon is None:
                raise ValueError("Panel requires either xywh or polygon")
            rect = cv.boundingRect(polygon)
            computed_xywh = (rect[0], rect[1], rect[2], rect[3])
        else:
            computed_xywh = xywh

        self.x: int = computed_xywh[0]  # panel's left edge
        self.y: int = computed_xywh[1]  # panel's top edge
        self.r: int = self.x + computed_xywh[2]  # panel's right edge
        self.b: int = self.y + computed_xywh[3]  # panel's bottom edge

        self.polygon = polygon
        self.splittable = splittable
        self.segments: Optional[List[Segment]] = None

    @staticmethod
    def from_xyrb(
        img_size: Tuple[int, int],
        small_panel_ratio: float,
        x: int,
        y: int,
        r: int,
        b: int,
    ) -> "InternalPanel":
        """Create panel from left, top, right, bottom coordinates."""
        return InternalPanel(img_size, small_panel_ratio, xywh=(x, y, r - x, b - y))

    def w(self) -> int:
        return self.r - self.x

    def h(self) -> int:
        return self.b - self.y

    def diagonal(self) -> Segment:
        return Segment((self.x, self.y), (self.r, self.b))

    def wt(self) -> float:
        """Width threshold (under which two edge coordinates are considered equal)."""
        return self.w() / 10

    def ht(self) -> float:
        """Height threshold."""
        return self.h() / 10

    def to_xywh(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w(), self.h())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, InternalPanel):
            return False
        return all(
            [
                abs(self.x - other.x) < self.wt(),
                abs(self.y - other.y) < self.ht(),
                abs(self.r - other.r) < self.wt(),
                abs(self.b - other.b) < self.ht(),
            ]
        )

    def area(self) -> int:
        return self.w() * self.h()

    def __str__(self) -> str:
        return f"{self.x}x{self.y}-{self.r}x{self.b}"

    def __hash__(self) -> int:
        return hash(self.__str__())

    def is_small(self, extra_ratio: float = 1.0) -> bool:
        """Check if panel is smaller than minimum size threshold."""
        return any(
            [
                self.w() < self.img_size[0] * self.small_panel_ratio * extra_ratio,
                self.h() < self.img_size[1] * self.small_panel_ratio * extra_ratio,
            ]
        )

    def is_very_small(self) -> bool:
        return self.is_small(1 / 10)

    def overlap_panel(self, other: "InternalPanel") -> Optional["InternalPanel"]:
        """Return the overlapping region of two panels, or None if they don't overlap."""
        if self.x > other.r or other.x > self.r:  # panels are left and right from one another
            return None
        if self.y > other.b or other.y > self.b:  # panels are above and below one another
            return None

        # if we're here, panels overlap at least a bit
        x = max(self.x, other.x)
        y = max(self.y, other.y)
        r = min(self.r, other.r)
        b = min(self.b, other.b)

        return InternalPanel(self.img_size, self.small_panel_ratio, xywh=(x, y, r - x, b - y))

    def overlap_area(self, other: "InternalPanel") -> int:
        opanel = self.overlap_panel(other)
        if opanel is None:
            return 0
        return opanel.area()

    def overlaps(self, other: "InternalPanel") -> bool:
        """Check if panels overlap significantly (> 10% of smaller panel)."""
        opanel = self.overlap_panel(other)
        if opanel is None:
            return False

        area_ratio = 0.1
        smallest_panel_area = min(self.area(), other.area())

        if smallest_panel_area == 0:  # probably a horizontal or vertical segment
            return True

        return opanel.area() / smallest_panel_area > area_ratio

    def contains(self, other: "InternalPanel") -> bool:
        """Check if this panel contains more than 50% of another panel."""
        o_panel = self.overlap_panel(other)
        if not o_panel:
            return False

        # self contains other if their overlapping area is more than 50% of other's area
        return o_panel.area() / other.area() > 0.50

    def same_row(self, other: "InternalPanel") -> bool:
        """Check if two panels are in the same row (vertically aligned)."""
        above, below = sorted([self, other], key=lambda p: p.y)

        if below.y > above.b:  # strictly above
            return False

        if below.b < above.b:  # contained
            return True

        # intersect
        intersection_y = min(above.b, below.b) - below.y
        min_h = min(above.h(), below.h())
        return min_h == 0 or intersection_y / min_h >= 1 / 3

    def same_col(self, other: "InternalPanel") -> bool:
        """Check if two panels are in the same column (horizontally aligned)."""
        left, right = sorted([self, other], key=lambda p: p.x)

        if right.x > left.r:  # strictly left
            return False

        if right.r < left.r:  # contained
            return True

        # intersect
        intersection_x = min(left.r, right.r) - right.x
        min_w = min(left.w(), right.w())
        return min_w == 0 or intersection_x / min_w >= 1 / 3

    def find_top_panel(self, all_panels: List["InternalPanel"]) -> Optional["InternalPanel"]:
        """Find the panel directly above this one in the same column."""
        all_top = [p for p in all_panels if p.b <= self.y and p.same_col(self)]
        return max(all_top, key=lambda p: p.b) if all_top else None

    def find_bottom_panel(self, all_panels: List["InternalPanel"]) -> Optional["InternalPanel"]:
        """Find the panel directly below this one in the same column."""
        all_bottom = [p for p in all_panels if p.y >= self.b and p.same_col(self)]
        return min(all_bottom, key=lambda p: p.y) if all_bottom else None

    def find_all_left_panels(self, all_panels: List["InternalPanel"]) -> List["InternalPanel"]:
        """Find all panels to the left of this one in the same row."""
        return [p for p in all_panels if p.r <= self.x and p.same_row(self)]

    def find_left_panel(self, all_panels: List["InternalPanel"]) -> Optional["InternalPanel"]:
        """Find the panel directly to the left of this one."""
        all_left = self.find_all_left_panels(all_panels)
        return max(all_left, key=lambda p: p.r) if all_left else None

    def find_all_right_panels(self, all_panels: List["InternalPanel"]) -> List["InternalPanel"]:
        """Find all panels to the right of this one in the same row."""
        return [p for p in all_panels if p.x >= self.r and p.same_row(self)]

    def find_right_panel(self, all_panels: List["InternalPanel"]) -> Optional["InternalPanel"]:
        """Find the panel directly to the right of this one."""
        all_right = self.find_all_right_panels(all_panels)
        return min(all_right, key=lambda p: p.x) if all_right else None

    def group_with(self, other: "InternalPanel") -> "InternalPanel":
        """Create a panel that encompasses both this panel and another."""
        min_x = min(self.x, other.x)
        min_y = min(self.y, other.y)
        max_r = max(self.r, other.r)
        max_b = max(self.b, other.b)
        return InternalPanel(
            self.img_size, self.small_panel_ratio, xywh=(min_x, min_y, max_r - min_x, max_b - min_y)
        )

    def merge(self, other: "InternalPanel", all_panels: List["InternalPanel"]) -> "InternalPanel":
        """
        Merge this panel with another by expanding in all directions where other is.
        Returns the largest merged panel that doesn't bump into other panels.
        """
        possible_panels: List[InternalPanel] = [self]

        # expand self in all four directions where other is
        if other.x < self.x:
            possible_panels.append(
                InternalPanel.from_xyrb(
                    self.img_size, self.small_panel_ratio, other.x, self.y, self.r, self.b
                )
            )

        if other.r > self.r:
            for pp in possible_panels.copy():
                possible_panels.append(
                    InternalPanel.from_xyrb(
                        self.img_size, self.small_panel_ratio, pp.x, pp.y, other.r, pp.b
                    )
                )

        if other.y < self.y:
            for pp in possible_panels.copy():
                possible_panels.append(
                    InternalPanel.from_xyrb(
                        self.img_size, self.small_panel_ratio, pp.x, other.y, pp.r, pp.b
                    )
                )

        if other.b > self.b:
            for pp in possible_panels.copy():
                possible_panels.append(
                    InternalPanel.from_xyrb(
                        self.img_size, self.small_panel_ratio, pp.x, pp.y, pp.r, other.b
                    )
                )

        # don't take a merged panel that bumps into other panels on page
        other_panels = [p for p in all_panels if p not in [self, other]]
        possible_panels = [p for p in possible_panels if not p.bumps_into(other_panels)]

        # take the largest merged panel
        return max(possible_panels, key=lambda p: p.area()) if len(possible_panels) > 0 else self

    def is_close(self, other: "InternalPanel") -> bool:
        """Check if two panels are close together (within 75% of combined dimensions)."""
        c1x = self.x + self.w() / 2
        c1y = self.y + self.h() / 2
        c2x = other.x + other.w() / 2
        c2y = other.y + other.h() / 2

        return all(
            [
                abs(c1x - c2x) <= (self.w() + other.w()) * 0.75,
                abs(c1y - c2y) <= (self.h() + other.h()) * 0.75,
            ]
        )

    def bumps_into(self, other_panels: List["InternalPanel"]) -> bool:
        """Check if this panel overlaps with any other panel."""
        for other in other_panels:
            if other == self:
                continue
            if self.overlaps(other):
                return True
        return False

    def contains_segment(self, segment: Segment) -> bool:
        """Check if this panel contains/overlaps with a segment."""
        other = InternalPanel.from_xyrb(self.img_size, self.small_panel_ratio, *segment.to_xyrb())
        return self.overlaps(other)

    def get_segments(self, all_segments: List[Segment]) -> List[Segment]:
        """Get all segments that are contained in this panel."""
        if self.segments is not None:
            return self.segments

        self.segments = [s for s in all_segments if self.contains_segment(s)]
        return self.segments

    def split(self, all_segments: List[Segment]) -> Optional["Split"]:
        """
        Attempt to split this panel into two sub-panels using detected segments.
        Returns Split object with two sub-panels if successful, None otherwise.
        """
        if not self.splittable:
            return None

        split_result = self._cached_split(all_segments)

        if split_result is None:
            self.splittable = False

        return split_result

    def _cached_split(self, all_segments: List[Segment]) -> Optional["Split"]:
        """Core splitting logic."""
        if self.polygon is None:
            return None

        if self.is_small(extra_ratio=2):  # panel should be splittable in two non-small subpanels
            return None

        min_hops = 3
        max_dist_x = int(self.w() / 3)
        max_dist_y = int(self.h() / 3)
        max_diagonal = math.sqrt(max_dist_x**2 + max_dist_y**2)
        dots_along_lines_dist = max_diagonal / 5
        min_dist_between_dots_x = max_dist_x / 10
        min_dist_between_dots_y = max_dist_y / 10

        # Compose modified polygon to optimize splits
        original_polygon = np.copy(self.polygon)
        polygon = np.ndarray(shape=(0, 1, 2), dtype=int, order="F")
        intermediary_dots = []

        for i in range(len(original_polygon)):
            j = (i + 1) % len(original_polygon)
            dot1 = tuple(original_polygon[i][0])
            dot2 = tuple(original_polygon[j][0])
            seg = Segment(dot1, dot2)

            # merge nearby dots together
            if seg.dist_x() < min_dist_between_dots_x and seg.dist_y() < min_dist_between_dots_y:
                original_polygon[j][0] = seg.center()
                continue

            polygon = np.append(polygon, [[dot1]], axis=0)

            # Add dots on *long* edges, by projecting other polygon dots on this segment
            add_dots = []

            # should be splittable in [dot1, dot1b(?), projected_dot3, dot2b(?), dot2]
            if seg.dist() < dots_along_lines_dist * 2:
                continue

            for k, dot3 in enumerate(original_polygon):
                if abs(k - i) < min_hops:
                    continue

                projected_dot3 = seg.projected_point(dot3)

                # Segment should be able to contain projected_dot3
                if not seg.may_contain(projected_dot3):
                    continue

                # dot3 should be close to current segment
                project = Segment(dot3[0], projected_dot3)
                if project.dist_x() > max_dist_x or project.dist_y() > max_dist_y:
                    continue

                # append dot3 as intermediary dot on segment(dot1, dot2)
                add_dots.append(projected_dot3)
                intermediary_dots.append(projected_dot3)

            # Add also a dot near each end of the segment (provoke segment matching)
            alpha_x = math.acos(seg.dist_x(keep_sign=True) / seg.dist())
            alpha_y = math.asin(seg.dist_y(keep_sign=True) / seg.dist())
            dist_x = int(math.cos(alpha_x) * dots_along_lines_dist)
            dist_y = int(math.sin(alpha_y) * dots_along_lines_dist)

            dot1b = (dot1[0] + dist_x, dot1[1] + dist_y)
            add_dots.append(dot1b)

            dot2b = (dot2[0] - dist_x, dot2[1] - dist_y)
            add_dots.append(dot2b)

            for dot in sorted(add_dots, key=lambda dot: Segment(dot1, dot).dist()):
                polygon = np.append(polygon, [[dot]], axis=0)

        # Re-merge nearby dots together
        original_polygon = np.copy(polygon)
        polygon = np.ndarray(shape=(0, 1, 2), dtype=int, order="F")

        for i in range(len(original_polygon)):
            j = (i + 1) % len(original_polygon)
            dot1 = tuple(original_polygon[i][0])
            dot2 = tuple(original_polygon[j][0])
            seg = Segment(dot1, dot2)

            # merge nearby dots together
            if seg.dist_x() < min_dist_between_dots_x and seg.dist_y() < min_dist_between_dots_y:
                intermediary_dots = [dot for dot in intermediary_dots if dot not in [dot1, dot2]]
                original_polygon[j][0] = seg.center()
                continue

            polygon = np.append(polygon, [[dot1]], axis=0)

        # Find dots nearby one another
        nearby_dots = []

        for i in range(len(polygon) - min_hops):
            for j in range(i + min_hops, len(polygon)):
                dot1 = polygon[i][0]
                dot2 = polygon[j][0]
                seg = Segment(dot1, dot2)

                if seg.dist_x() <= max_dist_x and seg.dist_y() <= max_dist_y:
                    nearby_dots.append([i, j])

        if len(nearby_dots) == 0:
            return None

        splits = []
        for dots in nearby_dots:
            poly1len = len(polygon) - dots[1] + dots[0]
            poly2len = dots[1] - dots[0]

            # A panel should have at least three edges
            if min(poly1len, poly2len) <= 2:
                continue

            # Construct two subpolygons by distributing the dots around our nearby dots
            poly1 = np.zeros(shape=(poly1len, 1, 2), dtype=int)
            poly2 = np.zeros(shape=(poly2len, 1, 2), dtype=int)

            x = y = 0
            for i in range(len(polygon)):
                if i <= dots[0] or i > dots[1]:
                    poly1[x][0] = polygon[i]
                    x += 1
                else:
                    poly2[y][0] = polygon[i]
                    y += 1

            panel1 = InternalPanel(self.img_size, self.small_panel_ratio, polygon=poly1)
            panel2 = InternalPanel(self.img_size, self.small_panel_ratio, polygon=poly2)

            if panel1.is_small() or panel2.is_small():
                continue

            if panel1 == self or panel2 == self:
                continue

            if panel1.overlaps(panel2):
                continue

            split_segment = Segment.along_polygon(polygon, dots[0], dots[1])
            split = Split(self, panel1, panel2, split_segment, all_segments)
            if split not in splits:
                splits.append(split)

        splits = [split for split in splits if split.segments_coverage() > 50 / 100]

        if len(splits) == 0:
            return None

        # return the split that best matches segments (~panel edges)
        best_split = max(splits, key=lambda split: split.covered_dist)

        return best_split


class Split:
    """Represents a panel split into two sub-panels."""

    def __init__(
        self,
        panel: InternalPanel,
        subpanel1: InternalPanel,
        subpanel2: InternalPanel,
        split_segment: Segment,
        all_segments: List[Segment],
    ) -> None:
        self.panel = panel
        self.subpanels = [subpanel1, subpanel2]
        self.segment = split_segment

        self.matching_segments = self.segment.intersect_all(panel.get_segments(all_segments))
        self.covered_dist = sum(s.dist() for s in self.matching_segments)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Split):
            return False
        return self.segment == other.segment

    def segments_coverage(self) -> float:
        """Return fraction of split segment covered by detected segments."""
        segment_dist = self.segment.dist()
        return self.covered_dist / segment_dist if segment_dist else 0
