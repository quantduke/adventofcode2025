"""Day 9 AoC 2025 solver."""

from itertools import product, combinations
import numpy as np
import matplotlib.pyplot as plt


class Solver9():
    """Provide solution to Day 9.

    Filename of text file to be passed as argument on initialisation.
    """

    def __init__(self, filename):
        # raw data
        self.data: np.array = np.array([])

        # formatted tile coordinates
        self.red_tiles: np.array = np.array([])
        self.perimeter: np.array = np.array([])

        # solutions dict
        self.solutions: dict = {"Part One": 0,
                                "Part Two": 0}

        # plot of solutions
        self.solutions_plot, self._ax = plt.subplots(
            num="Solutions Plot",
            figsize=(10., 10.)
            )

        # import data
        self.import_data(filename)

    def __call__(self):
        """Full solve."""
        self.solve_part_one()
        self.solve_part_two()

# =============================================================================
# ===== DATA INPUT AND CONVERSION =====
# =============================================================================

    def import_data(self, filename: str):
        """Import supplied data from text file."""
        self.data = np.loadtxt(
            fname=filename,
            dtype='O',
            delimiter=','
            )

        # ===== RED TILES ARRAY =====

        # convert raw data to 2D array
        # initialise array
        self.red_tiles = np.zeros(
            shape=self.data.shape
            )

        # iterate through raw data to populate array
        # itertools.product is cartesian product of two arrays
        # e.g. list(product([1,2], [3,4])) == [[1,3], [1,4], [2,3], [2,4]]

        # akin to nested for loops
        for i, j in product(*[range(x) for x in self.data.shape]):
            self.red_tiles[i, j] = int(self.data[i][j])

        # correct order of tile coordinates -> (x, y)
        self.red_tiles = self.red_tiles[:, ::-1]

        # ===== GENERATE PERIMETER TILES =====

        # concat of red tiles and green perimeter tiles
        self.perimeter = self.generate_perimeter_coords(self.red_tiles)

        # add perimeter to plot
        self.plot(
            ax=self._ax,
            data=self.perimeter,
            colour="0.5",
            label="Tile Perimeter"
                  )

# =============================================================================
# ===== HELPERS =====
# =============================================================================

    def rectangle_area(self, coord1: np.array, coord2: np.array) -> float:
        """Calculate area of one rectangle formed between two diagonal points.

        For given coordinates of two red tiles postioned at opposite corners, \
        calculate area of the rectangle formed.

        Can be used for single sets of coordinates, as well as a column array \
        of coordinates.
        """
        # calculate area
        area = np.prod(
            a=np.abs(coord2 - coord1) + 1,
            axis=-1
            )

        return area

    def rectangle_corners(self, coord1: np.array, coord2: np.array) -> float:
        """All four corner coordinates of one rectangle formed between two \
            diagonal points.

        For given coordinates of two red tiles postioned at opposite corners, \
        return all coordinates forming the corners of the rectangle.

        Output coordinates take form: \
            top-left, top-right, bottom-right, bottom-left
        """
        # ===== WORKING VARIABLES =====

        # initialise output
        corners = np.array([])

        # ===== GENERATE CORNERS OF RECTANGLE =====

        # calculate corner coordinates. cartesian product of zipped arrays.
        corners = np.array(
            list(
                product(*zip(coord1, coord2))
                )
        )

        # amend such that output coordinates are:
        # top-left, top-right, bottom-right, bottom-left
        corners[[-2, -1]] = corners[[-1, -2]]

        return corners

    def generate_perimeter_coords(
            self,
            red_tiles: np.array
            ) -> np.array:
        """Generate green tile perimeter coordinates, given red tile \
            coordinates.

        Usage can be extended to any coordinate array with a continuous \
        sequence i.e. each coordinate is linked to the previous coordinate by \
        row or column.
        """
        # ===== WORKING VARIABLES =====

        # initialise output array
        green_perim_tiles = np.array([])

        # initialise range working variables
        x_range = []
        y_range = []

        # output array
        perimeter = np.array([])

        # ===== BUILD PERIMETER =====

        # traverse each red tile coordinate to generate green tile perimeter
        # coordinates
        for index, coord in enumerate(red_tiles):

            # next red tile coordinate
            next_coord = red_tiles[
                (index+1) % len(red_tiles)
                ]

            # range of x values of green perimeter tiles
            # np.sign() to establish direction of range i.e. increasing or \
            # decreasing
            x_range = [coord[0]] if next_coord[0] == coord[0] else \
                range(
                    int(coord[0] + int(np.sign(next_coord[0] - coord[0]))),
                    int(next_coord[0]),
                    int(np.sign(next_coord[0] - coord[0]))
                    )

            # range of y values of green perimeter tiles
            y_range = [coord[1]] if next_coord[1] == coord[1] else \
                range(
                    int(coord[1] + int(np.sign(next_coord[1] - coord[1]))),
                    int(next_coord[1]),
                    int(np.sign(next_coord[1] - coord[1]))
                    )

            # generate green tile perimeter coordinates
            new_green_perim_tiles = np.array(
                list(
                    product(x_range, y_range)
                    )
                )

            # store green perimeter tile coordinates
            green_perim_tiles = np.concatenate(
                [green_perim_tiles, new_green_perim_tiles],
                axis=0
                ) if green_perim_tiles.size else new_green_perim_tiles

        # formulate concat of red/green perimeter
        perimeter = np.concatenate(
            [red_tiles, green_perim_tiles],
            axis=0
            )

        # 2-column sort and return
        return np.sort(
            perimeter.view(", ".join(
                [str(perimeter.dtype)] * perimeter.shape[1]
                )),
            axis=0,
            order=["f0", "f1"]
            ).view(perimeter.dtype)

    def corners_outside_perimeter(
            self,
            rect_bounds: np.void,
            perimeter: np.array,
            dyn_bounds: dict
            ):
        """Check if four corners of a rectangle lie outside a given perimeter.

        If so, dynamic boundaries are updated and returned to inform \
            subsequent tests.

        rect_bounds is np.void i.e. index slice of structured array
        perimeter is the whole perimeter
        dyn_bounds is the inner dict keyed by first index in tile_iter iterable
        """
        # ===== INITIALISE VARIABLES =====

        # all perimeter coordinates
        perim_coords = {
            "top": perimeter[:, 0][
                perimeter[:, 1] == rect_bounds['y_max']
                ],
            "left": perimeter[:, 1][
                perimeter[:, 0] == rect_bounds['x_min']
                ],
            "bottom": perimeter[:, 0][
                perimeter[:, 1] == rect_bounds['y_min']
                ],
            "right": perimeter[:, 1][
                perimeter[:, 0] == rect_bounds['x_max']
                ]
            }

        # check if top left corner is out of bounds
        if (
                (rect_bounds['x_min'] < perim_coords['top'][0]) &
                (rect_bounds['y_max'] > perim_coords['left'][-1])
                ):

            # top left breach -> x_min and y_max are reset into compliance
            dyn_bounds["x_min"] = perim_coords['top'][0]
            dyn_bounds["y_max"] = perim_coords['left'][-1]

            return True, dyn_bounds

        # check if top right corner is out of bounds
        if (
                (rect_bounds['x_max'] > perim_coords['top'][-1]) &
                (rect_bounds['y_max'] > perim_coords['right'][-1])
                ):

            # top right breach -> x_max and y_max are reset into compliance
            dyn_bounds["x_max"] = perim_coords['top'][-1]
            dyn_bounds["y_max"] = perim_coords['right'][-1]

            return True, dyn_bounds

        # check if bottom left corner is out of bounds
        if (
                (rect_bounds['x_min'] < perim_coords['bottom'][0]) &
                (rect_bounds['y_min'] < perim_coords['left'][0])
                ):

            # bottom left breach -> x_min and y_min are reset into
            # compliance
            dyn_bounds["x_min"] = perim_coords['bottom'][0]
            dyn_bounds["y_min"] = perim_coords['left'][0]

            return True, dyn_bounds

        # check if bottom right corner is out of bounds
        if (
                (rect_bounds['x_max'] > perim_coords['bottom'][-1]) &
                (rect_bounds['y_min'] < perim_coords['right'][0])
                ):

            # bottom right breach -> x_max and y_min are reset into
            # compliance
            dyn_bounds["x_max"] = perim_coords['bottom'][-1]
            dyn_bounds["y_min"] = perim_coords['right'][0]

            return True, dyn_bounds

        return False, dyn_bounds

    def inner_edge_perimeter_intersections(
            self,
            rect_bounds: np.void,
            perimeter: np.array,
            dyn_bounds: dict,
            coords: np.array
            ):
        """Check if four inner [non-corner] edges of a rectangle intersect \
            with a given perimeter, at any point.

        If so, dynamic boundaries are updated and returned to inform \
            subsequent tests.

        rect_bounds is np.void i.e. index slice of structured array
        perimeter is the whole perimeter
        dyn_bounds is the inner dict keyed by first index in tile_iter iterable
        """
        # ===== INITIALISE VARIABLES =====

        # all perimeter coordinates
        perim_coords = {
            "top": perimeter[:, 0][
                perimeter[:, 1] == rect_bounds['y_max']
                ],
            "left": perimeter[:, 1][
                perimeter[:, 0] == rect_bounds['x_min']
                ],
            "bottom": perimeter[:, 0][
                perimeter[:, 1] == rect_bounds['y_min']
                ],
            "right": perimeter[:, 1][
                perimeter[:, 0] == rect_bounds['x_max']
                ]
            }

        # ===== TEST TOP LINE INTERSECTIONS =====

        # for maximum y value of rectangle (top line), obtain all mutual x
        # coordinates between rectangle and perimeter (x intersections)

        # find intersection of rectangle line and perimeter, inside
        # of left/right perimeter bounds
        x_intersection_top = np.intersect1d(
            ar1=np.arange(
                rect_bounds['x_min'] + 1,
                rect_bounds['x_max']
                ),
            ar2=perim_coords['top']
            )

        # transform each inner intersection to the form:
        # (tile; perimeter; tile)
        # if the interesctions are not completely encapsulated by the perimeter
        x_inner_top = np.array(
            [
                (i-1, i, i+1) for i in x_intersection_top if
                (i-1 not in perim_coords['top']) &
                (i+1 not in perim_coords['top']) &
                (i-1 != rect_bounds['x_min']) &
                (i+1 != rect_bounds['x_max'])
                ]
            ).flatten() if not np.isin(
                element=np.arange(
                    rect_bounds['x_min'] + 1,
                    rect_bounds['x_max']
                    ),
                test_elements=perim_coords['top']
                ).all() else np.array([])

        # check if the top line intersection array is out of bounds above
        # perimeter (+y direction) or below perimeter (-y direction)
        if np.any(
                rect_bounds['y_max'] > [
                    perimeter[:, 1][perimeter[:, 0] == k].max()
                    for k in x_inner_top]
                ) | \
            np.any(
                rect_bounds['y_max'] < [
                    perimeter[:, 1][perimeter[:, 0] == k].min()
                    for k in x_inner_top]
                ):

            # if rectangle x coordinates formed in +ve x direction...
            if np.sign(coords[1, 0] - coords[0, 0]) == 1:
                # then limit maximum x value in dynamic boundaries...
                dyn_bounds["x_max"] = x_inner_top[1]
            else:
                # otherwise rectangle is formed in -ve x direction, and hence
                # minimum x value limited in dynamic boundaries
                dyn_bounds["x_min"] = x_inner_top[::-1][1]

            return True, dyn_bounds

        # ===== TEST LEFT LINE INTERSECTIONS =====

        # for minimum x value of rectangle (left line), obtain all mutual
        # y coordinates between rectangle and perimeter (y intersections)

        # find intersection of rectangle line and perimeter, inside
        # of top/bottom perimeter bounds
        y_intersection_left = np.intersect1d(
            ar1=np.arange(
                rect_bounds['y_min'] + 1,
                rect_bounds['y_max']
                ),
            ar2=perim_coords['left']
            )

        # transform each inner intersection to the form:
        # (tile; perimeter; tile)
        # if the interesctions are not completely encapsulated by the perimeter
        y_inner_left = np.array(
            [
                (i-1, i+1) for i in y_intersection_left if
                (i-1 not in perim_coords['left']) &
                (i+1 not in perim_coords['left']) &
                (i-1 != rect_bounds['y_min']) &
                (i+1 != rect_bounds['y_max'])
                ]
            ).flatten() if not np.isin(
                element=np.arange(
                    rect_bounds['y_min'] + 1,
                    rect_bounds['y_max']
                    ),
                test_elements=perim_coords['left']
                ).all() else np.array([])

        # check if the left line intersection array is out of bounds left of
        # perimeter (-x direction) at any point
        if np.any(
                rect_bounds['x_min'] < [
                    perimeter[:, 0][perimeter[:, 1] == k].min()
                    for k in y_inner_left]
                ) | \
            np.any(
                    rect_bounds['x_min'] > [
                        perimeter[:, 0][perimeter[:, 1] == k].max()
                        for k in y_inner_left]
                    ):

            # if rectangle y coordinates formed in +ve y direction...
            if np.sign(coords[1, 1] - coords[0, 1]) == 1:
                # then limit maximum y value in dynamic boundaries...
                dyn_bounds["y_max"] = y_inner_left[1]
            else:
                # otherwise rectangle is formed in -ve y direction, and hence
                # minimum y value limited in dynamic boundaries
                dyn_bounds["y_min"] = y_inner_left[::-1][1]

            return True, dyn_bounds

        # ===== TEST BOTTOM LINE INTERSECTIONS =====

        # for minimum y value of rectangle (bottom line), obtain all mutual
        # x coordinates between rectangle and perimeter (x intersections)

        # find intersection of rectangle line and perimeter, inside
        # of left/right perimeter bounds
        x_intersection_bottom = np.intersect1d(
            ar1=np.arange(
                rect_bounds['x_min'] + 1,
                rect_bounds['x_max']
                ),
            ar2=perim_coords['bottom']
            )

        # transform each inner intersection to the form:
        # (tile; perimeter; tile)
        # if the interesctions are not completely encapsulated by the perimeter
        x_inner_bottom = np.array(
            [
                (i-1, i+1) for i in x_intersection_bottom if
                (i-1 not in perim_coords['bottom']) &
                (i+1 not in perim_coords['bottom']) &
                (i-1 != rect_bounds['x_min']) &
                (i+1 != rect_bounds['x_max'])
                ]
            ).flatten() if not np.isin(
                element=np.arange(
                    rect_bounds['x_min'] + 1,
                    rect_bounds['x_max']
                    ),
                test_elements=perim_coords['bottom']
                ).all() else np.array([])

        # check if the bottom line intersection array is out of bounds above
        # perimeter (+y direction) at any point
        if np.any(
                rect_bounds['y_min'] > [
                     perimeter[:, 1][perimeter[:, 0] == k].max()
                     for k in x_inner_bottom]
                 ) | \
            np.any(
                     rect_bounds['y_min'] < [
                         perimeter[:, 1][perimeter[:, 0] == k].min()
                         for k in x_inner_bottom]
                     ):
            # if rectangle x coordinates formed in +ve x direction...
            if np.sign(coords[1, 0] - coords[0, 0]) == 1:
                # then limit maximum x value in dynamic boundaries...
                dyn_bounds["x_max"] = x_inner_bottom[1]
            else:
                # otherwise rectangle is formed in -ve x direction, and hence
                # minimum x value limited in dynamic boundaries
                dyn_bounds["x_min"] = x_inner_bottom[::-1][1]

            return True, dyn_bounds

        # ===== TEST RIGHT LINE INTERSECTIONS =====

        # for maximum x value of rectangle (right line), obtain all mutual
        # y coordinates between rectangle and perimeter (y intersections)

        # find intersection of rectangle line and perimeter, inside
        # of top/bottom perimeter bounds
        y_intersection_right = np.intersect1d(
            ar1=np.arange(
                rect_bounds['y_min'] + 1,
                rect_bounds['y_max']
                ),
            ar2=perim_coords['right']
            )

        # transform each inner intersection to the form:
        # (tile; perimeter; tile)
        # if the interesctions are not completely encapsulated by the perimeter
        y_inner_right = np.array(
            [
                (i-1, i+1) for i in y_intersection_right if
                (i-1 not in perim_coords['right']) &
                (i+1 not in perim_coords['right']) &
                (i-1 != rect_bounds['y_min']) &
                (i+1 != rect_bounds['y_max'])
                ]
            ).flatten() if not np.isin(
                element=np.arange(
                    rect_bounds['y_min'] + 1,
                    rect_bounds['y_max']
                    ),
                test_elements=perim_coords['right']
                ).all() else np.array([])

        # check if the right line intersection array is out of bounds right of
        # perimeter (+x direction) at any point
        if np.any(
                rect_bounds['x_max'] > [
                    perimeter[:, 0][perimeter[:, 1] == k].max()
                    for k in y_inner_right]
                ) | \
            np.any(
                    rect_bounds['x_max'] < [
                        perimeter[:, 0][perimeter[:, 1] == k].min()
                        for k in y_inner_right]
                    ):

            # if rectangle y coordinates formed in +ve y direction...
            if np.sign(coords[1, 1] - coords[0, 1]) == 1:
                # then limit maximum y value in dynamic boundaries...
                dyn_bounds["y_max"] = y_inner_right[1]
            else:
                # otherwise rectangle is formed in -ve y direction, and hence
                # minimum y value limited in dynamic boundaries
                dyn_bounds["y_min"] = y_inner_right[::-1][1]

            return True, dyn_bounds

        return False, dyn_bounds

    def plot(self,
             ax: plt.Axes,
             data: np.array,
             colour: str,
             label: str
             ):
        """Visualise solutions."""
        # plot on given Axes
        ax.plot(
            data[:, 0],
            data[:, 1],
            linestyle='',
            marker='.',
            markersize=0.01,
            color=colour,
            label=label
            )
        # show legend
        ax.legend(
            labelcolor="linecolor"
            )

# =============================================================================
# ===== CALCULATE ALL POSSIBLE RECTANGLE AREAS =====
# =============================================================================

    def calculate_all_rectangle_areas(self, red_tiles: np.array):
        """Calculate all rectangle areas, given an input of red tile \
            coordinates.

        Rectangles are cornered by two opposite red tiles.
        """
        # ===== INITIALISE VARIABLES =====

        # initialise tile iterable: combinations of length 2 coordinate indices
        # combinatoric iterator works for i < j, i ≠ j ==> unique rectangles
        tile_iter = list(
            combinations(range(red_tiles.shape[0]), r=2)
            )

        # initialise array arguments for area calc
        area_array_1 = np.array([red_tiles[i] for i, j in tile_iter])

        area_array_2 = np.array([red_tiles[j] for i, j in tile_iter])

        # calculate all areas, and obtain maximum area and corresponding index
        areas = self.rectangle_area(
            coord1=area_array_1,
            coord2=area_array_2
            )

        # descending sort by area
        max_area_idx = np.argsort(areas)[::-1][0]
        max_area = areas[max_area_idx]

        return tile_iter[max_area_idx], max_area

# =============================================================================
# ===== CALCULATE RECTANGLE AREAS THAT ARE SOLELY ON RED/GREEN TILES =====
# =============================================================================

    def calculate_red_green_rectangle_areas(
            self,
            red_tiles: np.array,
            perimeter: np.array
            ):
        """Given an input of tile coordinates, calculate all valid rectangle \
        areas.

        Valid rectangles are cornered by two diagonally opposite red tiles, \
        with the remaining tiles either red or green.
        """
        # ===== INITIALISE VARIABLES =====

        # initialise working max rectangle area variable
        max_area: float = 0.0
        max_area_idx: int = 0

        # initialise skip next iteration variable
        # primary tool used for reduction of rectangle iterations / tests
        skip_next: bool = False

        # initialise tile iterable, excluding tiles that share a row or column
        # i.e. not adjacent in the red tile coordinates / iterable
        tile_iter = [
            (i, j) for i, j in combinations(
                range(red_tiles.shape[0]), r=2
                ) if (j - i) % (red_tiles.shape[0] - 1) >= 2]  # not 0 or 1

        # calculate all areas
        areas = self.rectangle_area(
            coord1=np.array([red_tiles[i] for i, j in tile_iter]),
            coord2=np.array([red_tiles[j] for i, j in tile_iter])
            )

        # intialise dynamic rectangle boundaries in X & Y directions
        # additional tool used to filter
        dyn_bounds = {}

        # ===== ALL RECTANGLE BOUNDARIES =====

        # initialise structured array with min/max coordinate values for
        # each axis (X & Y)
        rect_bounds = np.empty(
            shape=len(tile_iter),
            dtype=[
                ("x_min", float),
                ("x_max", float),
                ("y_min", float),
                ("y_max", float)
                ]
            )
        # populate min/max coordinate values
        for idx, combi in enumerate(tile_iter):
            rect_bounds[idx] = (
                # minimum x value for rectangle
                np.minimum(red_tiles[combi[0]][0], red_tiles[combi[1]][0]),
                # maximum x value for rectangle
                np.maximum(red_tiles[combi[0]][0], red_tiles[combi[1]][0]),
                # minimum y value for rectangle
                np.minimum(red_tiles[combi[0]][1], red_tiles[combi[1]][1]),
                # maximum y value for rectangle
                np.maximum(red_tiles[combi[0]][1], red_tiles[combi[1]][1]),
                )

        # ===== FILL INITIAL VALUES IN DYNAMIC BOUNDARIES ====

        # for each red tile, initialise initial boundaries across X & Y axes
        # used to filter out subsequent iterations based on failure of boundary
        # tests

        dyn_bounds = {
            i: {
                "x_min": rect_bounds['x_min'].min(),
                "x_max": rect_bounds['x_max'].max(),
                "y_min": rect_bounds['y_min'].min(),
                "y_max": rect_bounds['y_max'].max()
                }
            for i in np.unique([x for x, y in tile_iter])
            }

        # ===== TRAVERSE TILE COMBINATIONS =====

        for idx, red_tile_idx in enumerate(tile_iter):

            # ===== PREREQUISITE TESTS =====

            # check if next iteration should be skipped
            if skip_next:
                skip_next = False
                continue

            # only test larger area rectangles for compliance to perimeter
            if areas[idx] <= max_area:
                continue

            # filter out rectangles with x values that exceed dynamic
            # boundaries for a given x coordinate (informed by previous
            # boundary breaches)
            if (
                    (
                        rect_bounds[idx]['x_min'] <
                        dyn_bounds[red_tile_idx[0]]["x_min"]
                        ) |
                    (
                        rect_bounds[idx]['x_max'] >
                        dyn_bounds[red_tile_idx[0]]["x_max"]
                        )
                    ):
                skip_next = True
                continue

            # filter out rectangles with y values that exceed dynamic
            # boundaries for a given y coordinate (informed by previous
            # boundary breaches)
            if (
                    (
                        rect_bounds[idx]['y_min'] <
                        dyn_bounds[red_tile_idx[0]]["y_min"]
                        ) |
                    (
                        rect_bounds[idx]['y_max'] >
                        dyn_bounds[red_tile_idx[0]]["y_max"]
                        )
                    ):
                skip_next = True
                continue

            # ===== CHECK ALL CORNERS WITHIN PERIMETER =====

            # check all four corners for adherence to perimeter boundary
            corner_check = self.corners_outside_perimeter(
                    rect_bounds=rect_bounds[idx],
                    perimeter=perimeter,
                    dyn_bounds=dyn_bounds[red_tile_idx[0]]
                    )

            if corner_check[0]:

                # skip next iteration as the infringement will persist until
                # opposite corner is translated in both x & y direction i.e.
                # at least two iterations away
                skip_next = True

                # corner breaches are passed onto dynamic boundaries, ensuring
                # compliance for subsequent iterations
                dyn_bounds[red_tile_idx[0]] = corner_check[1]

                continue

            # ===== CHECK INNER-LINE / PERIMETER CROSSOVERS =====

            # check for inner (non-corner) rectangle edge adherence to
            # perimeter boundaries

            inner_line_check = self.inner_edge_perimeter_intersections(
                rect_bounds=rect_bounds[idx],
                perimeter=perimeter,
                dyn_bounds=dyn_bounds[red_tile_idx[0]],
                coords=red_tiles[list(red_tile_idx)]
                    )

            if inner_line_check[0]:

                # skip next iteration as the infringement will persist until
                # opposite corner is translated in both x & y direction i.e.
                # at least two iterations away
                skip_next = True

                # inner_line breaches are passed onto dynamic boundaries,
                # ensuring compliance for subsequent iterations
                dyn_bounds[red_tile_idx[0]] = inner_line_check[1]

                continue

            # ===== COMPLIANT RECTANGLE =====

            # log max area and associated red tile index
            max_area = areas[idx]
            max_area_idx = red_tile_idx

        return max_area_idx, max_area

# =============================================================================
# ===== PART ONE SOLVE =====
# =============================================================================

    def solve_part_one(self) -> float:
        """Solve of Part 1.

        Return largest possible rectangle area.
        """
        # Calculate all rectangle areas
        area = self.calculate_all_rectangle_areas(
            red_tiles=self.red_tiles,
            )
        # Max rectangle area
        self.solutions["Part One"] = area[1]

        # generate solution rectangle
        solution_rectangle = self.generate_perimeter_coords(
            self.rectangle_corners(
                *self.red_tiles[list(area[0])]
                )
            )

        # add to plot
        self.plot(
            ax=self._ax,
            data=solution_rectangle,
            colour='r',
            label=f"Part 1 Solution: {int(area[1]):,}"
            )

        # print and return
        print(f"Part One Solution:\t{self.solutions['Part One']}")

        return self.solutions["Part One"]

# =============================================================================
# ===== PART TWO SOLVE =====
# =============================================================================

    def solve_part_two(self) -> int:
        """Solve of Part 2.

        Return largest possible rectangle area, exclusively on red/green tiles.
        """
        # Calculate all "red/green" rectangle areas
        area = self.calculate_red_green_rectangle_areas(
            self.red_tiles,
            self.perimeter
            )
        # Max rectangle area
        self.solutions["Part Two"] = area[1]

        # generate solution rectangle
        solution_rectangle = self.generate_perimeter_coords(
            self.rectangle_corners(
                *self.red_tiles[list(area[0])]
                )
            )

        # add to plot
        self.plot(
            ax=self._ax,
            data=solution_rectangle,
            colour='g',
            label=f"Part 2 Solution: {int(area[1]):,}"
            )

        # print and return
        print(f"Part Two Solution:\t{self.solutions['Part Two']}")

        return self.solutions["Part Two"]


if __name__ == '__main__':
    solver = Solver9('Input9.txt')

    # from time import perf_counter

    # start = perf_counter()
    solver.solve_part_one()
    # end = perf_counter()
    # print(f"{end-start}")

    # start = perf_counter()
    solver.solve_part_two()
    # end = perf_counter()
    # print(f"{end-start}")
