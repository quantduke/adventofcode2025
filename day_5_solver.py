"""Day 5 AoC 2025 solver."""

import numpy as np


class Solver5():
    """Provide solution to Day 5.

    Filename of text file to be passed as argument on initialisation.
    Calling the class instance triggers a full solve.
    """

    def __init__(self, filename):
        # raw data - fresh ID ranges
        self.data_ranges: np.array = np.array([])
        # raw data - ingredient IDs
        self.data_ingredients: np.array = np.array([])

        # formatted data - fresh ID ranges
        self.fresh_id_ranges: np.array = np.array([])
        # formatted data - ingredient IDs
        self.ingredient_ids: np.array = np.array([])

        # solutions dict
        self.solutions: dict = {"Part One": 0,
                                "Part Two": 0}

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
        """Import supplied data from text file.

        Data imported as two separate numpy arrays:\
            one for fresh ID ranges, another for ingredient IDs.
        """
        # ===== IMPORT & FORMAT FRESH ID RANGES =====

        # import data
        self.data_ranges = np.loadtxt(
            fname=filename,
            dtype='O',
            # conditional format
            converters=lambda a: a if '-' in a else ''
        )

        # filter non-ranges out
        self.data_ranges = self.data_ranges[self.data_ranges != '']

        # initialise 2-column structured array
        self.fresh_id_ranges = np.zeros(
            shape=(len(self.data_ranges),),
            dtype=[('start', int), ('end', int)]
            )

        # iterate through raw data to populate array
        for index, ranges in enumerate(self.data_ranges):
            _id_range = ranges.split('-')
            self.fresh_id_ranges['start'][index] = int(_id_range[0])
            self.fresh_id_ranges['end'][index] = int(_id_range[1])

        # ===== IMPORT & FORMAT INGREDIENT IDS =====

        # import data
        self.data_ingredients = np.loadtxt(
            fname=filename,
            dtype='O',
            converters=lambda a: '' if '-' in a else int(a)  # convert to int
        )

        # filter non-IDs out
        self.ingredient_ids = self.data_ingredients[
            self.data_ingredients != ''
            ]

# =============================================================================
# ===== HELPERS =====
# =============================================================================

    def in_range(
            self,
            input_int: int,
            range_start: int,
            range_end: int
            ) -> bool:
        """For a given integer input and range, determine whether \
            integer is in range.

        Range is inclusive of start and end.
        """
        return (input_int >= range_start) & (input_int <= range_end)

# =============================================================================
# ===== CHECK FRESH INGREDIENTS =====
# =============================================================================

    def check_ingredients(self,
                          ingredient_ids: np.array,
                          fresh_id_ranges: np.array
                          ) -> int:
        """Check how many of the ingredients fall within fresh ID ranges."""
        # initialise counter for fresh IDs
        fresh_ingredients = 0

        # loop through list of ingredients...
        for ingredient in ingredient_ids:
            # ...against list of ranges...
            for index in range(len(fresh_id_ranges)):
                # ...checking for freshness in any range
                if self.in_range(
                        ingredient,
                        fresh_id_ranges['start'][index],
                        fresh_id_ranges['end'][index]
                        ):

                    # increment counter of fresh ingredients
                    fresh_ingredients += 1

                    # next ingredient
                    break

        return fresh_ingredients

# =============================================================================
# ===== GENERATE FRESH IDS =====
# =============================================================================

    def generate_fresh_ids(self, input_id_ranges: np.array) -> int:
        """For given id ranges: test, generate and sum number of \
            unique fresh IDs."""
        # ===== WORKING VARIABLES =====

        # initialise counter for unique fresh IDs
        fresh_id_counter = 0

        # note: some of the ranges are very large \
        # e.g. 500bio fresh IDs for one range.
        # to avoid generating the full array, the start/end of each range \
        # will be tested for overlap and edited accordingly
        # once complete, each unique range length will be counted.

        # duplicate array, available for editing
        fresh_id_ranges = input_id_ranges.copy()

        # ===== TEST RANGES =====

        # loop through each ID range (aka TESTED range)...
        for i in range(len(fresh_id_ranges)):
            # ...and compare it to each range in fresh ID range list
            # (aka COMPARED range)
            for j in range(len(fresh_id_ranges)):

                # ===== test 1 =====

                # tested range encounters itself, move on
                # (should only happen once per range)

                if i == j:
                    continue  # next compared range

                # ===== test 2 =====

                # tested range is entirely greater than compared range
                # with no overlap

                if fresh_id_ranges['start'][i] > fresh_id_ranges['end'][j]:
                    continue  # next compared range

                # ===== test 3 =====

                # tested range is entirely less than compared range
                # with no overlap
                if fresh_id_ranges['end'][i] < fresh_id_ranges['start'][j]:
                    continue  # next compared range

                # ===== test 4 =====

                # tested range is COMPLETELY encapsulated by compared range
                # set tested range to zero (start = end = 0)
                if (
                        fresh_id_ranges['start'][i]
                        >= fresh_id_ranges['start'][j]
                            ) & \
                    (
                        fresh_id_ranges['end'][i]
                        <= fresh_id_ranges['end'][j]
                            ):

                    # set tested range to 0
                    # to be filtered out from final fresh ID count
                    fresh_id_ranges['start'][i] = 0
                    fresh_id_ranges['end'][i] = 0

                    break  # next tested range

                # ===== test 5 =====

                # test subject COMPLETELY encapsulates compared range
                # move on
                if (
                        fresh_id_ranges['start'][i]
                        <= fresh_id_ranges['start'][j]
                        ) & \
                    (
                        fresh_id_ranges['end'][i]
                        >= fresh_id_ranges['end'][j]
                        ):
                    continue

                # (following tests are for partial overlap between tested and
                # compared ranges)

                # ===== test 6 =====

                # start of tested range overlaps over end of compared range
                if (
                        fresh_id_ranges['start'][i]
                        > fresh_id_ranges['start'][j]
                        ) & \
                    (
                        fresh_id_ranges['end'][i]
                        > fresh_id_ranges['end'][j]
                        ):

                    # truncate tested range start to one more than
                    # compared range end
                    fresh_id_ranges['start'][i] = fresh_id_ranges['end'][j] + 1

                # ===== test 7 =====

                # end of tested range overlaps into start of compared range
                if (
                        fresh_id_ranges['start'][i]
                        < fresh_id_ranges['start'][j]
                        ) & \
                    (
                        fresh_id_ranges['end'][i]
                        < fresh_id_ranges['end'][j]
                        ):

                    # truncate tested range end to one less than
                    # compared range start
                    fresh_id_ranges['end'][i] = fresh_id_ranges['start'][j] - 1

        # ===== UNIQUE FRESH ID RANGES OBTAINED ====

        # filter out overlapped ranges (already set to 0)
        fresh_id_ranges = fresh_id_ranges[
            fresh_id_ranges['start'] + fresh_id_ranges['end'] != 0
            ]

        # sum the ranges
        for i in range(len(fresh_id_ranges)):
            fresh_id_counter += \
                fresh_id_ranges['end'][i] - fresh_id_ranges['start'][i] + 1

        return int(fresh_id_counter)

# =============================================================================
# ===== PART ONE SOLVE =====
# =============================================================================

    def solve_part_one(self) -> int:
        """Solve of Part 1.

        Return number of fresh ingredients.
        """
        # sum of max joltages
        self.solutions["Part One"] = self.check_ingredients(
            self.ingredient_ids,
            self.fresh_id_ranges
            )

        # print and return
        print(f"Part One Solution:\t{self.solutions['Part One']}")

        return self.solutions["Part One"]

# =============================================================================
# ===== PART TWO SOLVE =====
# =============================================================================

    def solve_part_two(self) -> int:
        """Solve of Part 2.

        Return number of possible fresh ingredient IDs.
        """
        # sum of max joltages
        self.solutions["Part Two"] = self.generate_fresh_ids(
            self.fresh_id_ranges
            )

        # print and return
        print(f"Part Two Solution:\t{self.solutions['Part Two']}")

        return self.solutions["Part Two"]


if __name__ == '__main__':
    solver = Solver5('Input5.txt')

    # from time import perf_counter

    # start = perf_counter()
    solver.solve_part_one()
    # end = perf_counter()
    # print(f"{end-start}")

    # start = perf_counter()
    solver.solve_part_two()
    # end = perf_counter()
    # print(f"{end-start}")
