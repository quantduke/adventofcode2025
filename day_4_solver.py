"""Day 4 AoC 2025 solver."""

import numpy as np


class Solver4():
    """Provide solution to Day 4.

    Filename of text file to be passed as argument on initialisation.
    Calling the class instance triggers a full solve.
    """

    def __init__(self, filename):
        # raw data
        self.data: np.array = np.array([])
        # array of battery banks
        self.roll_matrix: np.array = np.array([])
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
        """Import supplied data from text file."""
        # ===== IMPORT DATA =====

        self.data = np.loadtxt(
            fname=filename,
            dtype='O'
            )

        # ===== FORMATTING =====

        # initialise array
        self.roll_matrix = np.zeros(
            shape=(
                len(self.data),
                # each row is same length
                len(self.data[0])
                ),
            dtype=int
            )

        # iterate through raw data to populate array
        for i, j in enumerate(self.data):
            # k is iterator over each row
            for k in enumerate(j):
                # represent each paper roll with 1, else 0
                self.roll_matrix[i, k[0]] = 1 if k[1] == '@' else 0

# =============================================================================
# ===== HELPERS =====
# =============================================================================

    def eight_adjacent(self,
                       row: int,
                       column: int,
                       input_array: np.array
                       ) -> np.array:
        """For a given input array address for a supplied array, return \
            sub-array with element surrounded by eight adjacent elements.

        Relies on good input of row/column indices.
        """
        # rows and columns floored at 0 index, capped at array len
        start_row = min(max(0, row - 1), input_array.shape[0])
        end_row = min(max(0, row + 2), input_array.shape[0])

        start_column = min(max(0, column - 1), input_array.shape[1])
        end_column = min(max(0, column + 2), input_array.shape[1])

        return input_array[start_row:end_row, start_column:end_column]

# =============================================================================
# ===== ACCESSIBLE / REMOVABLE ROLLS =====
# =============================================================================

    def accessible_paper_rolls(self,
                               input_array: np.array,
                               max_adjacent_rolls: int = 3
                               ) -> int:
        """Given a 2d array of rolls (1) and spaces (0), return number \
            accessible paper rolls."""
        # initialise counter for accessible rolls
        accessible_rolls_counter = 0

        # loop through array elements
        for i in range(input_array.shape[0]):  # i is row index
            for j in range(input_array.shape[1]):  # j is column index

                # if element is not a roll -> next element
                if not input_array[i, j]:
                    continue

                # check if roll is accessible
                if (
                        self.eight_adjacent(i, j, input_array).sum() - 1
                        <= max_adjacent_rolls
                        ):
                    accessible_rolls_counter += 1

        return accessible_rolls_counter

    def removable_paper_rolls(self,
                              rolls_array: np.array,
                              max_adjacent_rolls: int = 3
                              ) -> int:
        """Given a 2d array of rolls (1) and spaces (0), return number of \
            paper rolls that can be removed until no accessible rolls are \
                left."""
        # initialise counter for removed rolls
        removed_rolls_counter = 0

        # initialise loop condition
        accesible = 0

        # duplicate input array for editing
        input_array = rolls_array.copy()

        while True:
            # assume no more accessible paper rolls at the start of each loop
            accesible = 0

            # loop through all array elements
            for i in range(input_array.shape[0]):  # i is row index
                for j in range(input_array.shape[1]):  # j is column index

                    # if element is not a roll -> next element
                    if not input_array[i, j]:
                        continue

                    # check if roll is accessible
                    if (
                            self.eight_adjacent(
                                i, j, input_array
                                ).sum() - 1
                            <= max_adjacent_rolls
                            ):
                        # if so, remove it
                        input_array[i, j] = 0

                        # increment counter for removed rolls
                        removed_rolls_counter += 1

                        # signal that there are accessible paper rolls
                        accesible = 1

            # if no accessible paper rolls have been found, break the loop
            if not accesible:
                break

        return removed_rolls_counter

# =============================================================================
# ===== PART ONE SOLVE =====
# =============================================================================

    def solve_part_one(self) -> int:
        """Solve of Part 1.

        Return number of accessible paper rolls.
        """
        # sum of max joltages
        self.solutions["Part One"] = self.accessible_paper_rolls(
            self.roll_matrix
            )

        # print and return
        print(f"Part One Solution:\t{self.solutions['Part One']}")

        return self.solutions["Part One"]

# =============================================================================
# ===== PART TWO SOLVE =====
# =============================================================================

    def solve_part_two(self) -> int:
        """Solve of Part 2.

        Return number of removable paper rolls.
        """
        # sum of max joltages
        self.solutions["Part Two"] = self.removable_paper_rolls(
            self.roll_matrix
            )

        # print and return
        print(f"Part Two Solution:\t{self.solutions['Part Two']}")

        return self.solutions["Part Two"]


if __name__ == '__main__':
    solver = Solver4('Input4.txt')

    # from time import perf_counter

    # start = perf_counter()
    solver.solve_part_one()
    # end = perf_counter()
    # print(f"{end-start}")

    # start = perf_counter()
    solver.solve_part_two()
    # end = perf_counter()
    # print(f"{end-start}")
