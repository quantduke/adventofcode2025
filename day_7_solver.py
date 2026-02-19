"""Day 7 AoC 2025 solver."""

import numpy as np


class Solver7():
    """Provide solution to Day 7.

    Filename of text file to be passed as argument on initialisation.
    Calling the class instance triggers a full solve.
    """

    def __init__(self, filename):
        # raw data
        self.data: np.array = np.array([])
        # tachyon manifold array
        self.manifold: np.array = np.array([])

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

        # import data
        self.data = np.loadtxt(
            fname=filename,
            dtype='O',
        )

        # ===== TACHYON MANIFOLD ARRAY =====

        # convert raw data to 2D integer array

        # instantiate ndarray of strings
        self.manifold = np.empty(
            shape=(
                self.data.shape[0],
                len(self.data[0])
                ),
            dtype='O'
        )

        # iterate through raw data to populate array
        for i in range(self.data.shape[0]):
            for j in range(len(self.data[0])):
                self.manifold[i, j] = self.data[i][j]

# =============================================================================
# ===== SPLIT TACHYON BEAM =====
# =============================================================================

    def split_beam(
            self,
            input_manifold: np.array,
            save_to_file: bool = False
            ) -> int:
        """Return total count of beam splits, given a tachyon manifold \
            diagram input."""
        # initialise beam split count
        split_count = 0

        # initialise index of initial beam position
        beam_init = 0

        # initialise output array for beam
        beam_array = input_manifold.copy()

        # ===== START BEAM =====

        # find index of initial tachyon beam
        for i, j in enumerate(input_manifold[0]):
            if j == 'S':
                beam_init = i

        # start beam to in beam array
        beam_array[0, beam_init] = 'S'  # source of beam into manifold
        beam_array[1, beam_init] = '|'  # initial beam

        # loop through manifold to determine path of beam
        for i in range(input_manifold.shape[0]):  # rows, i
            for j in range(input_manifold.shape[1]):  # columns, j

                # already established inital beam above
                if i <= 1:  # skip first two rows
                    break

                # check for beam in prior row of beam array
                if beam_array[i-1, j] == '|':

                    # if so, 1) check manifold for splitter
                    # 2) increment count, and 3) split beam in beam array
                    if input_manifold[i, j] == '^':
                        split_count += 1
                        beam_array[i, j-1] = '|'
                        beam_array[i, j+1] = '|'

                    # if no splitter, continue beam in beam array
                    else:
                        beam_array[i, j] = '|'

        # optional output to text
        if save_to_file:
            np.savetxt(
                "Output7-p1.txt",
                beam_array,
                fmt="%1c",
                delimiter=''
                )

        return split_count

    def count_pathways(
            self,
            input_manifold: np.array,
            save_to_file: bool = False
            ) -> int:
        """Return total count of tachyon particle pathways, given a \
            quantum tachyon manifold diagram input."""
        # initialise pathway count
        path_count = 0

        # initialise index of initial particle position
        particle_init = 0

        # initialise output array for pathways
        path_array = np.zeros_like(input_manifold)

        # each pathway to be represented with a trail of 1s.
        # if there is any overlap between pathways at any point, then this \
        # point incremented by number of overlaps
        # by the final row, we can sum the number of pathways

        # ===== STARTING PATHWAY =====

        # find index of initial tachyon particle
        for i, j in enumerate(input_manifold[0]):
            if j == 'S':
                particle_init = i

        # start beam to in beam array
        path_array[0, particle_init] = 1  # source of particle into manifold
        path_array[1, particle_init] = 1  # initial particle path

        # loop through manifold to determine path of particle
        for i in range(input_manifold.shape[0]):  # rows, i
            for j in range(input_manifold.shape[1]):  # columns, j

                # already established inital particle path above
                if i <= 1:  # skip first two rows
                    break

                # check for particle pathway in prior row of pathway array
                if path_array[i-1, j]:  # if not zero

                    # 1) check manifold for splitter
                    # 2) account for possibilities in pathway array
                    if input_manifold[i, j] == '^':
                        path_array[i, j-1] += path_array[i-1, j]
                        path_array[i, j+1] += path_array[i-1, j]

                    # if no splitter, continue beam in beam array
                    else:
                        path_array[i, j] += path_array[i-1, j]

        # final row sum == total paths
        path_count = path_array[-1, :].sum()

        # optional output to text
        if save_to_file:
            np.savetxt(
                "Output7-p2.txt",
                path_array,
                fmt="%1d",
                delimiter=''
                )

        return path_count

# =============================================================================
# ===== PART ONE SOLVE =====
# =============================================================================

    def solve_part_one(self) -> int:
        """Solve of Part 1.

        Return total count of tachyon beam splits.
        """
        # sum of max joltages
        self.solutions["Part One"] = self.split_beam(
            self.manifold
            )

        # print and return
        print(f"Part One Solution:\t{self.solutions['Part One']}")

        return self.solutions["Part One"]

# =============================================================================
# ===== PART TWO SOLVE =====
# =============================================================================

    def solve_part_two(self) -> int:
        """Solve of Part 2.

        Return total count of tachyon particle pathways.
        """
        # sum of max joltages
        self.solutions["Part Two"] = self.count_pathways(
            self.manifold
            )

        # print and return
        print(f"Part Two Solution:\t{self.solutions['Part Two']}")

        return self.solutions["Part Two"]


if __name__ == '__main__':
    solver = Solver7('Input7.txt')

    # from time import perf_counter

    # start = perf_counter()
    solver.solve_part_one()
    # end = perf_counter()
    # print(f"{end-start}")

    # start = perf_counter()
    solver.solve_part_two()
    # end = perf_counter()
    # print(f"{end-start}")
