"""Day 1 AoC 2025 solver."""

import numpy as np


class Solver1():
    """Provide solution to Day 1.

    Filename of text file to be passed as argument on initialisation.
    Calling the class instance triggers a full solve.
    """

    def __init__(self, filename):
        # raw data
        self.data: np.array = np.array([])
        # formatted rotations data
        self.rotations_array: np.array = np.array([])
        # solutions dict
        self.solutions: dict = {"Part One": 0,
                                "Part Two": 0}

        # import data
        self.import_data(filename)
        # format data
        self.rotations_array = self.convert_input(self.data)

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
            dtype='O'
            )

# =============================================================================
# ===== HELPERS =====
# =============================================================================

    def convert_input(self, input_array: np.array) -> np.array:
        """Convert puzzle input into an array of safe rotations.."""
        # instantiate output array
        output_array = np.zeros_like(
            a=input_array,
            dtype=int
            )

        # test input array for correct prefix
        direction_test = [(x[0].startswith('L') | x[0].startswith('R'))
                          for x in input_array]
        assert False not in direction_test, \
            "one or more input rotations do not start with L or R"

        # convert input into integer rotations
        # negative for left rotations, positive for right
        for i, j in enumerate(input_array):
            if j.startswith('L'):
                rotation = j.split('L')  # split into list of ['L', '123']
                # add integer rotation to output
                output_array[i] = -int(rotation[1])  # negative integer
            else:
                rotation = j.split('R')
                output_array[i] = int(rotation[1])  # positive integer

        # array of signed ints
        return output_array

# =============================================================================
# ===== PROCESS ROTATIONS =====
# =============================================================================

    def process_rotations(self,
                          input_array: np.array,
                          start: int
                          ) -> np.array:
        """From a given starting point, process the given rotations.

        Output is an array of resultant dial points for each rotation.
        """
        # ===== WORKING VARIABLES =====

        # initialise array
        output_array = np.zeros_like(
            a=input_array,
            dtype=int
            )

        # cumulative sum of rotations
        cum_sum = 0

        # ===== TRAVERSE ROTATIONS ARRAY =====

        for i, j in enumerate(input_array):
            # populate output with start and cum. sum. of rotations
            # at each step. Modulo 100 to keep within the circular
            # range of dial points
            cum_sum += j
            output_array[i] = (start + cum_sum) % 100

        return output_array

    def process_rotations_v2(self,
                             input_array: np.array,
                             start: int
                             ) -> tuple:
        """From a given starting point, process the given rotations.

        Function calculates rotations that end on dial point 0, as well as \
            rotations that pass through dial point 0.
        """
        # ===== WORKING VARIABLES =====

        # initialise current dial point
        current_dial_point = start  # given argument

        # initialise counter for 0 ending, and for 0 passing
        zero_end_count = 0
        zero_pass_count = 0

        # initialise counter for full rotations
        full_rotations = 0

        # ===== TRAVERSE ROTATIONS ARRAY =====

        for rotation in input_array:

            # calculate and record new dial point after rotation.
            # modulo 100 to keep within the circular range of dial points
            new_dial_point = (current_dial_point + rotation) % 100

            # for zero-ending rotations, increment counter
            zero_end_count += (new_dial_point == 0)

            # ===== ZERO PASS LOGIC =====

            full_rotations = abs(
                (
                    # start point + rotation raw amount...
                    (current_dial_point + rotation) \
                    # ... minus ending dial point = number of rotations * 100
                    - new_dial_point
                ) / 100  # divide by 100 to get number of rotations
            )

            # ===== CORRECTIONS =====

            # correct rotation over-count for dial point zero for both
            # start & end
            full_rotations -= (current_dial_point == new_dial_point == 0)

            # correct positive rotation over-count for zero ends
            if new_dial_point == 0:
                full_rotations -= 1 if np.sign(rotation) == 1 else 0

            # correct negative rotation over-count for zero starts
            if current_dial_point == 0:
                full_rotations -= 1 if np.sign(rotation) == -1 else 0

            # ===== CONTINUE =====

            # increment counter for zero-passing rotations
            zero_pass_count += full_rotations

            # update dial point for next rotation
            current_dial_point = new_dial_point

        # return tuple of counts of zero-ending and zero-passing rotations
        return int(zero_end_count), int(zero_pass_count)

# =============================================================================
# ===== PART ONE SOLVE =====
# =============================================================================

    def solve_part_one(self, start: int = 50) -> int:
        """Solve of Part 1.

        Return number of times the dial is left pointing at 0 after any \
            rotation in the sequence.
        """
        # Process rotations, yielding an array of resultant dial points
        points_array = self.process_rotations(
            input_array=self.rotations_array,
            start=start
            )
        # Derive product of top three circuit lengths
        self.solutions["Part One"] = len(
            points_array[points_array == 0]
            )
        # print and return
        print(f"Part One Solution:\t{self.solutions['Part One']}")

        return self.solutions["Part One"]

# =============================================================================
# ===== PART TWO SOLVE =====
# =============================================================================

    def solve_part_two(self, start: int = 50) -> int:
        """Solve of Part 2.

        Return number of times the dial is left pointing at 0 or passes 0 \
            after any rotation in the sequence.
        """
        # Derive password from total counts
        self.solutions["Part Two"] = int(
            sum(
                self.process_rotations_v2(
                    input_array=self.rotations_array,
                    start=start
                )
                )
            )
        # print and return
        print(f"Part Two Solution:\t{self.solutions['Part Two']}")

        return self.solutions["Part Two"]


if __name__ == '__main__':
    solver = Solver1('Input1.txt')

    # from time import perf_counter

    # start = perf_counter()
    solver.solve_part_one()
    # end = perf_counter()
    # print(f"{end-start}")

    # start = perf_counter()
    solver.solve_part_two()
    # end = perf_counter()
    # print(f"{end-start}")
