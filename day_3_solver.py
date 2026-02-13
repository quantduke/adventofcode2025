"""Day 3 AoC 2025 solver."""

import numpy as np


class Solver3():
    """Provide solution to Day 3.

    Filename of text file to be passed as argument on initialisation.
    Calling the class instance triggers a full solve.
    """

    def __init__(self, filename):
        # raw data
        self.data: np.array = np.array([])
        # array of battery banks
        self.bank_matrix: np.array = np.array([])
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
        self.bank_matrix = np.zeros(
            shape=(
                len(self.data),
                # each row is same length
                self.digit_count(int(self.data[0]))
                ),
            dtype=int
            )

        # iterate through raw data to populate array
        for i, j in enumerate(self.data):
            # k is iterator over bank length
            for k in range(self.digit_count(int(j))):
                self.bank_matrix[i, k] = self.retrieve_digits(
                    int(j),
                    self.digit_count(int(j)) - k
                    )

# =============================================================================
# ===== HELPERS =====
# =============================================================================

    def digit_count(self, input_int: int) -> int:
        """For a given integer input, return number of digits."""
        # initialise counter
        digit_counter = 0

        # increment digit counter using modulo function
        while True:
            if (input_int % (10**digit_counter)) == input_int:
                return digit_counter
            digit_counter += 1

    def retrieve_digits(self,
                        number: int,
                        position: int,
                        count: int = 1
                        ) -> int:
        """Return digit(s) of integer from given position.

        1-indexed from units upwards.
        count is number of digits to return.
        """
        # bound the position argument
        if (position > self.digit_count(number)) | (position < 1):
            return 0

        # modulo function logic for retrieval of digits
        return int(
            (
                (number % 10**position) - (number % 10**(position - count))
                ) / 10**(position - count)
        )

    def new_argmax(self, input_array: np.array, *max_value: int) -> list:
        """Return every index of max values in an array.

        Works for 1d arrays.
        Returns tuple: (max value, array of indices of occurence)
        """
        # check for null array
        if not any(input_array):
            return (0, np.array([]))

        # max value, if supplied or found in the array
        val = max_value if any(max_value) else (max(input_array),)

        return val[0], np.where(input_array == val[0])[0]

    def stitch(self, numbers: list) -> int:
        """Given a list of single integers, stitch back to one integer.

        Be aware of lists that are too long; \
            limited by the length of long type in C.
        """
        # output integer
        output = 0

        # error check argument
        if len(numbers) < 1:
            return 0

        # stitch together
        for i in range(len(numbers), 0, -1):
            output += (10**(i-1)) * numbers[len(numbers) - i]
        return int(output)

# =============================================================================
# ===== MAX JOLTAGE =====
# =============================================================================

    def max_joltage_2_batteries(self, input_array: np.array) -> int:
        """Ouput max joltage for a given input array.

        Logic is for two batteries.
        """
        # select 2 batteries
        batteries = 2

        # check for null array
        if not any(input_array):
            return 0

        # ===== WORKING VARIABLES =====

        # initialise array (before stitching) of max joltage
        output_joltage = np.zeros(
            shape=(batteries,),
            dtype=int
        )

        # indices of initial max value occurences
        indices = self.new_argmax(input_array)[1]

        # working variable for list of array splits
        splits = []

        # working variable for remaining batteries to select
        remaining_batteries = max(0, batteries - len(indices))

        # ===== AT LEAST 2 MAX VALUES =====

        # if there are more max value occurences than batteries needed...
        if not remaining_batteries:
            # ...then populate digits for joltage
            for i in range(batteries):
                output_joltage[i] = input_array[indices[i]]

            return self.stitch(output_joltage)

        # ===== ONE MAX VALUE =====

        # split array to see position context of batteries
        splits = np.split(input_array, indices)

        # edge case 1: the max value the last in the array
        if len(splits[-1]) == 1:
            for i in range(batteries):
                # populate digits for joltage
                output_joltage[i] = self.new_argmax(splits[i])[0]

        # edge case 2: the max value the first in the array
        elif len(splits[0]) == 0:

            for i in range(batteries):
                # populate digits of joltage
                output_joltage[i] = splits[1][i] if i == 0 else \
                    self.new_argmax(splits[i][1:])[0]

        # all other positions of max value
        else:

            for i in range(batteries):
                # populate digits of joltage
                output_joltage[i] = splits[1][i] if i == 0 else \
                    self.new_argmax(splits[i][1:])[0]

        return self.stitch(output_joltage)

    def max_joltage_n_batteries(self,
                                input_array: np.array,
                                batteries: int = 12
                                ) -> int:
        """Ouput max joltage for a given input array.

        Default is for twelve batteries, but can be up to number of batteries \
            in the bank.
        """
        # error checker
        if batteries >= len(input_array):
            return self.stitch(input_array)

        # ===== WORKING VARIABLES =====

        # initialise array (before stitching) of max joltage
        output_joltage = np.zeros(
            shape=(batteries,),
            dtype=int
        )

        # initial max value
        max_value_init = self.new_argmax(input_array)[0]

        # indices of initial max value occurences
        indices = self.new_argmax(input_array)[1]

        # initial list of array splits
        splits_init = []

        # working variable for remaining batteries to calculate
        # (apart from max values)
        remaining_batteries = max(0, batteries - len(indices))

        # counter for batteries added
        batteries_added = 0

        # working variable for a 'sub-joltage' in recursion of sub-splits
        sub_joltage = 0

        # ===== MORE MAX VALUES THAN BATTERIES =====

        # if there are more max value occurences than batteries needed...
        if not remaining_batteries:
            # ...then populate digits for joltage
            output_joltage = \
                [input_array[indices[i]] for i in range(batteries)]

            return self.stitch(output_joltage)

        # ===== LESS MAX VALUES THAN BATTERIES =====

        # split array to see position context of batteries
        # splits by each max value, which is placed at start of sub-array.
        # there are always len(indices) + 1 splits, even if a split is empty.
        splits_init = np.split(input_array, indices)

        # aside from max values, the remaining batteries need to be calculated
        # for the maximum joltage.
        # PLAN OF ATTACK: iterate from right-most split, filling up calculated
        # digits until remaining_batteries == 0
        # and also fill the max values

        # i is from 1 to len(splits_init)
        for i in range(1, len(splits_init) + 1):

            # skip over 0 length splits -> go to next split
            if not splits_init[-i].all():
                continue

            # ===== case 1 =====
            # no more remaining batteries to calculate.
            # populate with the initial max value (first in split)

            if not remaining_batteries:

                # terminate insertion at end of output array
                if (
                        (batteries_added + 1) <= len(output_joltage)
                        ) & \
                    (
                        i <= len(indices)
                        ):
                    # add split max value to output
                    output_joltage[-(batteries_added+1)] = \
                        input_array[indices[-i]]

                # update added battery after update
                batteries_added += 1

                continue

            # ===== case 2 =====
            # need to use all of split digits (excluding max value) for joltage

            # start from rightmost split
            if len(splits_init[-i]) <= remaining_batteries:

                # decrement remaining batteries to be calculated
                remaining_batteries -= len(splits_init[-i][1:])

                # populate output with whole split from rhs
                # j is from 1 to len(sub-split)
                for j in range(1, len(splits_init[-i]) + 1):

                    # terminate insertion at end of output array
                    if (j + batteries_added) <= len(output_joltage):
                        output_joltage[-(j+batteries_added)] = \
                            splits_init[-i][-j]

                # update added batteries after full update
                batteries_added += len(splits_init[-i])

                continue

            # ===== case 3 =====
            # otherwise, more split digits than batteries outstanding.
            # Recursively find maximal sub-joltage for split

            # if first element of this split is a max value
            # (i.e. will be included in joltage),
            # then find max joltage of other elements,
            # otherwise find max joltage of entire split

            sub_joltage = self.max_joltage_n_batteries(
                splits_init[-i][1:],
                remaining_batteries
                ) if \
                (max_value_init in splits_init[-i]) else \
                self.max_joltage_n_batteries(
                    splits_init[-i],
                    remaining_batteries
                    )

            # add split sub-joltage digits to output
            # k is from 1 to remaining batteries

            output_joltage[
                -batteries_added-1:
                    -batteries_added-remaining_batteries-1:
                        -1
                    ] = \
                [
                    self.retrieve_digits(sub_joltage, k)
                    for k in range(1, remaining_batteries+1)
                    ]

            # update added batteries after full update of sub-joltage
            batteries_added += self.digit_count(sub_joltage)

            # add split max value to output (if applicable)
            if i <= len(indices):
                if splits_init[-i][0] == input_array[indices[-i]]:
                    # terminate insertion at end of output array
                    if (batteries_added + 1) <= len(output_joltage):
                        output_joltage[-(batteries_added+1)] = \
                            splits_init[-i][0]

                    # update added batteries after update of max value
                    batteries_added += 1

            # no more batteries to be chosen
            remaining_batteries = 0

        return self.stitch(output_joltage)

# =============================================================================
# ===== PART ONE SOLVE =====
# =============================================================================

    def solve_part_one(self) -> int:
        """Solve of Part 1.

        Return sum of total max joltage (2 batteries) across battery bank.
        """
        # sum of max joltages
        self.solutions["Part One"] = sum(
            (self.max_joltage_2_batteries(i) for i in self.bank_matrix)
            )

        # print and return
        print(f"Part One Solution:\t{self.solutions['Part One']}")

        return self.solutions["Part One"]

# =============================================================================
# ===== PART TWO SOLVE =====
# =============================================================================

    def solve_part_two(self, n: int = 12) -> int:
        """Solve of Part 2.

        Return sum of total max joltage (n batteries) across battery bank.
        """
        # sum of max joltages
        self.solutions["Part Two"] = sum(
            (self.max_joltage_n_batteries(i, n) for i in self.bank_matrix)
            )

        # print and return
        print(f"Part Two Solution:\t{self.solutions['Part Two']}")

        return self.solutions["Part Two"]


if __name__ == '__main__':
    solver = Solver3('Input3.txt')

    # from time import perf_counter

    # start = perf_counter()
    solver.solve_part_one()
    # end = perf_counter()
    # print(f"{end-start}")

    # start = perf_counter()
    solver.solve_part_two()
    # end = perf_counter()
    # print(f"{end-start}")
