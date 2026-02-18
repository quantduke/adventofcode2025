"""Day 6 AoC 2025 solver."""

import numpy as np


class Solver6():
    """Provide solution to Day 6.

    Filename of text file to be passed as argument on initialisation.
    Calling the class instance triggers a full solve.
    """

    def __init__(self, filename):
        # raw data
        self.data: np.array = np.array([])

        # formatted data - operands
        self.operands: np.array = np.array([])
        # formatted data - operands in string form
        self.operands_str: np.array = np.array([])
        # formatted data - operators
        self.operators: np.array = np.array([])

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
        # ===== IMPORT & FORMAT DATA =====

        # import data

        # forced import of whitespace by setting delimiter to arbitrary \
        # symbol '@' that is not in dataset
        self.data = np.loadtxt(
            fname=filename,
            dtype='O',
            delimiter='@'
        )

        # split data by whitespace column
        self.data = self.split_array_by_common_char(
            input_array=self.data,
            char=' '
            )

        # ===== OPERANDS ARRAY =====

        # convert raw data to 2D integer array

        # initialise array of ints
        self.operands = np.zeros(
            shape=(
                self.data.shape[0] - 1,
                len(self.data[0])
                ),
            dtype=int
        )

        # initialise array of string
        self.operands_str = np.zeros(
            shape=(
                self.data.shape[0] - 1,
                len(self.data[0])
                ),
            dtype='O'
        )

        # iterate through raw data to populate both int and str arrays
        for i in range(self.data.shape[0] - 1):
            for j in range(len(self.data[0])):
                # integer
                self.operands[i, j] = int(self.data[i][j])
                # string
                self.operands_str[i, j] = self.data[i][j]

        # ===== OPERATORS ARRAY =====

        # convert raw data to 1D string array

        # initialise ndarray
        self.operators = np.zeros(
            shape=(
                len(self.data[0]),
                ),
            dtype=str
        )

        # iterate through raw data to populate array
        for i in range(len(self.data[0])):
            self.operators[i] = self.data[-1][i].strip()

# =============================================================================
# ===== HELPERS =====
# =============================================================================

    def split_string(
            self,
            string: str,
            indices: list,
            remove_index_char: bool = True
            ) -> list:
        """Split a string at the given positions, with option to remove \
        the character at the split point."""
        # initialise output array of split strings
        output_array = []

        # initialise starting index
        start_index = 0

        # traverse indices of splits
        for current_index in indices:
            # append each split to output
            output_array.append(
                string[start_index:current_index]
                )

            # update indices, with option to remove the character \
            # at the split point
            start_index = current_index + remove_index_char

        # finally, add last split
        output_array.append(
            string[start_index:]
            )

        return output_array

    def split_array_by_common_char(
            self,
            input_array: np.array,
            char: str = ' '
            ) -> np.array:
        """For a given input array, return output array split by blank \
            columns, if any.

        "Blank" means 1 whitespace character, and "blank column" means 1 \
            whitespace character in the same position across all rows of data.

        Input is a 1d array of row data that would yield a 2d array if rows \
            were vertically stacked.
        """
        # initialise array of indices
        indices = []

        # initialise output array of split strings
        output_array = np.empty_like(input_array)

        # number of rows
        rows = len(input_array)

        # number of columns
        cols = len(input_array[0])

        # traverse each char of each element, 'columns'
        for i in range(cols):
            # check if all element are whitespace
            if (np.array([j[i] for j in input_array]) == char).all():
                # if so, make note of index
                indices.append(i)

        # once list of indices completed, split the strings by index
        for k in range(rows):
            output_array[k] = self.split_string(
                input_array[k],
                indices,
                True
                )

        return output_array

    def calculate_problem(
            self,
            input_operands: np.array,
            operator: str
            ) -> int:
        """For a given operand array and operator, evaluate and return \
            solution."""
        # multiplication
        if operator == '*':
            return np.prod(input_operands)

        # addition
        if operator == '+':
            return np.sum(input_operands)

        return None

    def stitch_strings_to_int(self, numbers: list) -> int:
        """Given a list of integer strings, stitch back to one integer.

        E.g. ['4 ', ' 2', '  1'] -> 421
        """
        # output integer
        output = ''

        # error check argument
        if len(numbers) < 1:
            return 0

        # stitch strings together
        for i in numbers:
            output += i

        # return integer with whitespace stripped
        return int(output.strip())

# =============================================================================
# ===== PROBLEM WORKSHEET =====
# =============================================================================

    def problem_worksheet_total(
            self,
            operands: np.array,
            operators: np.array
            ) -> int:
        """Return total of worksheet problems, given operand and operator \
            inputs."""
        # initialise output
        grand_total = 0

        # loop through list of operators...
        for i, j in enumerate(operators):
            grand_total += self.calculate_problem(
                operands[:, i],
                j
                )  # add problem solution to grand total

        return grand_total

    def problem_worksheet_total_new(
            self,
            input_operands: np.array,
            input_operators: np.array
            ) -> np.int64:
        """Return total of worksheet problems, given operand and operator \
            inputs.

        Operand array is 2d array of strings of integers/whitespace.

        New method of parsing numbers (right-to-left, column-wise) in use.
        """
        # initialise output
        grand_total = np.int64(0)

        # working variable for columns width
        col_width = 0

        # working list for parsed numbers
        parsed_ints = []

        # using the list of operators as the counter ...
        for counter, operator in enumerate(input_operators):

            # number of chars in column
            col_width = len(input_operands[:, counter][0])

            # ... loop through each char position in the column
            # parsing the operands with the new column-wise method
            for char in range(1, col_width + 1):
                # add list of integer strings, read left-to-right
                parsed_ints.append(
                    self.stitch_strings_to_int(
                        [
                            string[-char] for string in
                            input_operands[:, counter]
                            ]
                    )
                )

            grand_total += self.calculate_problem(
                parsed_ints,
                operator
                )  # add problem solution to grand total

            # reset parsed ints
            parsed_ints.clear()

        return grand_total

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

        Return sum total of worksheet problems.
        """
        # sum of max joltages
        self.solutions["Part One"] = self.problem_worksheet_total(
            self.operands,
            self.operators
            )

        # print and return
        print(f"Part One Solution:\t{self.solutions['Part One']}")

        return self.solutions["Part One"]

# =============================================================================
# ===== PART TWO SOLVE =====
# =============================================================================

    def solve_part_two(self) -> int:
        """Solve of Part 2.

        Return sum total of worksheet problems, read right-to-left, \
            column-wise.
        """
        # sum of max joltages
        self.solutions["Part Two"] = self.problem_worksheet_total_new(
            self.operands_str,
            self.operators
            )

        # print and return
        print(f"Part Two Solution:\t{self.solutions['Part Two']}")

        return self.solutions["Part Two"]


if __name__ == '__main__':
    solver = Solver6('Input6.txt')

    # from time import perf_counter

    # start = perf_counter()
    solver.solve_part_one()
    # end = perf_counter()
    # print(f"{end-start}")

    # start = perf_counter()
    solver.solve_part_two()
    # end = perf_counter()
    # print(f"{end-start}")
