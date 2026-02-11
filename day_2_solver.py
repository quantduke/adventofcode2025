"""Day 2 AoC 2025 solver."""

import numpy as np


class Solver2():
    """Provide solution to Day 2.

    Filename of text file to be passed as argument on initialisation.
    Calling the class instance triggers a full solve.
    """

    def __init__(self, filename):
        # raw data
        self.data: np.array = np.array([])
        # formatted array of id ranges
        self.id_ranges: np.array = np.array([])
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
            delimiter=',',
            dtype='O'
            )

        # ===== FORMATTING (STRUCTURED ARRAY) =====

        # initialise array
        self.id_ranges = np.zeros(
            shape=(len(self.data),),
            dtype=[('start', int), ('end', int)]
            )

        # iterate through raw data to populate array
        for i, j in enumerate(self.data):
            _id_range = j.split('-')
            self.id_ranges['start'][i] = _id_range[0]
            self.id_ranges['end'][i] = _id_range[1]

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
            else:
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

    def invalids(self, start: int, end: int) -> list:
        """Given an integer range, produce a list of invalid IDs."""
        # ===== WORKING VARIABLES =====

        # instantiate output list of invalid IDs
        output_list = []
        # instantiate list of digit lengths to generate IDs from
        digit_lengths = []
        # instantiate working variable for ID generation
        current_id = 0

        # ===== FILTER DOMAIN =====

        # restrict domain to numbers with even number of digits
        for i in range(self.digit_count(start), self.digit_count(end)+1):
            if i % 2 == 0:
                digit_lengths.append(i)

        # ===== TRAVERSE DOMAIN =====

        # for each set of digit lengths, set min / max of the domain
        for i in digit_lengths:
            domain_start = 10**(i-1) if self.digit_count(start) != i else \
                start  # e.g. if start=3, floor at 10
            domain_end = 10**(i)-1 if self.digit_count(end) != i else \
                end  # e.g. if end=1,000,000, cap at 999,999

            # start from beginning of domain
            current_id = domain_start

            # cut the start of range into halves

            # e.g. 935177 -> 935
            half_one = self.retrieve_digits(current_id, i, i/2)
            # e.g. 935177 -> 177
            half_two = self.retrieve_digits(current_id, i/2, i/2)

            # plan of attack:
            #   increment the start half 1 (h1) and replicate to h2
            #   subject to range constraints
            while current_id <= domain_end:
                if half_one >= half_two:
                    current_id = int(half_one*(10**(i/2)) + half_one)
                    if current_id <= domain_end:
                        output_list.append(current_id)
                # increment
                half_one += 1
                half_two = half_one
                # stitch the two halves back together...
                current_id = int(half_one*(10**(i/2)) + half_one)

        return output_list

    def stitch(self, numbers: list) -> int:
        """Given a list of single integers, stitch back to one integer."""
        # output integer
        output = 0

        # error check argument
        if (len(numbers) < 1):
            return 0

        # stitch together
        for i in range(len(numbers), 0, -1):
            output += (10**(i-1)) * numbers[len(numbers) - i]
        return int(output)

    def invalids_v2(self, start: int, end: int) -> list:
        """Given an integer range, produce a list of invalid IDs.

        Any sequence repeated at least twice e.g. 998-1012 has two \
            invalid IDs, 999 and 1010.
        """
        # ===== WORKING VARIABLES =====

        # initialise output list of invalid IDs
        output_list = []
        # initialise list of digit lengths to generate IDs from
        digit_lengths = []
        # initialise working variable for ID generation
        current_id = 0
        # initialise list of equal length 'pieces' of each ID
        pieces = []

        # ===== FILTER DOMAIN =====

        # generate domain of digits lengths
        for i in range(self.digit_count(start), self.digit_count(end)+1):
            digit_lengths.append(i)

        # ===== TRAVERSE DOMAIN =====

        # for each set of digit lengths (i), set min / max of the domain
        for i in digit_lengths:
            # e.g. if start=3, floor at 10
            domain_start = 10**(i-1) if self.digit_count(start) != i \
                else start
            # e.g. if end=1,000,000, cap at 999,999
            domain_end = 10**(i)-1 if self.digit_count(end) != i \
                else end

            # start from beginning of domain
            current_id = domain_start

            # cut the ID into between [2,i] pieces (j)
            for j in range(2, i+1):
                if i % j == 0:  # if evenly cut

                    # digit size of each piece
                    piece_size = int(i/j)
                    # number of pieces... just j but helps to work in english
                    num_pieces = int(j)

                    for k in range(i):
                        # break into single digits w/ iterator k
                        # (note: piece size = i/j still here!)
                        pieces.append(
                            # e.g. 998 -> [9, 9, 8]
                            self.retrieve_digits(current_id, i - k)
                        )

                    # plan of attack:
                    #   increment the start piece (p1) and
                    #   replicate to all others (px)
                    #   subject to range constraints
                    while current_id <= domain_end:

                        # if first piece is larger than next piece
                        if self.stitch(pieces[:piece_size]) >= \
                                self.stitch(pieces[piece_size:piece_size*2]):

                            # make an invalid ID based on repetition of
                            # first piece
                            current_id = self.stitch(
                                [
                                    y for x, y in enumerate(pieces)
                                    if (x % piece_size) == x
                                ] * num_pieces
                            )
                            # add to output, given still in range
                            if (current_id >= domain_start) & \
                                    (current_id <= domain_end):
                                output_list.append(current_id)

                        # else increment first piece
                        pieces[piece_size - 1] += 1

                        # then make a new invalid ID...
                        current_id = self.stitch(
                                [
                                    y for x, y in enumerate(pieces)
                                    if (x % piece_size) == x
                                    ] * num_pieces
                            )

                        # then store updated ID in pieces again
                        pieces.clear()
                        for k in range(i):
                            pieces.append(
                                self.retrieve_digits(current_id, i - k)
                            )

                # start from beginning of domain
                current_id = domain_start
                # # re-initialise pieces for new piece size (j)
                pieces.clear()

            # re-initialise pieces for new digit length (i)
            pieces.clear()

        # return invalids
        return np.unique(output_list)

# =============================================================================
# ===== PROCESS RANGES =====
# =============================================================================

    def generate_invalid_ids(self, input_array: np.array) -> np.array:
        """For a given input array: test, generate and sum invalid IDs \
            for each range."""
        # ===== WORKING VARIABLES =====

        # Generate a boolean mask where there are potential invalid IDs.
        # Premised on even number of digits in either start or end of range.
        mask_array = np.zeros(
            shape=(len(input_array),),
            dtype=bool
        )

        for i, j in enumerate(input_array):
            mask_array[i] = (self.digit_count(j['start']) % 2 == 0) | \
                (self.digit_count(j['end']) % 2 == 0)

        # filtered array
        filtered_array = input_array[mask_array]

        # initialise output array
        output_array = np.zeros(
            shape=len(filtered_array,),
            dtype=object
        )

        # initialise counter for invalid ID sum
        counter = 0

        # array of dicts: key = tuple of range, value = list of invalid IDs
        for i, j in enumerate(filtered_array):
            output_array[i] = {
                (int(j['start']), int(j['end'])):
                    self.invalids(j['start'], j['end'])
                }

            counter += sum(self.invalids(j['start'], j['end']))

        return output_array, counter

    def generate_invalid_ids_v2(self, input_array: np.array) -> np.array:
        """For a given input array: test, generate and sum invalid IDs \
            for each range."""
        # instantiate output array
        output_array = np.zeros(
            shape=len(input_array,),
            dtype=object
        )
        # instantiate counter for invalid ID sum
        counter = 0

        # array of dicts: key = tuple of range, value = list of invalid IDs
        for i, j in enumerate(input_array):
            output_array[i] = {
                (int(j['start']), int(j['end'])):
                    self.invalids_v2(j['start'], j['end'])
                    }

            counter += sum(self.invalids_v2(j['start'], j['end']))

        return output_array, counter

# =============================================================================
# ===== PART ONE SOLVE =====
# =============================================================================

    def solve_part_one(self) -> int:
        """Solve of Part 1.

        Return sum of invalid IDs: IDs with some sequence of digits repeated \
            twice.
        """
        # Process ID ranges, yielding an array of resultant dicts containing
        # ranges with associated invalid IDs
        invalids = self.generate_invalid_ids(
            input_array=self.id_ranges
            )
        # sum of all invalid IDs
        self.solutions["Part One"] = invalids[1]

        # print and return
        print(f"Part One Solution:\t{self.solutions['Part One']}")

        return self.solutions["Part One"]

# =============================================================================
# ===== PART TWO SOLVE =====
# =============================================================================

    def solve_part_two(self, start: int = 50) -> int:
        """Solve of Part 2.

        Return sum of invalid IDs: IDs with some sequence of digits repeated \
            AT LEAST twice.
        """
        # Process ID ranges, yielding an array of resultant dicts containing
        # ranges with associated invalid IDs
        invalids = self.generate_invalid_ids_v2(
            input_array=self.id_ranges
            )
        # Derive password from total counts
        self.solutions["Part Two"] = invalids[1]

        # print and return
        print(f"Part Two Solution:\t{self.solutions['Part Two']}")

        return self.solutions["Part Two"]


if __name__ == '__main__':
    solver = Solver2('Input2.txt')

    # from time import perf_counter

    # start = perf_counter()
    solver.solve_part_one()
    # end = perf_counter()
    # print(f"{end-start}")

    # start = perf_counter()
    solver.solve_part_two()
    # end = perf_counter()
    # print(f"{end-start}")
