"""Day 8 AoC 2025 solver."""

import numpy as np


class Solver8():
    """Provide solution to Day 8.

    Filename of text file to be passed as argument on initialisation.
    """

    def __init__(self, filename):
        # raw data
        self.data: np.array = np.array([])
        # formatted junction box data
        self.jbox: np.array = np.array([])
        # matrix of euclidean distances
        self.distance_matrix: np.array = np.array([])
        # circuits: list of sets of junction box indices
        self.circuits: list = []
        # connections between jboxes
        self.connections: np.array = np.array([])
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
        self.data = np.loadtxt(
            fname=filename,
            dtype='O',
            delimiter=','
            )

        # ===== JUNCTION BOX ARRAY =====

        # convert raw data to 2D integer ndarray
        # initialise ndarray of strings
        self.jbox = np.zeros(
            shape=self.data.shape
            )

        # iterate through raw data to populate array
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                self.jbox[i, j] = int(self.data[i][j])

        # ===== POPULATE DISTANCE MATRIX =====

        # initialise euclidean distance matrix
        self.distance_matrix = np.zeros(
            (len(self.jbox), len(self.jbox))
        )

        # opting for upper diagonal matrix, so that convention can be:
        # distance between matrix 1 + matrix 2 is [r, c]

        for j in enumerate(self.jbox):  # columns
            for i in range(j[0]):  # rows, only upper diag
                # calculate the euclidean distance and populate
                self.distance_matrix[i, j[0]] = self.eucl_nd(
                    self.jbox[i],
                    self.jbox[j[0]]
                    )

# =============================================================================
# ===== HELPERS =====
# =============================================================================

    def eucl_nd(self, coord1: np.array, coord2: np.array) -> float:
        """For given n-dimension cartesian coordinates of two entities, \
            calculate Euclidean distance.

        Euclidean distance is square root of the sum of squared differences.
        """
        # error check
        assert len(coord1) == len(coord2), \
            "Coordinates not of equal dimensionality."

        # initialise working sum
        sum_of_sq_diffs = 0.0

        # traverse coordinate dimension
        for i in enumerate(coord1):
            # sum of squared differences
            sum_of_sq_diffs += (coord2[i[0]] - coord1[i[0]])**2

        return np.sqrt(sum_of_sq_diffs)

    def new_argmin_n_lowest_2d(self,
                               input_array: np.array,
                               n_lowest: int,
                               ignore_zeroes: bool = True
                               ) -> np.array:
        """Return indices of n lowest min values in a 2D array.

        Return format: array of coordinate tuples.
        """
        # check for null array
        if not input_array.any():
            return np.array([])

        # n lowest parameter check
        if n_lowest > input_array.size:
            n_lowest = input_array.size
        elif n_lowest < 1:
            n_lowest = 1

        # initialise list of min value coordinates
        output_array = np.zeros((n_lowest,), dtype='O')

        # initialise flattened array
        flat = np.ravel(input_array.copy())

        # convert zeroes in array to NaN, if ignoring
        if ignore_zeroes:
            np.place(flat, flat == 0, np.nan)

        # initialise working variables for indices
        idx_1d = np.argsort(flat)  # sorted list of ascending indices (1d)
        idx_2d = [0, 0]  # 2d coordinate index

        # search array for minimum values n times
        for i in range(n_lowest):

            # convert to 2d coordinate tuple
            idx_2d[1] = int(
                idx_1d[i] % input_array.shape[1]
                )
            idx_2d[0] = int(
                (idx_1d[i] - idx_2d[1]) / input_array.shape[0]
                )

            # add 2d coordinate tuple to output
            output_array[i] = tuple(idx_2d)

            # mark current min as NaN, to \
            # reveal next minimum value
            flat[idx_1d[i]] = np.nan

        return output_array

# =============================================================================
# ===== BUILD CIRCUITS =====
# =============================================================================

    def build_circuits(self,
                       n_shortest_conns: int = 1000
                       ):
        """Calculate jbox circuit sizes, given an input of jbox coordinates.

        Circuits built on increasing euclidean distance between components.
        """
        # ===== INITIALISE VARIABLES =====

        # total possible number of connections
        # nCr (combinatorics): len(jbox) choose 2 = total number of pairs
        possible_conns = int(
            (len(self.jbox) * (len(self.jbox) - 1)) / 2
            )

        # cap / floor size of n
        if n_shortest_conns > possible_conns:
            n_shortest_conns = possible_conns
        elif n_shortest_conns < 1:
            n_shortest_conns = 1

        # initialise circuits (list of sets)
        self.circuits.clear()

        # initialise working dict of circuits (dict of sets of jboxes)
        circuit_made = {}

        # ===== ESTABLISH CONNECTIONS TO MAKE (N SHORTEST) =====

        # obtain array of coordinate tuples, \
        # representing bilateral connections
        self.connections = self.new_argmin_n_lowest_2d(
            self.distance_matrix,
            n_shortest_conns
            )

        # ===== MAKE CONNECTIONS =====

        # Note: watch out for O(n^2) time complexity!

        # This works as each jbox connection is derived from upper diagonal
        # matrix i.e. first jbox index < second jbox index, and any jbox index
        # only has higher value jboxes in its set

        # populate circuits made

        # ...firstly, with single, unconnected jboxes
        for i in enumerate(self.jbox):
            circuit_made[i[0]] = {i[0]}

        # ...then loop through connections, adding to made circuits...
        for j in self.connections:
            # for each connection, add second jbox to set indexed by first jbox
            circuit_made[j[0]].add(j[0])
            circuit_made[j[0]].add(j[1])

        # ...finally, combine related circuits...

        # traverses each circuit from highest index descending, checking for a
        # lower indexed circuit for mutual jboxes
        for circ_1 in range(len(circuit_made) - 1, 0, -1):  # descending
            # all indices lower than current circuit
            for circ_2 in range(circ_1):
                # if there is a mutual jbox between circuits...
                if len(
                        circuit_made[circ_1].intersection(
                            circuit_made[circ_2]
                            )
                        ) != 0:
                    # ... combine them in the latter circuit...
                    circuit_made[circ_2] = circuit_made[circ_1].union(
                        circuit_made[circ_2]
                        )
                    # ... and delete the former circuit
                    circuit_made[circ_1].clear()
                    # next circuit
                    break

        # final made circuits are list of sets of jbox indices
        self.circuits = list(circuit_made.values())
        # sort circuits by descending circuit length
        self.circuits.sort(
            key=len, reverse=True
            )

    def build_one_circuit(self):
        """Construct unifying jbox circuit, given an input of jbox coordinates.

        Circuits built on increasing euclidean distance between components.
        """
        # ===== INITIALISE VARIABLES =====

        # total possible number of connections
        # nCr (combinatorics): len(jbox) choose 2 = total number of pairs
        possible_conns = int(
            (len(self.jbox) * (len(self.jbox) - 1)) / 2
            )

        # initialise loop conditions / parameters
        circuits_amended = 0
        made_conns = 0

        # initialise circuits (list of sets)
        self.circuits.clear()

        # initialise working dict of circuits (dict of sets)
        circuit_made = {}
        circuit_wait = {}

        # initialise working list of related jboes
        related_jboxes = []

        # ===== ESTABLISH ALL POSSIBLE CONNECTIONS =====

        # obtain array of coordinate tuples, \
        # representing bilateral connections
        self.connections = self.new_argmin_n_lowest_2d(
            self.distance_matrix,
            possible_conns
            )

        # ===== MAKE CONNECTIONS =====

        # premise is to build a circuit with incrementally more connections \
        # until one unifying circuit is built. This is done by traversing \
        # all possible connections, and connecting to circuits already made, \
        # or putting them in a waiting list to be revisited.

        # Note: To avoid O(n^2) time complexity, using more space (2 dicts).
        # also, specifically dicts because of O(1) 'in' time complexity.

        # begin loop in which a single new connection is added into circuits
        # each epoch

        # add first connection (made_conns = 0) to dict to seed the loop
        # first bilateral connection made
        # keys are indices of each jbox, value is index of connection

        circuit_made = {
            i: {made_conns} for i in self.connections[made_conns]
            }

        # increment connection
        made_conns += 1

        # ===== TRAVERSE POSSIBLE CONNECTIONS =====

        while made_conns < possible_conns:  # limit to max connections:

            # compare next connection to made circuits, checking \
            # if there is a mutual element between circuits...
            if len(
                    set(circuit_made).intersection(
                        set(self.connections[made_conns])
                        )
                    ) != 0:
                # ... if so, combine them...
                for i in self.connections[made_conns]:

                    circuit_made[i] = circuit_made[i].union({made_conns}) \
                        if i in circuit_made else {made_conns}

            else:
                # ... else add to circuits awaiting connection...
                for i in self.connections[made_conns]:

                    circuit_wait[i] = circuit_wait[i].union({made_conns}) \
                        if i in circuit_wait else {made_conns}

            # increment connection
            made_conns += 1

            # ===== COMBINE CIRCUITS MADE & WAITING LIST =====

            # ...finally, check for related jboxes amongst waiting list \
            # and made circuits (up to 2 degrees of separation)...

            while True:

                # assume no more amendments to any circuits \
                # at start of each loop
                circuits_amended = 0

                # ===== CONNECTIONS: 1 DEGREE OF SEPARATION =====

                # traverse sets of connection indices for jboxes in \
                # the waiting list that are also in made circuits
                for conn_indices in \
                    [
                        conn_indices for jbox, conn_indices in
                        circuit_wait.items() if jbox in circuit_made
                        ]:

                    # break traversal in waiting list if amended
                    # -> restart
                    if circuits_amended:
                        break

                    # ===== CONNECTIONS: 2 DEGREES OF SEPARATION =====

                    # traverse each connection index in those index sets to \
                    # find jboxes in waiting list that are related to made \
                    # circuits by 2 degrees of separation

                    for index in conn_indices:

                        # ... filter for all jboxes in waiting \
                        # list that contain a connection index that is \
                        # connected to a jbox that is in made circuits; phew!

                        # related_jboxes = list(
                        #         filter(
                        #             lambda x: len(
                        #                 circuit_wait.get(x).intersection(
                        #                     {index}
                        #                     )
                        #                 ) != 0,
                        #             circuit_wait
                        #             )
                        #         )

                        related_jboxes = \
                            [
                                x for x, y in circuit_wait.items() if
                                len(y.intersection({index})) != 0
                                ]

                        # for each related jbox....
                        for i in related_jboxes:

                            # ... combine with the made circuit
                            circuit_made[i] = circuit_made[i].union(
                                    circuit_wait[i]
                                    ) if \
                                i in circuit_made else circuit_wait[i]

                            # ... and delete the connection index in \
                            # waiting list, as it has been combined...
                            circuit_wait[i].remove(index)

                        # flag amendment to loop
                        circuits_amended = 1

                        # next connection index set
                        break

                if not circuits_amended:
                    break

            # when single circuit achieved...
            if len(circuit_made) == len(self.jbox):
                # ... truncate connections...
                self.connections = self.connections[:made_conns]
                # sort circuits by descending circuit length
                self.circuits = list(circuit_made.values())
                # ... end loop...
                break

# =============================================================================
# ===== PART ONE SOLVE =====
# =============================================================================

    def solve_part_one(self, n: int = 1000) -> int:
        """Solve of Part 1.

        Return product of top three jbox circuit sizes.
        """
        # Build circuits, based on problem parameters
        self.build_circuits(
            n_shortest_conns=n
            )
        # Derive product of top three circuit lengths
        self.solutions["Part One"] = int(
            np.prod(
                [len(x) for x in self.circuits[:3]]
                )
            )
        # print and return
        print(f"Part One Solution:\t{self.solutions['Part One']}")

        return self.solutions["Part One"]

# =============================================================================
# ===== PART TWO SOLVE =====
# =============================================================================

    def solve_part_two(self) -> int:
        """Solve of Part 2.

        Return product of X coordinates of last jbox connected \
        making one large circuit.
        """
        # Build circuits, based on problem parameters
        self.build_one_circuit()
        # Derive product of last two connected jboxes x coordinates
        self.solutions["Part Two"] = int(
            np.prod(
                self.jbox[
                    [*self.connections[-1]]
                    ][:, 0]
                )
            )
        # print and return
        print(f"Part Two Solution:\t{self.solutions['Part Two']}")

        return self.solutions["Part Two"]


if __name__ == '__main__':
    solver = Solver8('Input8.txt')

    # from time import perf_counter

    # start = perf_counter()
    solver.solve_part_one()
    # end = perf_counter()
    # print(f"{end-start}")

    # start = perf_counter()
    solver.solve_part_two()
    # end = perf_counter()
    # print(f"{end-start}")
