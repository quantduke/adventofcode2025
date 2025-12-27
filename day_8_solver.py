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
                       jbox_coordinates: np.array,
                       n_shortest_conns: int = 1000
                       ):
        """Calculate jbox circuit sizes, given an input of jbox coordinates.

        Circuits built on increasing euclidean distance between components.
        """
        # ===== INITIALISE VARIABLES =====

        # total possible number of connections
        # nCr (combinatorics): len(jbox) choose 2 = total number of pairs
        possible_conns = int(
            (len(jbox_coordinates) * (len(jbox_coordinates) - 1)) / 2
            )

        # cap / floor size of n
        if n_shortest_conns > possible_conns:
            n_shortest_conns = possible_conns
        elif n_shortest_conns < 1:
            n_shortest_conns = 1

        # initialise euclidean distance matrix
        self.distance_matrix = np.zeros(
            (len(jbox_coordinates), len(jbox_coordinates))
        )

        # initialise circuits (list of sets)
        self.circuits.clear()

        # initialise working dict of circuits (dict of sets)
        circuit_curr = {}

        # ===== POPULATE DISTANCE MATRIX =====

        # opting for upper diagonal matrix, so that convention can be:
        # distance between matrix 1 + matrix 2 is [r, c]

        for j in enumerate(jbox_coordinates):  # columns
            for i in range(j[0]):  # rows, only upper diag
                # calculate the euclidean distance and populate
                self.distance_matrix[i, j[0]] = self.eucl_nd(
                    jbox_coordinates[i],
                    jbox_coordinates[j[0]]
                    )

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

        # initialise circuits...
        # ...firstly, with single, unconnected jboxes
        for i in enumerate(jbox_coordinates):
            circuit_curr[i[0]] = {i[0]}

        # ...then loop through connections, adding to dictionary...
        for j in self.connections:
            # for each connection, add second jbox to set indexed by first jbox
            circuit_curr[j[0]].add(j[0])
            circuit_curr[j[0]].add(j[1])

        # ...finally, combine related circuits...

        # traverses each circuit from highest index descending, checking for a
        # lower indexed circuit for mutual jboxes
        for circ_1 in range(len(circuit_curr) - 1, 0, -1):  # descending
            # all indices lower than current circuit
            for circ_2 in range(circ_1):
                # if there is a mutual jbox between circuits...
                if len(
                        circuit_curr[circ_1].intersection(
                            circuit_curr[circ_2]
                            )
                        ) != 0:
                    # ... combine them in the latter circuit...
                    circuit_curr[circ_2] = circuit_curr[circ_1].union(
                        circuit_curr[circ_2]
                        )
                    # ... and delete the former circuit
                    circuit_curr[circ_1].clear()
                    # next circuit
                    break

        # sort circuits by descending circuit length
        self.circuits = list(circuit_curr.values())

        self.circuits.sort(
            key=len, reverse=True
            )

    def build_one_circuit(self,
                          jbox_coordinates: np.array
                          ):
        """Construct unifying jbox circuit, given an input of jbox coordinates.

        Circuits built on increasing euclidean distance between components.
        """
        # ===== INITIALISE VARIABLES =====

        # total possible number of connections
        # nCr (combinatorics): len(jbox) choose 2 = total number of pairs
        possible_conns = int(
            (len(jbox_coordinates) * (len(jbox_coordinates) - 1)) / 2
            )

        # initialise euclidean distance matrix
        self.distance_matrix = np.zeros(
            (len(jbox_coordinates), len(jbox_coordinates))
        )

        # initialise loop conditions / parameters
        circuits_amended = 0
        current_conns = 0

        # initialise circuits (list of sets)
        self.circuits.clear()

        # initialise working dict of circuits (dict of sets)
        circuit_curr = {}
        circuit_wait = {}

        # initialise working list of related connections
        related_conns = []

        # ===== POPULATE DISTANCE MATRIX =====

        # opting for upper diagonal matrix, so that convention can be:
        # distance between matrix 1 + matrix 2 is [r, c]

        for j in enumerate(jbox_coordinates):  # columns
            for i in range(j[0]):  # rows, only upper diag
                # calculate the euclidean distance and populate
                self.distance_matrix[i, j[0]] = self.eucl_nd(
                    jbox_coordinates[i],
                    jbox_coordinates[j[0]]
                    )

        # ===== ESTABLISH ALL POSSIBLE CONNECTIONS =====

        # obtain array of coordinate tuples, \
        # representing bilateral connections
        self.connections = self.new_argmin_n_lowest_2d(
            self.distance_matrix,
            possible_conns
            )

        # ===== MAKE CONNECTIONS =====

        # premise is to build a circuit with incrementally more connections \
        # until one unifying circuit is built

        # Note: To avoid O(n^2) time complexity, using more space (2 dicts).
        # also, specifically dicts because of O(1) 'in' time complexity.

        # begin loop in which a single new connection is added into circuits
        # each epoch

        # add first connection to dict to seed the loop
        for i in self.connections[current_conns]:
            circuit_curr[i] = {current_conns}  # value is index of connection
        # increment connection
        current_conns += 1

        while True:

            if current_conns < possible_conns:  # limit to max connections

                # compare next connection to current circuit, checking \
                # if there is a mutual element between circuits...
                if len(
                        set(circuit_curr).intersection(
                        set(self.connections[current_conns]))
                        ) != 0:
                    # ... if so, combine them...
                    for i in self.connections[current_conns]:
                        if i in circuit_curr:
                            circuit_curr[i].add(current_conns)
                        else:
                            circuit_curr[i] = {current_conns}
                else:
                    # ... else add to circuits awaiting connection...
                    for i in self.connections[current_conns]:
                        if i in circuit_wait:
                            circuit_wait[i].add(current_conns)
                        else:
                            circuit_wait[i] = {current_conns}

                # increment connection
                current_conns += 1

            # ...finally, check for related jboxes amongst waiting list \
            # and current circuits...

            while True:

                # assume no more amendments to any circuits \
                # at start of each loop
                circuits_amended = 0

                # traverses waiting list checking for a mutual element in \
                # current circuit dict
                for jbox, conn_ind in circuit_wait.items():

                    # break traversal in waiting list if amended
                    # -> restart
                    if circuits_amended:
                        break

                    # traverse each connection index
                    for ind in conn_ind:

                        # if there is a mutual jbox between circuits...
                        if jbox in circuit_curr:

                            # ... filter for related connections in waiting \
                            # list...
                            related_conns = list(
                                    filter(
                                        lambda x: len(
                                            circuit_wait.get(x).intersection(
                                                {ind}
                                                )
                                            ) != 0,
                                        circuit_wait
                                        )
                                    )

                            # for each related connection....
                            for i in related_conns:
                                # ... combine them in the current circuit dict
                                if i in circuit_curr:  # O(1)
                                    circuit_curr[i] = circuit_curr[i].union(
                                        circuit_wait[i]
                                        )
                                else:
                                    circuit_curr[i] = circuit_wait[i]

                                # ... and delete the connection index in \
                                # waiting list...
                                circuit_wait[i].remove(ind)

                            # flag amendment to loop
                            circuits_amended = 1

                            # next jbox
                            break

                if not circuits_amended:
                    break

            # when single circuit achieved...
            if len(circuit_curr) == len(jbox_coordinates):
                # ... truncate connections...
                self.connections = self.connections[:current_conns]
                # sort circuits by descending circuit length
                self.circuits = list(circuit_curr.values())
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
            jbox_coordinates=self.jbox,
            n_shortest_conns=n
            )
        # Derive product of top three circuit lengths
        self.solutions["Part One"] = int(
            np.prod([len(x) for x in self.circuits[:3]])
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
        self.build_one_circuit(
            jbox_coordinates=self.jbox
            )
        # Derive product of last two connected jboxes x coordinates
        self.solutions["Part Two"] = int(
            np.prod(self.jbox[[*self.connections[-1]]][:, 0])
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
