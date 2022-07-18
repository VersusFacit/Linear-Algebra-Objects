#
# Matrix library
#

# Pages 20-21 of Sedgewick hint towards a nested list matrix interface.
# This module provides those and expands with traversals, rotations, and
# fundamental linear operations.

# assumption: subsequent vectors produce subducting rows
import copy
import itertools

from collections import namedtuple


class MatrixOperationError(Exception):
    pass


class MatrixSolveError(Exception):
    pass


class MatrixInitializationError(Exception):
    pass


class MatrixRow(object):
    def __init__(self, row):
        self.__row__ = row
        self.size = len(row)

    def extend(self, other):
        return MatrixRow(self.__row__ + other.__row__)

    def is_zero_row(self):
        return all((item == 0 for item in self.__row__))

    def __getitem__(self, key):
        if isinstance(key, slice):
            indices = range(*key.indices(len(self.__row__)))
            return MatrixRow([self.__row__[i] for i in indices])
        return self.__row__[key]

    def __reversed__(self):
        for elem in self.__row__[::-1]:
            yield elem

    def __setitem__(self, key, value):
        self.__row__[key] = value

    def __add__(self, other):
        return MatrixRow(
            [e1+e2 for (e1, e2) in zip(self.__row__, other.__row__)]
        )

    def __sub__(self, other):
        return MatrixRow(
            [e1-e2 for (e1, e2) in zip(self.__row__, other.__row__)]
        )

    def __mul__(self, factor):
        return MatrixRow(
            [factor*e for e in self.__row__]
        )

    def __floordiv__(self, factor):
        return self.__truediv__(factor)

    def __truediv__(self, factor):
        return MatrixRow(
            [e/factor for e in self.__row__]
        )

    def __str__(self):
        return "MatrixRow(" + ", ".join(
            map(lambda x: str(x), self.__row__)
        ) + ")"


class MatrixRowList(object):
    def __init__(self, row_list):
        head_row_size = row_list[0].size
        if any([head_row_size != row.size for row in row_list]):
            raise ValueError(
                'MatrixRowList has non uniform row size.'
            )

        self.__row_list__ = row_list

        self.row_count = len(row_list)
        if self.row_count <= 0:
            raise ValueError(
                'MatrixRowList has no rows.'
            )

        self.column_count = head_row_size
        if self.column_count <= 0:
            raise ValueError(
                'MatrixRowList has no columns.'
            )

    def __getitem__(self, key):
        return self.__row_list__[key]

    def __setitem__(self, key, value):
        self.__row_list__[key] = value

    def __add__(self, other):
        return MatrixRowList(
            [r1+r2 for (r1, r2) in zip(self.__row_list__, other.__row_list__)]
        )

    def __reversed__(self):
        for elem in self.__row_list__[::-1]:
            yield elem

    def __str__(self):
        newline = "\n"
        header = newline + " " * 2
        return "MatrixRowList(" + header + header.join(
            map(lambda x: str(x), self.__row_list__)
        ) + newline + ")"


class MatrixBase(object):
    def __init__(self, init_param):
        if isinstance(init_param, list):
            self.validate_input_list(init_param)
            self.row_count = len(init_param)
            self.column_count = len(init_param[0])
            self.data = MatrixRowList([MatrixRow(row) for row in init_param])
        elif isinstance(init_param, MatrixRowList):
            self.row_count = init_param.row_count
            self.column_count = init_param.column_count
            self.data = init_param
        else:
            raise MatrixInitializationError(
                f'Encountered unhandled initialization parameter {init_param}'
            )

    def validate_input_list(self, input_2d_list):
        if not input_2d_list:
            raise MatrixInitializationError(
                'No input list given for initialization.'
            )

        if len(input_2d_list) == 0:
            raise MatrixInitializationError(
                'Cannot initialize Matrix from empty list (input'
                ' represents rowless matrix)'
            )

        if len(input_2d_list[0]) == 0:
            raise MatrixInitializationError(
                'Cannot initialize Matrix from list with empty columns'
                ' (input represents columnness matrix)'
            )

        row_length = len(input_2d_list[0])
        if any([row_length != len(row) for row in input_2d_list]):
            raise MatrixInitializationError(
                'Length of rows in input list is not uniform.'
            )

    def __getitem__(self, key):
        return self.data[key]

    def __str__(self):
        '''Fails if self.data is unset but this shouldn't happen since
        all MatrixBase types set the data attribute on initialization.'''
        newline = '\n'
        tabstop = 2
        header = newline + ' ' * tabstop
        padding = max(map(
            lambda x: len(str(round(x, 5))),
            itertools.chain.from_iterable(self.data))
        )

        comma_sep_rows = (
            ', '.join(map(
                lambda x: str(round(x, 5)).rjust(padding),
                row
            )) for row in self.data
        )
        return '[' + header + header.join(comma_sep_rows) + newline + ']'


class CoefficientMatrix(MatrixBase):
    '''Factory function to assemble full augmented matrix from constituents.'''
    def build_augmented_matrix(self, rhs_constants_matrix):
        # check is actually a constants matrix
        if not isinstance(rhs_constants_matrix, RHSConstantsMatrix):
            bad_value = rhs_constants_matrix
            raise MatrixOperationError(
                f'Cannot build an augmented matrix using a {type(bad_value)}'
                f' with value {bad_value}.'
            )

        if self.row_count != rhs_constants_matrix.row_count:
            raise MatrixOperationError(
                'This CoefficientMatrix and the supplied RHSConstantMatrix'
                f' have conflicting row counts ({self.row_count} and'
                f' {rhs_constants_matrix.row_count}, respectively).'
            )

        combined_list = MatrixRowList([
            row_lhs.extend(row_rhs) for row_lhs, row_rhs
            in zip(self.data, rhs_constants_matrix.data)
        ])
        return Matrix(combined_list)


class RHSConstantsMatrix(MatrixBase):
    def validate_input_list(self, init_param):
        if not isinstance(init_param, MatrixRowList):
            super().validate_input_list(init_param)
            column_count = len(init_param[0])
            if column_count != 1:
                raise MatrixInitializationError(
                    'A RHSConstantsMatrix can only be initialized as a'
                    ' one-column matrix (input given has too many columns to'
                    ' represent the righthand equation constants of a linear'
                    ' system).'
                )


class Matrix(MatrixBase):
    '''Construct a MatrixRowList of MatrixRows'''
    def __init__(self, init_param):
        '''Initialize a two dimensional zero matrix based on row and column values'
        or a sufficiently large 2d list.

        Uses range class to avoid shared references from concatenation of'
        sequence class.'''
        # 1. Zero Matrix by dimension
        if isinstance(init_param, tuple):
            self.validate_input_dimensions(init_param)

            row_count = init_param[0]
            column_count = init_param[1]

            self.data = MatrixRowList(
                [MatrixRow([0]*column_count) for _ in range(row_count)]
            )

            self.row_count = row_count
            self.column_count = column_count
            self.coefficient_column_count = column_count-1

        # 2. 2d list supplied (i.e. list of lists)
        elif isinstance(init_param, list):
            self.validate_input_list(init_param)

            self.row_count = len(init_param)
            self.column_count = len(init_param[0])
            self.coefficient_column_count = len(init_param[0])-1
            self.data = MatrixRowList([MatrixRow(row) for row in init_param])

        # 3. MatrixRowList (i.e. already existing matrices)
        elif isinstance(init_param, MatrixRowList):
            self.row_count = init_param.row_count
            self.column_count = init_param.column_count
            self.coefficient_column_count = init_param.column_count-1
            self.data = init_param

        # 4. default, error out
        else:
            raise MatrixInitializationError(
                'Unhandled initialization parameter handed to Matrix: ' +
                str(init_param)
            )

    def get_coefficient_matrix(self):
        coefficient_list = MatrixRowList([
            row[0:self.coefficient_column_count] for row in self.data
        ])
        return CoefficientMatrix(coefficient_list)

    def get_rhs_constants_matrix(self):
        constants_list = MatrixRowList([
            row[self.coefficient_column_count:] for row in self.data
        ])
        return RHSConstantsMatrix(constants_list)

    def validate_input_list(self, init_param):
        super().validate_input_list(init_param)
        print(init_param)
        if len(init_param[0]) == 1:
            raise MatrixInitializationError(
                'Cannot initialize a one-column matrix (input given has'
                ' insufficient columns to represent a linear system)'
            )

    def validate_input_dimensions(self, init_param):
        # verify tuple is sized correctly; leave values to user
        if len(init_param) != 2:
            raise MatrixInitializationError(
                'Tuple initialization of a Matrix requires exactly two'
                ' params, row count and column count'
            )

        if init_param[0] <= 0:
            raise MatrixInitializationError(
                'Cannot initialize a rowless matrix'
            )
        elif init_param[1] <= 0:
            raise MatrixInitializationError(
                'Cannot initialize a columnless matrix'
            )
        elif init_param[1] == 1:
            raise MatrixInitializationError(
                'Cannot initialize a one-column matrix (input given has'
                ' insufficient columns to represent a linear system)'
            )

    #
    # Traversals
    #
    # m = self.row_count
    # n = self.column_count
    # array = self.data
    #
    # Row First
    def walk_ltr_down(self):
        # for(i=0; i<m; ++i):           # start at the top, move down
        #     for(j=0; i<n; ++j):       # move left to right
        #         array[i][j]
        return MatrixRowList([row for row in self.data])

    def walk_ltr_up(self):
        # for(i=m-1; i>=0; --i):        # start at the bottom, move up
        #     for(j=0; j<n; ++j):       # move left to right
        #         array[i][j]
        return MatrixRowList([row for row in reversed(self.data)])

    def walk_rtl_down(self):
        # for(i=0; i<m; ++i):           # start at the top, move down
        #     for(j=n-1; j>=0; --j):    # move right to left
        #         array[i][j]
        return MatrixRowList(
            [MatrixRow(list(reversed(row))) for row in self.data]
        )

    def walk_rtl_up(self):
        # for(i=m-1; i>=0; --i):        # start at bottom, move up
        #     for(j=n-1; j>=0; --j):    # move right to left
        #         array[i][j]
        return MatrixRowList(
            [MatrixRow(list(reversed(row))) for row in reversed(self.data)]
        )

    # Column First
    def walk_ttb_right(self):
        # for(j=0; j<n; ++j):           # start at leftmost column, move right
        #     for(i=0; i<m; ++i):       # move top to bottom
        #         array[i][j]
        return MatrixRowList([MatrixRow(column) for column in zip(*self.data)])

    def walk_btt_right(self):
        # for(j=0; j<n; ++j):           # start at leftmost column, move right
        #     for(i=m-1; i>=0; --i):    # move bottom to top
        #         array[i][j]
        return MatrixRowList(
            [MatrixRow(list(reversed(column))) for column in zip(*self.data)]
        )

    def walk_ttb_left(self):
        # for(j=n-1; j>=0; --j):        # start at rightmost column, move left
        #     for(i=0; i<m; ++i):       # move top to bottom
        #         array[i][j]
        return MatrixRowList(
            [MatrixRow(column) for column in reversed(list(zip(*self.data)))]
        )

    def walk_btt_left(self):
        # for(j=n-1; j>=0; --j):        # start at rightmost column, move left
        #     for(i=m-1; i>=0; --i):    # move bottom to top
        #         array[i][j]
        return MatrixRowList(
            [MatrixRow(list(reversed(column))) for column
                in list(reversed(list(zip(*self.data))))]
        )

    #
    # Rotations
    #
    def turn_left(self):
        return Matrix(self.walk_btt_right())

    def turn_right(self):
        return Matrix(self.walk_ttb_left())

    def mirror(self):
        return Matrix(self.walk_rtl_up())

    #
    # Arithmetic Operations
    #
    def add(self, other):
        if (
            self.row_count != other.row_count or
            self.column_count != other.column_count
        ):
            raise MatrixOperationError(
                'Matrices are not the same size and cannot be added:'
                f' {self.row_count}x{self.column_count} and'
                f' {other.row_count}x{other.column_count}.'
            )

        return Matrix(self.data + other.data)

    # multiply

    # scale

    #
    # Solving Matrices
    #
    def _find_lead_row(self, row_list, initial_row, column):
        '''Identifies highest magnitude value for column as lead to improve
        the numeric stability property.'''
        Lead = namedtuple('Lead', 'row_pointer magnitude')

        current_lead = Lead(None, 0)
        for row_num in range(initial_row, self.row_count):
            possible_lead = Lead(row_num, abs(row_list[row_num][column]))
            if possible_lead.magnitude > current_lead.magnitude:
                current_lead = possible_lead
        return current_lead.row_pointer

    def _cleanup_tall_matrices(self, row_list, aux_row, col):
        '''Sink all-zero rows to the bottom of the matrix.'''
        swap_row_offset = self.row_count - 1
        while aux_row < swap_row_offset:
            if row_list[aux_row][col] == 0:
                # When swapping a pivot row, thread the swap pointer up until
                # reaching a nonzero row so as not to interchange two zero
                # rows (i.e. a no-op).
                while row_list[swap_row_offset][col] == 0:
                    swap_row_offset -= 1
                row_list[swap_row_offset], row_list[aux_row] = (
                    row_list[aux_row], row_list[swap_row_offset]
                )
                swap_row_offset -= 1
            aux_row += 1
        return row_list

    def _check_solvability(self, matrix):
        for row in matrix:
            if row[-1] != 0 and set(row[0:-1]) == {0}:
                raise MatrixOperationError(
                    "Matrix has a row of all-zero coefficients and a nonzero "
                    "constant, so the Matrix is inconsistent."
                )

        if self.row_count < self.coefficient_column_count:
            raise MatrixSolveError(
                "Matrix has fewer equations than unknowns, so there is either "
                "not enough information to solve the Matrix or it has "
                "infinitely many solutions."
            )

        for row in matrix:
            if row.is_zero_row():
                raise MatrixSolveError(
                    "Matrix has some all-zero row, so it has infinitely many "
                    "solutions."
                )

    def row_echelon_form(self):
        '''Gaussian Elimination     O(m*n*min(m,n))

        A matrix is in row echelon form if it satisfies the following:

        * Each row has a leading value in the diagonal of the matrix.
        * Each column with a leading value has zeros in all its other entries.

        This implementation selects pivots in descending order of
        magnitude to reinforce numeric stability.
        '''
        row_list = copy.deepcopy(self.data)
        pivot_row = 0
        lead_offset = 0
        while (
            pivot_row < self.row_count and                  # stay within m
            lead_offset < self.coefficient_column_count     # stay within n-1
        ):
            # 1. Scan through unreduced rows of the matrix for next pivot
            row_pointer = self._find_lead_row(row_list, pivot_row, lead_offset)

            # no lead for pivot row/column offset pair; uptick and retry on row
            if row_pointer is None:
                lead_offset += 1
                continue

            # 2. Swap lead row with pivot to assure triangular matrix shape
            row_list[row_pointer], row_list[pivot_row] = (
                row_list[pivot_row], row_list[row_pointer]
            )

            # 3. Eliminate rows underneath
            for i in range(pivot_row + 1, self.row_count):
                ratio = -1 * (
                    row_list[i][lead_offset] / row_list[pivot_row][lead_offset]
                )
                row_list[i] += row_list[pivot_row] * ratio

            # Move down and designate as next pivot row
            pivot_row += 1

        # Add. For tall matrices, redorder zero rows
        if self.row_count > self.column_count:
            row_list = self._cleanup_tall_matrices(
                row_list,
                pivot_row,
                lead_offset
            )

        return Matrix(row_list)

    def solve_by_gaussian_elimination(self):
        '''Invokes Gaussian elimination and performs back substitution to
        solve the upper triangular matrix. Returns ordered list where ith
        value value matches x_i in the linear system. Ignores zero rows.'''
        reduced_row_list = self.row_echelon_form()

        self._check_solvability(reduced_row_list)

        solutions = [0 for _ in range(self.coefficient_column_count)]
        # For each row that might still have a nonzero unknown term.
        for diagonal in reversed(range(self.coefficient_column_count)):
            # Determine the value a for solving the equiation a*x_i = c
            solutions[diagonal] = (
                reduced_row_list[diagonal][self.coefficient_column_count] /
                reduced_row_list[diagonal][diagonal]
            )
            # Subtract newfound solution from upper constants to eliminate it
            # from further calculations (the equivalent of substituting into
            # an equation and then that newly determined term from both sides).
            for upper_row in reversed(range(diagonal)):
                reduced_row_list[upper_row][self.column_count-1] -= (
                    reduced_row_list[upper_row][diagonal] * solutions[diagonal]
                )
        return solutions

    def reduced_row_echelon_form(self):
        '''Gauss-Jordan Elimination O(m*n*min(m,n))

        A matrix is in reduced row echelon form (also called row canonical
        form) if it satisfies the following conditions:

        * It is in row echelon form.
        * The leading entry in each nonzero row is a 1 (called a leading 1).
        * Each column with a leading 1 has zeros in all its other entries.

        This implementation selects pivots in descending order of
        magnitude to reinforce numeric stability.
        '''
        row_list = copy.deepcopy(self.data)
        pivot_row = 0
        lead_offset = 0
        while (
            pivot_row < self.row_count and                  # stay within m
            lead_offset < self.coefficient_column_count     # stay within n-1
        ):
            # 1. Scan through unreduced rows of the matrix for next pivot
            row_pointer = self._find_lead_row(row_list, pivot_row, lead_offset)

            # no lead found so uptick the column offset and retry on pivot row
            if row_pointer is None:
                lead_offset += 1
                continue

            # 2. Swap lead row with pivot and unitize
            row_list[row_pointer], row_list[pivot_row] = (
                row_list[pivot_row], row_list[row_pointer]
            )
            row_list[pivot_row] /= row_list[pivot_row][lead_offset]

            # 3. Eliminate other rows
            for j in filter(lambda x: x != pivot_row, range(self.row_count)):
                scale = -1 * row_list[j][lead_offset]
                row_list[j] += row_list[pivot_row]*scale

            # Move down and designate as next pivot row
            pivot_row += 1

        # Add. For tall matrices, redorder zero rows
        if self.row_count > self.column_count:
            row_list = self._cleanup_tall_matrices(
                row_list,
                pivot_row,
                lead_offset
            )

        return Matrix(row_list)

    def solve_by_gauss_jordan(self):
        '''Invokes Gauss-Jordan elimination and extracts results from
        cannonical matrix. Returns ordered list where ith value value matches
        x_i in the linear system. Ignores zero rows.'''
        reduced_row_list = self.reduced_row_echelon_form()

        self._check_solvability(reduced_row_list)

        return [row[self.column_count-1] for row in reduced_row_list]
