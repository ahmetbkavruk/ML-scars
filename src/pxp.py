from scipy.sparse import csr_matrix


class basisStates:
    def __init__(self, L):
        self.L = L
        self.dim = 2**L

    def state(self, index):
        return format(index, f"0{self.L}b")

    def getIndex(self, state):
        return int(state, 2)
    
    def getAllBasisStates(self):
        return [self.state(i) for i in range(self.dim)]
    
    def x(self, index, site):
        if index is None:
            return None

        site = site % self.L
        operated_state = self.state(index)

        flipped_bit = str(1 - int(operated_state[site]))

        new_state = operated_state[:site] + flipped_bit + operated_state[site+1:]

        return self.getIndex(new_state)

    def z(self, index, site):
        if index is None:
            return 0

        operated_state = self.state(index)

        if operated_state[site % self.L] == '0':
            return 1
        else:
            return -1
        
    def p(self, index, site):
        if index is None:
            return None

        operated_state = self.state(index)
    
        if operated_state[site % self.L] == '1':
            return None
        else:
            return index

    def pxp(self, index, site):

        new_index = self.p(index, (site - 1) % self.L)

        if new_index is None:
            return None

        new_index = self.p(new_index, (site + 1) % self.L)

        if new_index is None:
            return None

        new_index = self.x(new_index, site)

        return new_index

    def translate(self, index, shift = 1):
        if index is None:
            return None

        state = self.state(index)
        translated_state = state[-shift:] + state[:-shift]
        return self.getIndex(translated_state)


def build_deformed_pxp_hamiltonian(state, lambda2=0, lambda3=0, lambda22=0, delta=0, V=0, J=1):
    rows = []
    cols = []
    data = []

    for i in range(state.dim):
        diagonal = 0

        for site in range(state.L):

            # Diagonal terms
            n_i = (1 - state.z(i, site)) / 2
            n_i2 = (1 - state.z(i, site + 2)) / 2

            diagonal += J * (delta * n_i + V * n_i * n_i2)

            # Off-diagonal terms
            j = state.pxp(i, site)

            if j is not None:
                coefficient = J * (1
                    + lambda2 * (state.z(i, site - 2) + state.z(i, site + 2))
                    + lambda3 * (state.z(i, site - 3) + state.z(i, site + 3))
                    + lambda22 * (state.z(i, site - 2) * state.z(i, site + 2))
                )

                rows.append(j)
                cols.append(i)
                data.append(coefficient)

        if diagonal != 0:
            rows.append(i)
            cols.append(i)
            data.append(diagonal)

    H = csr_matrix((data, (rows, cols)), shape=(state.dim, state.dim))

    return H

def build_constrained_basis(L):
    basis = [0]

    for site in range(L):
        new_states = []

        current_bit = 1 << site

        for state in basis:
            if site > 0:
                previous_bit = 1 << (site - 1)

                if state & previous_bit:
                    continue

            # periodic boundary conditions
            if site == L - 1:
                first_bit = 1

                if state & first_bit:
                    continue

            new_state = state | current_bit
            new_states.append(new_state)

        basis.extend(new_states)

    return basis