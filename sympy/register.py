import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt


class QuantumRegister(object):
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.n_states = 2 ** n_qubits
        self.qubits = np.zeros(self.n_states, dtype=complex)
        self.qubits[0] = 1.0

    def reset(self):
        self.qubits = np.zeros(self.n_states, dtype=complex)
        self.qubits[0] = 1.0

    # REGISER MANIPULATION

    def isset(self, state, n):
        return state & 1 << (self.n_qubits - 1 - n) != 0

    def flip(self, state, n):
        return state ^ 1 << (self.n_qubits - 1 - n)

    def set_qubit(self, n, a, b):  # a|0>+b|1>
        tmp_qubits = np.zeros(self.n_states, dtype=complex)
        for state in range(self.n_states):
            current_amplitude = self.qubits[state] + self.qubits[self.flip(state, n)]
            if self.isset(state, n):
                tmp_qubits[state] = current_amplitude * b
            else:
                tmp_qubits[state] = current_amplitude * a
        self.qubits = tmp_qubits

    # MEASUREMENT OPERATIONS

    def measure(self):
        probabilities = np.absolute(self.qubits)**2
        return random.choice(len(probabilities), p=probabilities.flatten())

    def probability(self, qubits):
        assert len(qubits) == self.n_qubits
        probability = 0.0
        for state in range(self.n_states):
            selected = True
            for i in range(self.n_qubits):
                if qubits[i] is not None:
                    selected &= (self.isset(state, i) == qubits[i])
            if selected:
                probability += np.absolute(self.qubits[i])**2
            print(state, selected, probability)
        return probability

    # QUANTUM GATES

    def hadamar(self, qubits=None):
        if qubits is None:
            qubits = [1] * self.n_qubits
        H = 1. / np.sqrt(2) * np.array([[1., 1.], [1., -1.]])
        m = np.array([1])
        for indicator in reversed(qubits):
            m = np.kron(H, m) if indicator else np.kron(np.eye(2), m)
        self.qubits = m.dot(self.qubits)
        return self

    def hadamar_alternative(self):
        hadamar = np.zeros((self.n_states, self.n_states))
        for target in range(self.n_states):
            for state in range(self.n_states):
                hadamar[target, state] = (2.**(-self.n_qubits / 2.))*(-1)**bin(state & target).count("1")
        self.qubits = hadamar.dot(self.qubits)
        return self

    def cswap(self, c, a, b):
        cswap = np.zeros((self.n_states, self.n_states))
        for state in range(self.n_states):
            if self.isset(state, c):
                if self.isset(state, a) != self.isset(state, b):
                    flipstate = self.flip(self.flip(state, b), a)
                    cswap[state, flipstate] = 1.0
                else:
                    cswap[state, state] = 1.0
            else:
                cswap[state, state] = 1.0
        self.qubits = cswap.dot(self.qubits)
        return self

    # IMPLEMENTATION ESSENTIALS

    def __str__(self):
        string = ""
        for state in range(self.n_states):
            string += "{0:0>3b}".format(state) + " => {:.2f}".format(self.qubits[state]) + "\n"
        return string[:-1]

    def plot(self):
        plt.bar(range(self.n_states), np.absolute(self.qubits), color='k')
        plt.title(str(self.n_qubits) + ' qubit register')
        plt.axis([0, self.n_states, 0.0, 1.0])
        plt.show()

    def savefig(self, name):
        plt.bar(range(self.n_states), np.absolute(self.qubits), color='k')
        plt.title(str(self.n_qubits) + ' qubit register')
        plt.axis([0, self.n_states, 0.0, 1.0])
        plt.savefig("img/" + name + ".pdf")

    def plot2(self, save=None, name=None):
        cols = 2 ** (self.n_qubits / 2)  # integer division!
        rows = 2 ** (self.n_qubits - (self.n_qubits / 2))

        x = []
        y = []
        c = []

        for i in range(self.n_states):
            x.append(i % cols)
            y.append(i / cols)
            c.append(np.absolute(self.qubits[i]))

        plt.xlim(-0.5, cols-0.5)
        plt.ylim(-0.5, rows-0.5)

        plt.axes().set_aspect('equal')

        plt.scatter(x, y, s=2e3, c=c, linewidths=2, vmin=0, vmax=1, cmap=plt.get_cmap('jet'))

        if save is None:
            plt.show()
        else:
            plt.axis('off')
            plt.title('('+name+')')

            fig = plt.gcf()
            fig.set_size_inches(cols, rows)
            fig.savefig("img/" + save + ".pdf", transparent=True, pad_inches=0)