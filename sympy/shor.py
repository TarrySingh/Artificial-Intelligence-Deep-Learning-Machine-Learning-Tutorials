from __future__ import print_function

import math
import random
import sys
from fractions import Fraction
try:
    from math import gcd
except ImportError:
    from fractions import gcd

from builtins import input

import projectq.libs.math
import projectq.setups.decompositions
from projectq.backends import Simulator, ResourceCounter
from projectq.cengines import (MainEngine,
                               AutoReplacer,
                               LocalOptimizer,
                               TagRemover,
                               InstructionFilter,
                               DecompositionRuleSet)
from projectq.libs.math import (AddConstant,
                                AddConstantModN,
                                MultiplyByConstantModN)
from projectq.meta import Control
from projectq.ops import (X,
                          Measure,
                          H,
                          R,
                          QFT,
                          Swap,
                          get_inverse,
                          BasicMathGate)


def run_shor(eng, N, a, verbose=False):
    """
    Runs the quantum subroutine of Shor's algorithm for factoring.
    Args:
        eng (MainEngine): Main compiler engine to use.
        N (int): Number to factor.
        a (int): Relative prime to use as a base for a^x mod N.
        verbose (bool): If True, display intermediate measurement results.
    Returns:
        r (float): Potential period of a.
    """
    n = int(math.ceil(math.log(N, 2)))

    x = eng.allocate_qureg(n)

    X | x[0]

    measurements = [0] * (2 * n)  # will hold the 2n measurement results

    ctrl_qubit = eng.allocate_qubit()

    for k in range(2 * n):
        current_a = pow(a, 1 << (2 * n - 1 - k), N)
        # one iteration of 1-qubit QPE
        H | ctrl_qubit
        with Control(eng, ctrl_qubit):
            MultiplyByConstantModN(current_a, N) | x

        # perform inverse QFT --> Rotations conditioned on previous outcomes
        for i in range(k):
            if measurements[i]:
                R(-math.pi/(1 << (k - i))) | ctrl_qubit
        H | ctrl_qubit

        # and measure
        Measure | ctrl_qubit
        eng.flush()
        measurements[k] = int(ctrl_qubit)
        if measurements[k]:
            X | ctrl_qubit

        if verbose:
            print("\033[95m{}\033[0m".format(measurements[k]), end="")
            sys.stdout.flush()

    Measure | x
    # turn the measured values into a number in [0,1)
    y = sum([(measurements[2 * n - 1 - i]*1. / (1 << (i + 1)))
             for i in range(2 * n)])

    # continued fraction expansion to get denominator (the period?)
    r = Fraction(y).limit_denominator(N-1).denominator

    # return the (potential) period
    return r


# Filter function, which defines the gate set for the first optimization
# (don't decompose QFTs and iQFTs to make cancellation easier)
def high_level_gates(eng, cmd):
    g = cmd.gate
    if g == QFT or get_inverse(g) == QFT or g == Swap:
        return True
    if isinstance(g, BasicMathGate):
        return False
        if isinstance(g, AddConstant):
            return True
        elif isinstance(g, AddConstantModN):
            return True
        return False
    return eng.next_engine.is_available(cmd)


if __name__ == "__main__":
    # build compilation engine list
    resource_counter = ResourceCounter()
    rule_set = DecompositionRuleSet(modules=[projectq.libs.math,
                                             projectq.setups.decompositions])
    compilerengines = [AutoReplacer(rule_set),
                       InstructionFilter(high_level_gates),
                       TagRemover(),
                       LocalOptimizer(3),
                       AutoReplacer(rule_set),
                       TagRemover(),
                       LocalOptimizer(3),
                       resource_counter]

    # make the compiler and run the circuit on the simulator backend
    eng = MainEngine(Simulator(), compilerengines)

    # print welcome message and ask the user for the number to factor
    print("\n\t\033[37mprojectq\033[0m\n\t--------\n\tImplementation of Shor"
          "\'s algorithm.", end="")
    N = int(input('\n\tNumber to factor: '))
    print("\n\tFactoring N = {}: \033[0m".format(N), end="")

    # choose a base at random:
    a = int(random.random()*N)
    if not gcd(a, N) == 1:
        print("\n\n\t\033[92mOoops, we were lucky: Chose non relative prime"
              " by accident :)")
        print("\tFactor: {}\033[0m".format(gcd(a, N)))
    else:
        # run the quantum subroutine
        r = run_shor(eng, N, a, True)

        # try to determine the factors
        if r % 2 != 0:
            r *= 2
        apowrhalf = pow(a, r >> 1, N)
        f1 = gcd(apowrhalf + 1, N)
        f2 = gcd(apowrhalf - 1, N)
        if ((not f1 * f2 == N) and f1 * f2 > 1
           and int(1. * N / (f1 * f2)) * f1 * f2 == N):
            f1, f2 = f1*f2, int(N/(f1*f2))
        if f1 * f2 == N and f1 > 1 and f2 > 1:
            print("\n\n\t\033[92mFactors found :-) : {} * {} = {}\033[0m"
                  .format(f1, f2, N))
        else:
            print("\n\n\t\033[91mBad luck: Found {} and {}\033[0m".format(f1,
                                                                          f2))

        print(resource_counter)  # print resource usage