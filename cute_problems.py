import numpy as np
from f_bfgs import gradient_descent, bfgs, l_bfgs, fast_bfgs


# All unconstrained problems are refer to these two article:
# [1] Ladislav Luksan, Ctirad Mationoha, Jan Vlcek: Modified CUTE Problems for Sparse Unconstrained Optimizatiuon.
# [2] Neculai Andrei: An Unconstrained Optimization Test Functions Collection.

# ------------------------- [m = 8, ||gk|| < 1e-5] ------------------------- #
# +--------------------------------------------------------------------------+
# |          |      | GD    | BFGS  | L-BFGS | secant=False  |  secant=True  |
# |          |      |       |       |        +-------+-------+-------+-------+
# | name     | n    |       |       |        | ver-A | ver-B | ver-A | ver-B |
# +----------+------+-------+-------+--------+-------+-------+-------+-------+
# | ARWHEAD  | 1024 | >1000 | 39    | 26     | 21    | 16    | 21    | 16    |
# | BDQRTIC  | 1024 | >1000 | --    | --     | 491   | 317   | 516   | 315   |
# | BDEXP    | 1024 | >1000 | 19    | 19     | 9     | 9     | 9     | 9     |
# | COSINE   | 1024 | >1000 | --    | --     | 44    | 16    | 43    | 16    |
# | DIXMAANE | 1500 | >1000 | 195   | 244    | 586   | 326   | 597   | 312   |
# | DIXMAANF | 1500 | >1000 | 336   | 216    | 423   | 265   | 391   | 206   |
# | DIXMAANG | 1500 | >1000 | 954   | 384    | 460   | 211   | 414   | 206   |
# | DQRTIC   | 1000 | --    | --    | --     | 35    | 31    | 35    | 31    |
# | EDENSCH  | 1000 | 59    | 86    | 52     | 42    | 23    | 42    | 23    |
# | ENGVAL1  | 1000 | 66    | 154   | 119    | 39    | 24    | 38    | 24    |
# | EG2      | 1000 | 7     | 6     | 6      | 8     | 8     | 8     | 8     |
# | EXTROSNB | 1000 | 63    | 309   | 333    | 76    | 41    | 73    | 43    |
# | FLETCHER | 100  | >1000 | --    | --     | >1000 | 734   | >1000 | 752   |
# | FREUROTH | 1000 | --    | --    | --     | 51    | 45    | 52    | --    |
# | GENROSE  | 1000 | >1000 | >1000 | 39     | 48    | --    | 50    | --    |
# | HIMMELBG | 1000 | >1000 | 3     | 3      | 3     | 3     | 3     | 3     |
# | HIMMELH  | 1000 | 20    | 9     | 9      | 19    | 16    | 19    | 16    |
# | LIARWHD  | 1000 | >1000 | --    | 28     | 40    | 30    | 40    | 30    |
# | NONDIA   | 1000 | >1000 | --    | 55     | 97    | 76    | 89    | --    |
# | NONDQUAR | 1000 | >1000 | 270   | 320    | 344   | 230   | 296   | 260   |
# | NONSCOMP | 1000 | 86    | 286   | 238    | 101   | 45    | 102   | 46    |
# | POWELLSG | 1000 | >1000 | 459   | 49     | 69    | 63    | 64    | 64    |
# | SCHMVETT | 1000 | 181   | 26    | 24     | 45    | 25    | 53    | 24    |
# | SINQUAD  | 1000 | >1000 | 140   | 143    | --    | --    | 109   | --    |
# | SROSENBR | 1000 | >1000 | --    | 39     | 48    | --    | 48    | --    |
# | TOINTGSS | 1000 | 6     | 9     | 9      | 8     | 7     | 8     | 7     |
# | TQUARTIC | 1000 | >1000 | 16    | 17     | 28    | 24    | 28    | 24    |
# | WOODS    | 1000 | >1000 | --    | 92     | --    | 48    | --    | --    |
# +--------------------------------------------------------------------------+


class Problem:
    _name = None

    @classmethod
    def function(cls, x):
        return

    @classmethod
    def gradients(cls, x):
        return

    @classmethod
    def gradients_check(cls, func, grad, x0, *args, **kwargs):
        n = x0.__len__()

        numeric_grad = np.zeros_like(x0)
        for i in range(n):
            x0_shift = np.zeros_like(x0)
            x0_shift[i] += 1e-6

            f_add = func(x0 + x0_shift, *args, **kwargs)
            f_sub = func(x0 - x0_shift, *args, **kwargs)

            numeric_grad[i] = (f_add - f_sub) / 2e-6

        # print(numeric_grad - grad(x0, *args, **kwargs))
        return np.max(np.abs(numeric_grad - grad(x0, *args, **kwargs)))

    @classmethod
    def gen_x0(cls, n):
        return 0

    @classmethod
    def solve(cls, n, GD=True, BFGS=True, L_BFGS=True, F_BFGS=True, m=8, secant=False, version="A"):
        print('-' * 32 + "\tSolving {}-dim {} problem\t".format(n, cls._name) + '-' * 32)

        # ---------------- check the gradients ----------------
        check = cls.gradients_check(cls.function, cls.gradients, np.random.rand(n))
        print("Check the gradients of {}: {:.2e}".format(cls._name, check))

        # ---------------- generate x0 ----------------
        x0 = cls.gen_x0(n)

        # ---------------- gradient descent ----------------
        if GD:
            _, log = gradient_descent(cls.function, cls.gradients, x0=x0)
            fk, gk, k = log[-1, -2], log[-1, -1], log.__len__()
            print("gd method: fk = {:.2e}, ||gk|| = {:.2e}, iters = {}".format(fk, gk, k))

        # ---------------- BFGS ----------------
        if BFGS:
            _, log = bfgs(cls.function, cls.gradients, x0=x0)
            fk, gk, k = log[-1, -2], log[-1, -1], log.__len__()
            print("bfgs method: fk = {:.2e}, ||gk|| = {:.2e}, iters = {}".format(fk, gk, k))

        # ---------------- L-BFGS ----------------
        if L_BFGS:
            _, log = l_bfgs(cls.function, cls.gradients, m, x0)
            fk, gk, k = log[-1, -2], log[-1, -1], log.__len__()
            print("l-bfgs(m={}) method: fk = {:.2e}, ||gk|| = {:.2e}, iters = {}".format(m, fk, gk, k))

        # ---------------- P-BFGS ----------------
        if F_BFGS:
            m = m if isinstance(m, list) else [m]
            secant = secant if isinstance(secant, list) else [secant]
            version = version if isinstance(version, list) else version
            for arg1 in m:
                for arg2 in secant:
                    for arg3 in version:
                        _, log = fast_bfgs(cls.function, cls.gradients, x0, m=arg1, secant=arg2, version=arg3)
                        fk, gk, k = log[-1, -2], log[-1, -1], log.__len__()
                        print("fast-bfgs(m={},secant={},version={}) method: "
                              "fk = {:.2e}, ||gk|| = {:.2e}, iters = {}".format(
                               arg1, arg2, arg3, fk, gk, k))


class CUTE:
    class ARWHEAD(Problem):
        _name = "ARWHEAD"

        @classmethod
        def function(cls, x):
            return np.sum((x[:-1] ** 2 + x[-1] ** 2) ** 2 - 4 * x[:-1] + 3)

        @classmethod
        def gradients(cls, x):
            grad = np.zeros_like(x)
            grad[:-1] = 4 * (x[:-1] ** 2 + x[-1] ** 2) * x[:-1] - 4
            grad[-1] = 4 * np.sum(x[:-1] ** 2 + x[-1] ** 2) * x[-1]
            return grad

        @classmethod
        def gen_x0(cls, n):
            return np.ones(shape=(n, ))

    class BDQRTIC(Problem):
        _name = "BDQRTIC"

        @classmethod
        def function(cls, x):
            n = x.__len__()
            if n < 4:
                raise ValueError("The length of argument `x` should be greater than 3.")

            x0 = x[:-4]
            x1 = x[1:-3]
            x2 = x[2:-2]
            x3 = x[3:-1]
            xn = x[-1]

            func = np.sum(np.square(3 - 4 * x0))
            func += np.sum(np.square(x0 ** 2 + 2 * x1 ** 2 + 3 * x2 ** 2 + 4 * x3 ** 2 + 5 * xn ** 2))

            return func

        @classmethod
        def gradients(cls, x):
            n = x.__len__()
            if n < 4:
                raise ValueError("The length of argument `x` should be greater than 3.")

            x0 = x[:-4]
            x1 = x[1:-3]
            x2 = x[2:-2]
            x3 = x[3:-1]
            xn = x[-1]

            grad = np.zeros_like(x)

            grad[:-4] += -8 * (3 - 4 * x0)
            grad[:-4] += 4 * (x0 ** 2 + 2 * x1 ** 2 + 3 * x2 ** 2 + 4 * x3 ** 2 + 5 * xn ** 2) * x0

            grad[1:-3] += 8 * (x0 ** 2 + 2 * x1 ** 2 + 3 * x2 ** 2 + 4 * x3 ** 2 + 5 * xn ** 2) * x1

            grad[2:-2] += 12 * (x0 ** 2 + 2 * x1 ** 2 + 3 * x2 ** 2 + 4 * x3 ** 2 + 5 * xn ** 2) * x2

            grad[3:-1] += 16 * (x0 ** 2 + 2 * x1 ** 2 + 3 * x2 ** 2 + 4 * x3 ** 2 + 5 * xn ** 2) * x3

            grad[-1] += 20 * np.sum(x0 ** 2 + 2 * x1 ** 2 + 3 * x2 ** 2 + 4 * x3 ** 2 + 5 * xn ** 2) * xn

            return grad

        @classmethod
        def gen_x0(cls, n):
            return np.ones(shape=(n, ))

    class BDEXP(Problem):
        _name = "BDEXP"

        @classmethod
        def function(cls, x):
            return np.sum((x[:-2] + x[1:-1]) * np.exp(-x[2:] * (x[:-2] + x[1:-1])))

        @classmethod
        def gradients(cls, x):
            grad = np.zeros_like(x)

            exp = np.exp(-x[2:] * (x[:-2] + x[1:-1]))

            grad[:-2] += exp
            grad[1:-1] += exp

            grad[2:] += -(x[:-2] + x[1:-1]) ** 2 * exp
            grad[:-2] += - x[2:] * (x[:-2] + x[1:-1]) * exp
            grad[1:-1] += - x[2:] * (x[:-2] + x[1:-1]) * exp

            return grad

        @classmethod
        def gen_x0(cls, n):
            return np.ones(shape=(n, ))

    class BIGGSB1(Problem):
        _name = "BIGGSB1"

        @classmethod
        def function(cls, x):
            func = (x[0] - 1) ** 2 + (x[-1] - 1) ** 2
            func += np.sum(np.square(x[1:] - x[:-1]))

            return func

        @classmethod
        def gradients(cls, x):
            grad = np.zeros_like(x)

            grad[0] += 2 * (x[0] - 1)
            grad[-1] += 2 * (x[-1] - 1)

            grad[1:] += 2 * (x[1:] - x[:-1])
            grad[:-1] += -2 * (x[1:] - x[:-1])

            return grad

        @classmethod
        def gen_x0(cls, n):
            x0 = np.zeros(shape=(n, ))  # TODO: not work well!
            return x0

    class COSINE(Problem):
        _name = "COSINE"

        @classmethod
        def function(cls, x):
            return np.sum(np.cos(x[:-1] ** 2 - 0.5 * x[1:]))

        @classmethod
        def gradients(cls, x):
            grad = np.zeros_like(x)

            grad[:-1] += -2 * np.sin(x[:-1] ** 2 - 0.5 * x[1:]) * x[:-1]
            grad[1:] += 0.5 * np.sin(x[:-1] ** 2 - 0.5 * x[1:])
            return grad

        @classmethod
        def gen_x0(cls, n):
            return np.ones(shape=(n, ))

    class CUBE(Problem):
        _name = "CUBE"

        @classmethod
        def function(cls, x):
            func = (x[0] - 1) ** 2
            func += 100 * np.sum(np.square(x[1:] - x[:-1] ** 3))

            return func

        @classmethod
        def gradients(cls, x):
            grad = np.zeros_like(x)

            grad[0] += 2 * (x[0] - 1)

            grad[1:] += 200 * (x[1:] - x[:-1] ** 3)
            grad[:-1] += -600 * (x[1:] - x[:-1] ** 3) * x[:-1] ** 2

            return grad

        @classmethod
        def gen_x0(cls, n):
            x0 = np.ones(shape=(n, ))
            x0[::2] *= -1.2
            return x0  # TODO: not work well!

    class DIXMAANE(Problem):
        _name = "DIXMAANE"

        alpha = 1.
        beta = 0.
        gamma = 0.125
        delta = 0.125

        @classmethod
        def function(cls, x):
            n = x.__len__()
            if n % 3 != 0:
                raise ValueError("The length of argument `x` should be triple of a positive integer.")
            m = n // 3

            v = np.array([i / n for i in range(1, n + 1)])

            func = 1 + cls.alpha * np.sum(v * x ** 2)
            func += cls.beta * np.sum(x[:-1] ** 2 * (x[1:] + x[1:] ** 2) ** 2)
            func += cls.gamma * np.sum(x[:2 * m] ** 2 * x[m:] ** 4)
            func += cls.delta * np.sum(v[:m] * x[:m] * x[2 * m:])
            return func

        @classmethod
        def gradients(cls, x):
            n = x.__len__()
            if n % 3 != 0:
                raise ValueError("The length of argument `x` should be triple of a positive integer.")
            m = n // 3

            v = np.array([i / n for i in range(1, n + 1)])

            grad = np.zeros_like(x)

            grad += cls.alpha * 2 * v * x

            grad[:-1] += cls.beta * 2 * x[:-1] * (x[1:] + x[1:] ** 2) ** 2
            grad[1:] += cls.beta * x[:-1] ** 2 * 2 * (x[1:] + x[1:] ** 2) * (1 + 2 * x[1:])

            grad[:2 * m] += cls.gamma * 2 * x[:2 * m] * x[m:] ** 4
            grad[m:] += cls.gamma * 4 * x[:2 * m] ** 2 * x[m:] ** 3

            grad[:m] += cls.delta * v[:m] * x[2 * m:]
            grad[2 * m:] += cls.delta * v[:m] * x[:m]

            return grad

        @classmethod
        def gen_x0(cls, n):
            return 2 * np.ones(shape=(n, ))

    class DIXMAANF(DIXMAANE):
        _name = "DIXMAANF"

        alpha = 1.
        beta = 0.0625
        gamma = 0.0625
        delta = 0.0625

        @classmethod
        def gen_x0(cls, n):
            return 2 * np.ones(shape=(n, ))

    class DIXMAANG(DIXMAANE):
        _name = "DIXMAANG"

        alpha = 1.
        beta = 0.125
        gamma = 0.125
        delta = 0.125

        @classmethod
        def gen_x0(cls, n):
            return 2 * np.ones(shape=(n, ))

    class DIXON3DQ(Problem):
        _name = "DIXON3DQ"

        @classmethod
        def function(cls, x):
            func = (x[0] - 1) ** 2 + (x[-1] - 1) ** 2
            func += np.sum(np.square(x[:-1] - x[1:]))

            return func

        @classmethod
        def gradients(cls, x):
            grad = np.zeros_like(x)

            grad[0] += 2 * (x[0] - 1)
            grad[-1] += 2 * (x[-1] - 1)

            grad[:-1] += 2 * (x[:-1] - x[1:])
            grad[1:] += -2 * (x[:-1] - x[1:])

            return grad

        @classmethod
        def gen_x0(cls, n):
            x0 = -np.ones(shape=(n, ))  # TODO: not work well!
            return x0

    class DQRTIC(Problem):
        _name = "DQRTIC"

        @classmethod
        def function(cls, x):
            v = np.array([i for i in range(1, x.__len__() + 1)])
            return np.sum(np.power(x - v, 4))

        @classmethod
        def gradients(cls, x):
            v = np.array([i for i in range(1, x.__len__() + 1)])
            return 4 * np.power(x - v, 3)

        @classmethod
        def gen_x0(cls, n):
            return 2 * np.ones(shape=(n, ))

    class EDENSCH(Problem):
        _name = "EDENSCH"

        @classmethod
        def function(cls, x):
            func = 16 + np.sum(np.power(x[:-1] - 2, 4))
            func += np.sum(np.square(x[:-1] * x[1:] - 2 * x[1:]))
            func += np.sum(np.square(x[1:] + 1))
            return func

        @classmethod
        def gradients(cls, x):
            grad = np.zeros_like(x)

            grad[:-1] += 4 * np.power(x[:-1] - 2, 3)

            grad[:-1] += 2 * (x[:-1] * x[1:] - 2 * x[1:]) * x[1:]
            grad[1:] += 2 * (x[:-1] * x[1:] - 2 * x[1:]) * x[:-1]
            grad[1:] += -4 * (x[:-1] * x[1:] - 2 * x[1:])

            grad[1:] += 2 * (x[1:] + 1)

            return grad

        @classmethod
        def gen_x0(cls, n):
            return np.zeros(shape=(n, ))

    class ENGVAL1(Problem):
        _name = "ENGVAL1"

        @classmethod
        def function(cls, x):
            func = np.sum(np.square(np.square(x[:-1]) + np.square(x[1:])))
            func += np.sum(-4 * x[:-1] + 3)
            return func

        @classmethod
        def gradients(cls, x):
            grad = np.zeros_like(x)

            grad[:-1] += 4 * (np.square(x[:-1]) + np.square(x[1:])) * x[:-1]
            grad[1:] += 4 * (np.square(x[:-1]) + np.square(x[1:])) * x[1:]

            grad[:-1] += -4

            return grad

        @classmethod
        def gen_x0(cls, n):
            return 2 * np.ones(shape=(n, ))

    class EG2(Problem):
        _name = "EG2"

        @classmethod
        def function(cls, x):
            return np.sum(np.sin(x[0] + x[:-1] ** 2 - 1)) + 0.5 * np.sin(x[-1] ** 2)

        @classmethod
        def gradients(cls, x):
            grad = np.zeros_like(x)
            grad[0] += np.sum(np.cos(x[0] + x[:-1] ** 2 - 1))
            grad[:-1] += np.cos(x[0] + x[:-1] ** 2 - 1) * 2 * x[:-1]
            grad[-1] += np.cos(x[-1] ** 2) * x[-1]
            return grad

        @classmethod
        def gen_x0(cls, n):
            return np.zeros(shape=(n, ))

    class EXTROSNB(Problem):
        _name = "EXTROSNB"

        @classmethod
        def function(cls, x):
            return (x[0] - 1) ** 2 + 100 * np.sum(np.square(x[1:] - x[:-1] ** 2))

        @classmethod
        def gradients(cls, x):
            grad = np.zeros_like(x)

            grad[0] += 2 * (x[0] - 1)
            grad[1:] += 200 * (x[1:] - x[:-1] ** 2)
            grad[:-1] += -400 * (x[1:] - x[:-1] ** 2) * x[:-1]
            return grad

        @classmethod
        def gen_x0(cls, n):
            return 1.5 * np.ones(shape=(n, ))

    class FLETCHER(Problem):
        _name = "FLETCHER"

        @classmethod
        def function(cls, x):
            return 100 * np.sum(np.square(x[1:] - x[:-1] + 1 - x[:-1] ** 2))

        @classmethod
        def gradients(cls, x):
            grad = np.zeros_like(x)

            grad[1:] += 200 * (x[1:] - x[:-1] + 1 - x[:-1] ** 2)
            grad[:-1] += -200 * (x[1:] - x[:-1] + 1 - x[:-1] ** 2)
            grad[:-1] += -400 * (x[1:] - x[:-1] + 1 - x[:-1] ** 2) * x[:-1]

            return grad

        @classmethod
        def gen_x0(cls, n):
            return np.zeros(shape=(n, ))

    class FREUROTH(Problem):
        _name = "FREUROTH"

        @classmethod
        def function(cls, x):
            n = x.__len__()
            if n < 3:
                raise ValueError("The length of argument `x` should be greater than 2.")

            func = np.sum(np.square((5 - x[1:]) * x[1:] ** 2 + x[:-1] - 2 * x[1:] - 13))
            func += np.sum(np.square((1 + x[1:]) * x[1:] ** 2 + x[:-1] - 14 * x[1:] - 29))

            return func

        @classmethod
        def gradients(cls, x):
            n = x.__len__()
            if n < 3:
                raise ValueError("The length of argument `x` should be greater than 2.")

            grad = np.zeros_like(x)

            grad[1:] += 2 * ((5 - x[1:]) * x[1:] ** 2 + x[:-1] - 2 * x[1:] - 13) * (10 * x[1:] - 3 * x[1:] ** 2 - 2)
            grad[:-1] += 2 * ((5 - x[1:]) * x[1:] ** 2 + x[:-1] - 2 * x[1:] - 13)

            grad[1:] += 2 * ((1 + x[1:]) * x[1:] ** 2 + x[:-1] - 14 * x[1:] - 29) * (2 * x[1:] + 3 * x[1:] ** 2 - 14)
            grad[:-1] += 2 * ((1 + x[1:]) * x[1:] ** 2 + x[:-1] - 14 * x[1:] - 29)

            return grad

        @classmethod
        def gen_x0(cls, n):
            x0 = np.zeros(shape=(n, ))
            x0[0] = 0.5
            x0[1] = -2.0
            return x0

    class GENROSE(Problem):
        _name = "GENROSE"

        @classmethod
        def function(cls, x):
            n = x.__len__()
            if n % 2 != 0:
                raise ValueError("The length of argument `x` should be double of a positive integer.")
            return np.sum(100 * (x[1::2] - x[::2] ** 2) ** 2 + (1 - x[::2]) ** 2)

        @classmethod
        def gradients(cls, x):
            n = x.__len__()
            if n % 2 != 0:
                raise ValueError("The length of argument `x` should be double of a positive integer.")
            grad = np.zeros_like(x)

            grad[1::2] += 200 * (x[1::2] - x[::2] ** 2)
            grad[::2] += -400 * (x[1::2] - x[::2] ** 2) * x[::2] - 2 * (1 - x[::2])

            return grad

        @classmethod
        def gen_x0(cls, n):
            x0 = np.zeros(shape=(n, ))
            x0[1::2] += 1.
            x0[::2] += -1.2
            return x0

    class HARKERP2(Problem):
        _name = "HARKERP2"

        @classmethod
        def function(cls, x):
            n = x.__len__()

            func = np.square(np.sum(x))
            func += -np.sum(x + 0.5 * x ** 2)
            for i in range(1, n):
                func += 2 * np.square(np.sum(x[i:]))
            func += x[-1] ** 2

            return func

        @classmethod
        def gradients(cls, x):
            n = x.__len__()

            grad = np.zeros_like(x)

            grad += 2 * np.sum(x)

            grad += -1 - x

            for i in range(1, n):
                grad[i:] += 4 * np.sum(x[i:])

            grad[-1] += 2 * x[-1]

            return grad

        @classmethod
        def gen_x0(cls, n):
            return np.array([i for i in range(1, n+1)])  # TODO: not work well!

    class HIMMELBG(Problem):
        _name = "HIMMELBG"

        @classmethod
        def function(cls, x):
            if x.__len__() % 2 != 0:
                raise ValueError
            return np.sum((2 * x[::2] ** 2 + 3 * x[1::2] ** 2) * np.exp(-x[::2] - x[1::2]))

        @classmethod
        def gradients(cls, x):
            grad = np.zeros_like(x)

            exp = np.exp(-x[::2] - x[1::2])

            grad[::2] += 4 * x[::2] * exp
            grad[1::2] += 6 * x[1::2] * exp

            grad[::2] += -(2 * x[::2] ** 2 + 3 * x[1::2] ** 2) * exp
            grad[1::2] += -(2 * x[::2] ** 2 + 3 * x[1::2] ** 2) * exp

            return grad

        @classmethod
        def gen_x0(cls, n):
            return 1.5 * np.ones(shape=(n, ))

    class HIMMELH(Problem):
        _name = "HIMMELH"

        @classmethod
        def function(cls, x):
            if x.__len__() % 2 != 0:
                raise ValueError
            return np.sum(-3 * x[::2] - 2 * x[1::2] + 2 + x[::2] ** 3 + x[1::2] ** 2)

        @classmethod
        def gradients(cls, x):
            grad = np.zeros_like(x)

            grad[::2] += -3 + 3 * x[::2] ** 2
            grad[1::2] += -2 + 2 * x[1::2]
            return grad

        @classmethod
        def gen_x0(cls, n):
            return 1.5 * np.ones(shape=(n, ))

    class INDEF(Problem):
        _name = "INDEF"

        @classmethod
        def function(cls, x):
            func = np.sum(x)
            func += 0.5 * np.sum(np.cos(2 * x[1:-1] - x[-1] - x[0]))

            return func

        @classmethod
        def gradients(cls, x):
            grad = np.zeros_like(x)

            grad += x

            grad[1:-1] += -np.sin(2 * x[1:-1] - x[-1] - x[0])
            grad[-1] += 0.5 * np.sum(np.sin(2 * x[1:-1] - x[-1] - x[0]))
            grad[0] += 0.5 * np.sum(np.sin(2 * x[1:-1] - x[-1] - x[0]))

            return grad

        @classmethod
        def gen_x0(cls, n):
            return np.array([i / (n + 1) for i in range(1, n + 1)])

    class LIARWHD(Problem):
        _name = "LIARWHD"

        @classmethod
        def function(cls, x):
            func = 4 * np.sum(np.square(x ** 2 - x[0]))
            func += np.sum(np.square(x - 1))

            return func

        @classmethod
        def gradients(cls, x):
            grad = np.zeros_like(x)

            grad += 16 * (x ** 2 - x[0]) * x
            grad[0] += -8 * np.sum(x ** 2 - x[0])

            grad += 2 * (x - 1)

            return grad

        @classmethod
        def gen_x0(cls, n):
            return 4 * np.ones(shape=(n, ))

    class MCCORMCK(Problem):
        _name = "MCCORMCK"

        @classmethod
        def function(cls, x):
            func = np.sum(-1.5 * x[:-1] + 2.5 * x[1:] + 1)
            func += np.sum(np.square(x[:-1] - x[1:]))
            func += np.sum(np.sin(x[:-1] + x[1:]))
            return func

        @classmethod
        def gradients(cls, x):
            grad = np.zeros_like(x)

            grad[:-1] += -1.5 + 2 * (x[:-1] - x[1:]) + np.cos(x[:-1] + x[1:])
            grad[1:] += 2.5 - 2 * (x[:-1] - x[1:]) + np.cos(x[:-1] + x[1:])

            return grad

        @classmethod
        def gen_x0(cls, n):
            return np.ones(shape=(n, ))  # TODO: not work well!

    class NONDIA(Problem):
        _name = "NONDIA"

        @classmethod
        def function(cls, x):
            func = (x[0] - 1) ** 2
            func += np.sum(np.square(100 * x[0] - x ** 2))

            return func

        @classmethod
        def gradients(cls, x):
            grad = np.zeros_like(x)

            grad[0] += 2 * (x[0] - 1)

            grad[0] += 200 * np.sum(100 * x[0] - x ** 2)
            grad += -4 * (100 * x[0] - x ** 2) * x

            return grad

        @classmethod
        def gen_x0(cls, n):
            return -np.ones(shape=(n, ))

    class NONDQUAR(Problem):
        _name = "NONDQUAR"

        @classmethod
        def function(cls, x):
            n = x.__len__()
            if n < 3:
                raise ValueError("The length of argument `x` should be greater than 2.")
            return (x[0] - x[1]) ** 2 + (x[-2] - x[-1]) ** 2 + np.sum((x[:-2] + x[1:-1] + x[2:]) ** 4)

        @classmethod
        def gradients(cls, x):
            n = x.__len__()
            if n < 3:
                raise ValueError("The length of argument `x` should be greater than 2.")
            grad = np.zeros_like(x)

            grad[0] += 2 * (x[0] - x[1])
            grad[1] += -2 * (x[0] - x[1])
            grad[-2] += 2 * (x[-2] - x[-1])
            grad[-1] += -2 * (x[-2] - x[-1])

            grad[:-2] += 4 * (x[:-2] + x[1:-1] + x[2:]) ** 3
            grad[1:-1] += 4 * (x[:-2] + x[1:-1] + x[2:]) ** 3
            grad[2:] += 4 * (x[:-2] + x[1:-1] + x[2:]) ** 3
            return grad

        @classmethod
        def gen_x0(cls, n):
            x0 = np.ones(shape=(n, ))
            x0[1::2] *= -1.
            return x0

    class NONSCOMP(Problem):
        _name = "NONSCOMP"

        @classmethod
        def function(cls, x):
            func = (x[0] - 1) ** 2
            func += 4 * np.sum(np.square(x[1:] - x[:-1] ** 2))
            return func

        @classmethod
        def gradients(cls, x):
            grad = np.zeros_like(x)

            grad[0] += 2 * (x[0] - 1)

            grad[1:] += 8 * (x[1:] - x[:-1] ** 2)
            grad[:-1] += -16 * (x[1:] - x[:-1] ** 2) * x[:-1]
            return grad

        @classmethod
        def gen_x0(cls, n):
            return 3 * np.ones(shape=(n, ))

    class POWELLSG(Problem):
        _name = "POWELLSG"

        @classmethod
        def function(cls, x):
            x0 = x[::4]  # x_{x(i-1) + 1}, i = 1, 2, 3, ...
            x1 = x[1::4]  # x_{x(i-1) + 2}, i = 1, 2, 3, ...
            x2 = x[2::4]  # x_{x(i-1) + 3}, i = 1, 2, 3, ...
            x3 = x[3::4]  # x_{x(i-1) + 4}, i = 1, 2, 3, ...

            func = np.sum(np.square(x0 + 10 * x1))
            func += 5 * np.sum(np.square(x2 - x3))
            func += np.sum(np.power(x1 - 2 * x2, 4))
            func += 10 * np.sum(np.power(x0 - x3, 4))
            return func

        @classmethod
        def gradients(cls, x):
            x0 = x[::4]  # x_{x(i-1) + 1}, i = 1, 2, 3, ...
            x1 = x[1::4]  # x_{x(i-1) + 2}, i = 1, 2, 3, ...
            x2 = x[2::4]  # x_{x(i-1) + 3}, i = 1, 2, 3, ...
            x3 = x[3::4]  # x_{x(i-1) + 4}, i = 1, 2, 3, ...

            grad = np.zeros_like(x)

            grad[::4] += 2 * (x0 + 10 * x1)
            grad[1::4] += 20 * (x0 + 10 * x1)

            grad[2::4] += 10 * (x2 - x3)
            grad[3::4] += -10 * (x2 - x3)

            grad[1::4] += 4 * np.power(x1 - 2 * x2, 3)
            grad[2::4] += -8 * np.power(x1 - 2 * x2, 3)

            grad[::4] += 40 * np.power(x0 - x3, 3)
            grad[3::4] += -40 * np.power(x0 - x3, 3)

            return grad

        @classmethod
        def gen_x0(cls, n):
            x0 = np.zeros(shape=(n, ))
            x0[::4] += 3.0
            x0[1::4] += -1.0
            x0[2::4] += 0.0
            x0[3::4] += 1.0

            return x0

    class SCHMVETT(Problem):
        _name = "SCHMVETT"

        @classmethod
        def function(cls, x):
            func = - np.sum(1 / (1 + np.square(x[:-2] - x[1:-1])))

            func += -np.sum(np.sin(0.5 * np.pi * x[1:-1] + 0.5 * x[2:]))

            func += -np.sum(np.exp(-np.square((x[:-2] + x[2:]) / x[1:-1] - 2)))

            return func

        @classmethod
        def gradients(cls, x):
            grad = np.zeros_like(x)

            grad[:-2] += 2 * (x[:-2] - x[1:-1]) / np.square(1 + np.square(x[:-2] - x[1:-1]))
            grad[1:-1] += -2 * (x[:-2] - x[1:-1]) / np.square(1 + np.square(x[:-2] - x[1:-1]))

            grad[1:-1] += -0.5 * np.pi * np.cos(0.5 * np.pi * x[1:-1] + 0.5 * x[2:])
            grad[2:] += -0.5 * np.cos(0.5 * np.pi * x[1:-1] + 0.5 * x[2:])

            temp = 2 * np.exp(-np.square((x[:-2] + x[2:]) / x[1:-1] - 2)) * ((x[:-2] + x[2:]) / x[1:-1] - 2)
            grad[:-2] += temp / x[1:-1]
            grad[1:-1] += -temp * (x[:-2] + x[2:]) / x[1:-1] ** 2
            grad[2:] += temp / x[1:-1]

            return grad

        @classmethod
        def gen_x0(cls, n):
            return 3 * np.ones(shape=(n, ))

    class SINQUAD(Problem):
        _name = "SINQUAD"

        @classmethod
        def function(cls, x):
            func = (x[0] - 1) ** 4 + (x[-1] ** 2 - x[0] ** 2) ** 2
            func += np.sum(np.square(np.sin(x[1:-1] - x[-1]) - x[0] ** 2 + x[1:-1] ** 2))
            return func

        @classmethod
        def gradients(cls, x):
            grad = np.zeros_like(x)

            grad[0] += 4 * (x[0] - 1) ** 3

            grad[-1] += 4 * (x[-1] ** 2 - x[0] ** 2) * x[-1]
            grad[0] += -4 * (x[-1] ** 2 - x[0] ** 2) * x[0]

            grad[1:-1] += 2 * (np.sin(x[1:-1] - x[-1]) - x[0] ** 2 + x[1:-1] ** 2) * np.cos(x[1:-1] - x[-1])
            grad[-1] += np.sum(-2 * (np.sin(x[1:-1] - x[-1]) - x[0] ** 2 + x[1:-1] ** 2) * np.cos(x[1:-1] - x[-1]))
            grad[0] += np.sum(-4 * (np.sin(x[1:-1] - x[-1]) - x[0] ** 2 + x[1:-1] ** 2) * x[0])
            grad[1:-1] += 4 * (np.sin(x[1:-1] - x[-1]) - x[0] ** 2 + x[1:-1] ** 2) * x[1:-1]

            return grad

        @classmethod
        def gen_x0(cls, n):
            return 0.1 * np.ones(shape=(n, ))

    class SROSENBR(Problem):
        _name = "SROSENBR"

        @classmethod
        def function(cls, x):
            n = x.__len__()
            if n % 2 != 0:
                raise ValueError("The length of argument `x` should be twice as much as an integer.")

            func = 100 * np.sum(np.square(x[1::2] - x[::2] ** 2))
            func += np.sum(np.square(x[::2] - 1))

            return func

        @classmethod
        def gradients(cls, x):
            n = x.__len__()
            if n % 2 != 0:
                raise ValueError("The length of argument `x` should be twice as much as an integer.")

            grad = np.zeros_like(x)

            grad[1::2] += 200 * (x[1::2] - x[::2] ** 2)
            grad[::2] += -400 * (x[1::2] - x[::2] ** 2) * x[::2]

            grad[::2] += 2 * (x[::2] - 1)

            return grad

        @classmethod
        def gen_x0(cls, n):
            x0 = np.zeros(shape=(n, ))
            x0[1::2] += 1.0
            x0[0::2] += -1.2
            return x0

    class TOINTGSS(Problem):
        _name = "TOINTGSS"

        @classmethod
        def function(cls, x):
            n = x.__len__()
            if n < 3:
                raise ValueError("The length of argument `x` should be greater than 2.")

            return np.sum((10 / (n + 2) + x[2:] ** 2) * (2 - np.exp(-(x[:-2] - x[1:-1]) ** 2 / (0.1 + x[2:] ** 2))))

        @classmethod
        def gradients(cls, x):
            n = x.__len__()
            if n < 3:
                raise ValueError("The length of argument `x` should be greater than 2.")
            grad = np.zeros_like(x)

            grad[2:] += 2 * x[2:] * (2 - np.exp(-(x[:-2] - x[1:-1]) ** 2 / (0.1 + x[2:] ** 2)))

            temp = (10 / (n + 2) + x[2:] ** 2) * np.exp(-(x[:-2] - x[1:-1]) ** 2 / (0.1 + x[2:] ** 2))
            grad[:-2] += temp * 2 * (x[:-2] - x[1:-1]) / (0.1 + x[2:] ** 2)
            grad[1:-1] += temp * (-2) * (x[:-2] - x[1:-1]) / (0.1 + x[2:] ** 2)
            grad[2:] += temp * (-1) * (x[:-2] - x[1:-1]) ** 2 / (0.1 + x[2:] ** 2) ** 2 * 2 * x[2:]

            return grad

        @classmethod
        def gen_x0(cls, n):
            return 3.0 * np.ones(shape=(n, ))

    class TQUARTIC(Problem):
        _name = "TQUARTIC"

        @classmethod
        def function(cls, x):
            return (x[0] - 1) ** 2 + np.sum(np.square(x[0] ** 2 - x[1:] ** 2))

        @classmethod
        def gradients(cls, x):
            grad = np.zeros_like(x)

            grad[0] += 2 * (x[0] - 1)
            grad[0] += 4 * np.sum(x[0] ** 2 - x[1:] ** 2) * x[0]

            grad[1:] += -4 * (x[0] ** 2 - x[1:] ** 2) * x[1:]
            return grad

        @classmethod
        def gen_x0(cls, n):
            return 0.1 * np.ones(shape=(n, ))

    class WOODS(Problem):
        _name = "WOODS"

        @classmethod
        def function(cls, x):
            x3 = x[::4]  # x_{4i-3}, i = 1, 2, 3, ...
            x2 = x[1::4]  # x_{4i-2}, i = 1, 2, 3, ...
            x1 = x[2::4]  # x_{4i-1}, i = 1, 2, 3, ...
            x0 = x[3::4]  # x_{4i}, i = 1, 2, 3, ...

            func = 100 * np.sum(np.square(x2 - x3 ** 2))
            func += np.sum(np.square(1 - x3))
            func += 90 * np.sum(np.square(x0 - x1 ** 2))
            func += np.sum(np.square(1 - x1))
            func += 10 * np.sum(np.square(x2 + x0 - 2))
            func += 0.1 * np.sum(np.square(x2 + x0))
            return func

        @classmethod
        def gradients(cls, x):
            x3 = x[::4]  # x_{4i-3}, i = 1, 2, 3, ...
            x2 = x[1::4]  # x_{4i-2}, i = 1, 2, 3, ...
            x1 = x[2::4]  # x_{4i-1}, i = 1, 2, 3, ...
            x0 = x[3::4]  # x_{4i}, i = 1, 2, 3, ...

            grad = np.zeros_like(x)

            grad[1::4] += 200 * (x2 - x3 ** 2)
            grad[::4] += -400 * (x2 - x3 ** 2) * x3

            grad[::4] += -2 * (1 - x3)

            grad[3::4] += 180 * (x0 - x1 ** 2)
            grad[2::4] += -360 * (x0 - x1 ** 2) * x1

            grad[2::4] += -2 * (1 - x1)

            grad[1::4] += 20 * (x2 + x0 - 2)
            grad[3::4] += 20 * (x2 + x0 - 2)

            grad[1::4] += 0.2 * (x2 + x0)
            grad[3::4] += 0.2 * (x2 + x0)
            return grad

        @classmethod
        def gen_x0(cls, n):
            x0 = np.zeros(shape=(n, ))
            x0[::2] += -3.0
            x0[1::2] += -1.0
            return x0


def unit_test():
    """
    arguments:
        m = 8
        acculate = 8
        secant = True
    termination_criterion: ||gk|| < 1e-5

    +--------------------------------------------------------------------------+
    |          |      | GD    | BFGS  | L-BFGS | secant=False  |  secant=True  |
    |          |      |       |       |        +-------+-------+-------+-------+
    | name     | n    |       |       |        | ver-A | ver-B | ver-A | ver-B |
    +----------+------+-------+-------+--------+-------+-------+-------+-------+
    | ARWHEAD  | 1024 | >1000 | 39    | 26     | 21    | 16    | 21    | 16    |
    | BDQRTIC  | 1024 | >1000 | --    | --     | 491   | 317   | 516   | 315   |
    | BDEXP    | 1024 | >1000 | 19    | 19     | 9     | 9     | 9     | 9     |
    | COSINE   | 1024 | >1000 | --    | --     | 44    | 16    | 43    | 16    |
    | DIXMAANE | 1500 | >1000 | 195   | 244    | 586   | 326   | 597   | 312   |
    | DIXMAANF | 1500 | >1000 | 336   | 216    | 423   | 265   | 391   | 206   |
    | DIXMAANG | 1500 | >1000 | 954   | 384    | 460   | 211   | 414   | 206   |
    | DQRTIC   | 1000 | --    | --    | --     | 35    | 31    | 35    | 31    |
    | EDENSCH  | 1000 | 59    | 86    | 52     | 42    | 23    | 42    | 23    |
    | ENGVAL1  | 1000 | 66    | 154   | 119    | 39    | 24    | 38    | 24    |
    | EG2      | 1000 | 7     | 6     | 6      | 8     | 8     | 8     | 8     |
    | EXTROSNB | 1000 | 63    | 309   | 333    | 76    | 41    | 73    | 43    |
    | FLETCHER | 100  | >1000 | --    | --     | >1000 | 734   | >1000 | 752   |
    | FREUROTH | 1000 | --    | --    | --     | 51    | 45    | 52    | --    |
    | GENROSE  | 1000 | >1000 | >1000 | 39     | 48    | --    | 50    | --    |
    | HIMMELBG | 1000 | >1000 | 3     | 3      | 3     | 3     | 3     | 3     |
    | HIMMELH  | 1000 | 20    | 9     | 9      | 19    | 16    | 19    | 16    |
    | LIARWHD  | 1000 | >1000 | --    | 28     | 40    | 30    | 40    | 30    |
    | NONDIA   | 1000 | >1000 | --    | 55     | 97    | 76    | 89    | --    |
    | NONDQUAR | 1000 | >1000 | 270   | 320    | 344   | 230   | 296   | 260   |
    | NONSCOMP | 1000 | 86    | 286   | 238    | 101   | 45    | 102   | 46    |
    | POWELLSG | 1000 | >1000 | 459   | 49     | 69    | 63    | 64    | 64    |
    | SCHMVETT | 1000 | 181   | 26    | 24     | 45    | 25    | 53    | 24    |
    | SINQUAD  | 1000 | >1000 | 140   | 143    | --    | --    | 109   | --    |
    | SROSENBR | 1000 | >1000 | --    | 39     | 48    | --    | 48    | --    |
    | TOINTGSS | 1000 | 6     | 9     | 9      | 8     | 7     | 8     | 7     |
    | TQUARTIC | 1000 | >1000 | 16    | 17     | 28    | 24    | 28    | 24    |
    | WOODS    | 1000 | >1000 | --    | 92     | --    | 48    | --    | --    |
    +--------------------------------------------------------------------------+
    """
    print('\n' + '#' * 32 + "\tTEST ALL METHODS FOR CUTE PROBLEMS\t" + '#' * 32)
    kwargs = {
        "GD": True, "BFGS": True, "L_BFGS": True,
        "F_BFGS": True, "m": 8, "secant": [False, True], "version": ["A", "B"]
    }

    CUTE.ARWHEAD.solve(n=1024, **kwargs)
    CUTE.BDQRTIC.solve(n=1024, **kwargs)
    CUTE.BDEXP.solve(n=1024, **kwargs)
    CUTE.COSINE.solve(n=1024, **kwargs)
    CUTE.DIXMAANE.solve(n=1500, **kwargs)
    CUTE.DIXMAANF.solve(n=1500, **kwargs)
    CUTE.DIXMAANG.solve(n=1500, **kwargs)
    CUTE.DQRTIC.solve(n=1000, **kwargs)
    CUTE.EDENSCH.solve(n=1000, **kwargs)
    CUTE.ENGVAL1.solve(n=1000, **kwargs)
    CUTE.EG2.solve(n=1000, **kwargs)
    CUTE.EXTROSNB.solve(n=1000, **kwargs)
    CUTE.FLETCHER.solve(n=100, **kwargs)
    CUTE.FREUROTH.solve(n=1000, **kwargs)
    CUTE.GENROSE.solve(n=1000, **kwargs)
    CUTE.HIMMELBG.solve(n=1000, **kwargs)
    CUTE.HIMMELH.solve(n=1000, **kwargs)
    CUTE.LIARWHD.solve(n=1000, **kwargs)
    CUTE.NONDIA.solve(n=1000, **kwargs)
    CUTE.NONDQUAR.solve(n=1000, **kwargs)
    CUTE.NONSCOMP.solve(n=1000, **kwargs)
    CUTE.POWELLSG.solve(1000, **kwargs)
    CUTE.SCHMVETT.solve(n=1000, **kwargs)
    CUTE.SINQUAD.solve(n=1000, **kwargs)
    CUTE.SROSENBR.solve(n=1000, **kwargs)
    CUTE.TOINTGSS.solve(n=1000, **kwargs)
    CUTE.TQUARTIC.solve(n=1000, **kwargs)
    CUTE.WOODS.solve(n=1000, **kwargs)


def display(cls, n, GD=True, BFGS=True, L_BFGS=True, F_BFGS=True, m=8, secant=False, version="A"):
    def format(_gk, _k):
        if _gk < 1e-5:
            if _k <= 1000:
                return str(_k)
            else:
                return ">1000"
        elif _gk < 1:
            return ">1000"
        else:
            return "--"

    message = ""

    # ---------------- generate x0 ----------------
    x0 = cls.gen_x0(n)

    # ---------------- gradient descent ----------------
    if GD:
        _, log = gradient_descent(cls.function, cls.gradients, x0=x0)
        _, gk, k = log[-1, -2], log[-1, -1], log.__len__()
        message += " | " + format(gk, k)

    # ---------------- BFGS ----------------
    if BFGS:
        _, log = bfgs(cls.function, cls.gradients, x0=x0)
        _, gk, k = log[-1, -2], log[-1, -1], log.__len__()
        message += " | " + format(gk, k)

    # ---------------- L-BFGS ----------------
    if L_BFGS:
        _, log = l_bfgs(cls.function, cls.gradients, m, x0)
        _, gk, k = log[-1, -2], log[-1, -1], log.__len__()
        message += " | " + format(gk, k)

    # ---------------- P-BFGS ----------------
    if F_BFGS:
        m = m if isinstance(m, list) else [m]
        secant = secant if isinstance(secant, list) else [secant]
        version = version if isinstance(version, list) else version
        for arg1 in m:
            for arg2 in secant:
                for arg3 in version:
                    _, log = fast_bfgs(cls.function, cls.gradients, x0, m=arg1, secant=arg2, version=arg3)
                    fk, gk, k = log[-1, -2], log[-1, -1], log.__len__()
                    message += " | " + format(gk, k)

    print("| " + cls._name + " | " + str(n) + message + " |")


def memory_experiments():
    """
    arguments:
        secant = True
    termination_criterion: ||gk|| < 1e-5

    +-----------------------------------------------------------------------------------------------------------------+
    |          |      |             m = 2             |             m = 4             |             m = 8             |
    |          |      +-------------------------------+-------------------------------+-------------------------------+
    |          |      | secant=False  |  secant=True  | secant=False  |  secant=True  | secant=False  |  secant=True  |
    |          |      +---------------+---------------+---------------+---------------+---------------+---------------+
    | name     | n    | ver-A | ver-B | ver-A | ver-B | ver-A | ver-B | ver-A | ver-B | ver-A | ver-B | ver-A | ver-B |
    |----------+------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+
    | ARWHEAD  | 1024 | 21    | 16    | 21    | 16    | 21    | 16    | 21    | 16    | 21    | 16    | 21    | 16    |
    | BDQRTIC  | 1024 | >1000 | >1000 | >1000 | >1000 | >1000 | 427   | 874   | 604   | 491   | 317   | 516   | 315   |
    | BDEXP    | 1024 | 9     | 9     | 9     | 9     | 9     | 9     | 9     | 9     | 9     | 9     | 9     | 9     |
    | COSINE   | 1024 | 98    | 17    | 154   | 18    | 63    | 16    | 53    | 17    | 44    | 16    | 43    | 16    |
    | DIXMAANE | 1500 | 586   | 800   | 596   | 352   | 619   | 327   | 597   | 339   | 586   | 326   | 597   | 312   |
    | DIXMAANF | 1500 | 513   | 559   | 501   | 330   | 481   | 252   | 544   | 275   | 423   | 265   | 391   | 206   |
    | DIXMAANG | 1500 | 520   | 233   | 548   | 352   | 457   | 268   | 517   | 265   | 460   | 211   | 414   | 206   |
    | DQRTIC   | 1000 | 36    | 32    | 36    | 32    | 35    | 31    | 35    | 31    | 35    | 31    | 35    | 31    |
    | EDENSCH  | 1000 | 49    | 31    | 50    | 28    | 46    | 28    | 52    | 28    | 42    | 23    | 42    | 23    |
    | ENGVAL1  | 1000 | 55    | 30    | 55    | 28    | 44    | 26    | 45    | 28    | 39    | 24    | 38    | 24    |
    | EG2      | 1000 | >1000 | >1000 | 8     | 8     | 8     | 7     | 8     | >1000 | 8     | 8     | 8     | 8     |
    | EXTROSNB | 1000 | 77    | 44    | 78    | 43    | 77    | 42    | 78    | 43    | 76    | 41    | 73    | 43    |
    | FLETCHER | 100  | >1000 | >1000 | >1000 | >1000 | --    | 791   | >1000 | 938   | >1000 | 734   | >1000 | 752   |
    | FREUROTH | 1000 | 248   | 70    | 261   | --    | 65    | --    | 71    | --    | 51    | 45    | 52    | --    |
    | GENROSE  | 1000 | 48    | 55    | 48    | 53    | 50    | --    | >1000 | 79    | 48    | --    | 50    | --    |
    | HIMMELBG | 1000 | 3     | 3     | 3     | 3     | 3     | 3     | 3     | 3     | 3     | 3     | 3     | 3     |
    | HIMMELH  | 1000 | 19    | 16    | 19    | 16    | 19    | 16    | 19    | 16    | 19    | 16    | 19    | 16    |
    | LIARWHD  | 1000 | 40    | 30    | 40    | 30    | 39    | 30    | 40    | 30    | 40    | 30    | 40    | 30    |
    | NONDIA   | 1000 | 90    | 74    | 86    | 74    | 93    | 74    | --    | 71    | 97    | 76    | 89    | --    |
    | NONDQUAR | 1000 | 795   | 953   | 790   | 287   | 571   | 278   | 413   | 269   | 344   | 230   | 296   | 260   |
    | NONSCOMP | 1000 | 108   | 58    | 105   | 53    | 107   | 49    | 104   | 47    | 101   | 45    | 102   | 46    |
    | POWELLSG | 1000 | >1000 | 497   | 299   | 436   | 66    | 63    | 64    | 63    | 69    | 63    | 64    | 64    |
    | SCHMVETT | 1000 | 88    | 52    | 89    | 46    | 79    | 24    | 82    | 38    | 45    | 25    | 53    | 24    |
    | SINQUAD  | 1000 | >1000 | 316   | >1000 | --    | --    | >1000 | 218   | 180   | --    | --    | >1000 | --    |
    | SROSENBR | 1000 | 48    | 70    | 48    | 62    | 48    | 85    | 48    | 58    | 48    | --    | 48    | --    |
    | TOINTGSS | 1000 | 10    | 7     | 10    | 7     | 8     | 7     | 8     | 7     | 8     | 7     | 8     | 7     |
    | TQUARTIC | 1000 | 28    | 23    | 28    | 23    | 28    | 23    | 28    | 23    | 28    | 24    | 28    | 24    |
    | WOODS    | 1000 | 638   | 254   | 498   | 180   | 48    | 48    | 48    | 48    | --    | 48    | >1000 | --    |
    +-----------------------------------------------------------------------------------------------------------------+
    """
    print('\n' + '#' * 32 + "\tTEST P-BFGS(MEMORY) FOR CUTE PROBLEMS\t" + '#' * 32)
    kwargs = {
        "GD": False, "BFGS": False, "L_BFGS": False, "F_BFGS": True,
        "m": [2, 4, 8], "secant": [False, True], "version": ["A", "B"]
    }

    display(cls=CUTE.ARWHEAD, n=1024, **kwargs)
    display(cls=CUTE.BDQRTIC, n=1024, **kwargs)
    display(cls=CUTE.BDEXP, n=1024, **kwargs)
    display(cls=CUTE.COSINE, n=1024, **kwargs)
    display(cls=CUTE.DIXMAANE, n=1500, **kwargs)
    display(cls=CUTE.DIXMAANF, n=1500, **kwargs)
    display(cls=CUTE.DIXMAANG, n=1500, **kwargs)
    display(cls=CUTE.DQRTIC, n=1000, **kwargs)
    display(cls=CUTE.EDENSCH, n=1000, **kwargs)
    display(cls=CUTE.ENGVAL1, n=1000, **kwargs)
    display(cls=CUTE.EG2, n=1000, **kwargs)
    display(cls=CUTE.EXTROSNB, n=1000, **kwargs)
    display(cls=CUTE.FLETCHER, n=100, **kwargs)
    display(cls=CUTE.FREUROTH, n=1000, **kwargs)
    display(cls=CUTE.GENROSE, n=1000, **kwargs)
    display(cls=CUTE.HIMMELBG, n=1000, **kwargs)
    display(cls=CUTE.HIMMELH, n=1000, **kwargs)
    display(cls=CUTE.LIARWHD, n=1000, **kwargs)
    display(cls=CUTE.NONDIA, n=1000, **kwargs)
    display(cls=CUTE.NONDQUAR, n=1000, **kwargs)
    display(cls=CUTE.NONSCOMP, n=1000, **kwargs)
    display(cls=CUTE.POWELLSG, n=1000, **kwargs)
    display(cls=CUTE.SCHMVETT, n=1000, **kwargs)
    display(cls=CUTE.SINQUAD, n=1000, **kwargs)
    display(cls=CUTE.SROSENBR, n=1000, **kwargs)
    display(cls=CUTE.TOINTGSS, n=1000, **kwargs)
    display(cls=CUTE.TQUARTIC, n=1000, **kwargs)
    display(cls=CUTE.WOODS, n=1000, **kwargs)


unit_test()
memory_experiments()
