import numpy as np


#  --------------------------- line-search ---------------------------
def wolfe(func, grad, xk, alpha, pk):
    c1 = 1e-4
    return func(xk + alpha * pk) <= func(xk) + c1 * alpha * np.dot(grad(xk), pk)


def strong_wolfe(func, grad, xk, alpha, pk, c2):
    return wolfe(func, grad, xk, alpha, pk) and abs(
        np.dot(grad(xk + alpha * pk), pk)) <= c2 * abs(np.dot(grad(xk), pk))


def step_length(func, grad, xk, pk, alpha=1., c2=0.9, max_iters=20):
    f_alpha = lambda alpha: func(xk + alpha * pk)
    g_alpha = lambda alpha: np.dot(grad(xk + alpha * pk), pk)

    l = 0.0
    h = 1.0
    for i in range(max_iters):
        if strong_wolfe(func, grad, xk, alpha, pk, c2):
            return alpha

        half = (l + h) / 2
        alpha = - g_alpha(l) * (h ** 2) / (2 * (f_alpha(h) - f_alpha(l) - g_alpha(l) * h))
        if alpha < l or alpha > h:
            alpha = half
        if g_alpha(alpha) > 0:
            h = alpha
        elif g_alpha(alpha) <= 0:
            l = alpha

    return alpha


#  --------------------------- termination criterion ---------------------------
def termination_criterion(termination_version, grad, xk, xk1, tol):
    if termination_version == 0:
        return np.linalg.norm(grad(xk1)) < tol
    elif termination_version == 1:
        return np.linalg.norm(grad(xk1)) / np.linalg.norm(grad(xk)) < tol


TERMINATION_VERSION = 0  # TODO: Don't change this argument if not necessary!


#  --------------------------- gradient descent ---------------------------
def gradient_descent(func, grad, x0, tol=1e-5, max_iters=1000):
    k = 0
    xk = x0
    gk = grad(xk)

    logs = [[*xk, func(xk), np.linalg.norm(gk)]]
    while True:
        tau = step_length(func, grad, xk, -gk, alpha=1.0, c2=0.9)
        sk = -tau * gk
        xk1 = xk + sk

        if termination_criterion(TERMINATION_VERSION, grad, xk, xk1, tol):
            logs.append([*xk1, func(xk1), np.linalg.norm(grad(xk1))])
            break

        if k >= max_iters:
            break

        xk = xk1
        gk = grad(xk)
        k += 1
        logs.append([*xk, func(xk), np.linalg.norm(gk)])

    return xk, np.array(logs)


#  --------------------------- newton ---------------------------
def newton(func, grad, hess, x0, tol=1e-5, max_iters=1000):
    k = 0
    xk = x0
    gk = grad(xk)

    logs = [[*xk, func(xk), np.linalg.norm(gk)]]
    while True:
        hk = np.linalg.inv(hess(xk))
        tau = step_length(func, grad, xk, -hk@gk, alpha=1.0, c2=0.9)
        sk = -tau * hk@gk
        xk1 = xk + sk
        gk1 = grad(xk1)

        if termination_criterion(TERMINATION_VERSION, grad, xk, xk1, tol):
            logs.append([*xk1, func(xk1), np.linalg.norm(gk1)])
            break

        if k >= max_iters:
            break

        xk = xk1
        gk = gk1
        k += 1
        logs.append([*xk, func(xk), np.linalg.norm(gk)])

    return xk, np.array(logs)


#  --------------------------- bfgs ---------------------------
def bfgs(func, grad, x0, tol=1e-5, max_iters=1000):
    k = 0
    xk = x0
    gk = grad(xk)
    hk = e = np.eye(xk.__len__())

    logs = [[*xk, func(xk), np.linalg.norm(gk)]]
    while True:
        pk = -hk @ gk
        tau = step_length(func, grad, xk, alpha=1.0, pk=pk, c2=0.9)
        sk = tau * pk
        xk1 = xk + sk

        gk1 = grad(xk1)
        yk = gk1 - gk

        if termination_criterion(TERMINATION_VERSION, grad, xk, xk1, tol):
            logs.append([*xk1, func(xk1), np.linalg.norm(gk1)])
            break
        if np.linalg.norm(yk) == 0:
            logs = logs[:-1]
            break
        if k >= max_iters:
            break

        xk = xk1
        gk = gk1
        temp = e - np.outer(sk, yk) / np.inner(sk, yk)
        hk = temp @ hk @ temp.T + np.outer(sk, sk) / np.inner(sk, yk)
        k += 1
        logs.append([*xk, func(xk), np.linalg.norm(gk)])

    return xk, np.array(logs)


#  --------------------------- l-bfgs ---------------------------
def two_loop_recursion(sks, yks, H0, q):
    m_t = len(sks)
    a = np.zeros(m_t)
    b = np.zeros(m_t)
    # print('global_var = ', q)
    for i in reversed(range(m_t)):
        s = sks[i]
        y = yks[i]
        rho_i = float(1.0 / y.T.dot(s))
        a[i] = rho_i * s.dot(q)
        q = q - a[i] * y

    r = H0.dot(q)

    for i in range(m_t):
        s = sks[i]
        y = yks[i]
        rho_i = float(1.0 / y.T.dot(s))
        b[i] = rho_i * y.dot(r)
        r = r + s * (a[i] - b[i])

    return -r


def l_bfgs(func, grad, m, x0, tol=1e-5, max_iters=1000):
    k = 0
    xk = x0
    gk = grad(xk)
    I = np.identity(xk.size)

    sks = []
    yks = []

    logs = [[*xk, func(xk), np.linalg.norm(gk)]]
    while True:
        # compute search direction
        pk = two_loop_recursion(sks, yks, I, gk)

        # obtain step length by line search
        tau = step_length(func, grad, xk, alpha=1.0, pk=pk, c2=0.9)

        # update x
        xk1 = xk + tau * pk
        gk1 = grad(xk1)

        # define sk and yk for convenience
        sk = xk1 - xk
        yk = gk1 - gk

        if termination_criterion(TERMINATION_VERSION, grad, xk, xk1, tol):
            logs.append([*xk1, func(xk1), np.linalg.norm(gk1)])
            break
        if np.linalg.norm(yk) == 0:
            logs = logs[:-1]
            break
        if k >= max_iters:
            break

        xk = xk1
        gk = gk1
        sks.append(sk)
        yks.append(yk)
        if len(sks) > m:
            sks = sks[1:]
            yks = yks[1:]
        k += 1
        logs.append([*xk, func(xk), np.linalg.norm(gk)])

    return xk, np.array(logs)


#  --------------------------- fast-bfgs ---------------------------
def update_lk(m, yk, sks, lk, secant=False):
    """
    lk <- tk @ lk @ tk.T + [[0, 0], [0, 1]].
    # ----------------------- note 1 -----------------------
    tk = \begin{bmatrix}
        t_1                 & 1                & \cdots & 0              \\
        \vdots              & \vdots           & \ddots & \vdots         \\
        t_{m-1}             & 0                & \cdots & 1              \\
        t_m - y_k^T s_{k-m} & -y_k^T s_{k+1-m} & \cdots & -y_k^T s_{k-1}
    \begin{bmatrix}
    # ----------------------- note 2 -----------------------
    \begin{bmatrix} t_1 \\ \vdots \\ t_m \end{bmatrix}
    = \mathop{\arg\min}_t \| \tilde{S}_{k+1} t - \tilde{\boldsymbol{s}}_{k-m} \|_2
    subject to
    \tilde{\boldsymbol{y}}_k^T \tilde{S}_{k+1} t = \tilde{\boldsymbol{y}}_k^T \tilde{\boldsymbol{s}}_{k-m}
    """
    k = sks.__len__() - 1
    n = yk.__len__()

    if k < m:
        ys = np.array([[-np.inner(yk, sk) for sk in sks[:-1]]])
        tk = np.vstack((np.eye(k), ys))
    else:
        tk = np.zeros(shape=(m, m))
        tk[:-1, 1:] = np.eye(m - 1)
        tk[-1] = np.array([-np.inner(yk, sk) for sk in sks[:-1]])

        if m >= n:
            mat = np.array(sks[-n:]).T
            rhs = sks[0]

            try:
                c = np.linalg.solve(mat, rhs)
            except np.linalg.LinAlgError:
                # TODO: `1e-6` is set based on experimental results.
                c = np.linalg.solve(mat + 1e-6 * np.eye(rhs.__len__()), rhs)
            tk[-n:, 0] += c
        else:
            if secant:
                mat = np.zeros(shape=(m+1, m+1))
                mat[:-1, :-1] = 2 * np.array(sks[1:]) @ np.array(sks[1:]).T
                mat[:-1, -1] = np.array(sks[1:]) @ yk
                mat[-1, :-1] = np.array(sks[1:]) @ yk
                rhs = np.zeros(shape=(m+1, ))
                rhs[:-1] = 2 * np.array(sks[1:]) @ sks[0]
                rhs[-1] = np.inner(yk, sks[0])

                try:
                    c = np.linalg.solve(mat, rhs)
                except np.linalg.LinAlgError:
                    c = np.linalg.solve(mat + 1e-6 * np.eye(rhs.__len__()), rhs)
                tk[:, 0] += c[:-1]
            else:
                mat = np.array(sks[1:]) @ np.array(sks[1:]).T
                rhs = np.array(sks[1:]) @ sks[0]

                try:
                    c = np.linalg.solve(mat, rhs)
                except np.linalg.LinAlgError:
                    c = np.linalg.solve(mat + 1e-6 * np.eye(rhs.__len__()), rhs)
                tk[:, 0] += c

    lk = tk @ lk @ tk.T
    lk[-1, -1] += 1
    return lk


def estimate_vk(grad, xk, gk, pk, w1=0.5, w2=0.5):
    v1 = gk

    # TODO: `1e-6` is set based on experimental results.
    eps = 1e-6 / np.linalg.norm(gk)
    bg = (grad(xk + eps * gk) - gk) / eps

    if np.linalg.norm(pk) > 0:
        eps = 1e-6 / np.linalg.norm(pk)
        bp = (grad(xk + eps * pk) - gk) / eps
        eps = 1e-6 / np.linalg.norm(bp)
        b2p = (grad(xk + eps * bp) - gk) / eps
    else:
        b2p = 0

    v2 = bg - b2p

    orth_v1 = v1 - np.inner(v1, v2) / np.inner(v2, v2) * v2
    orth_v2 = v2 - np.inner(v1, v2) / np.inner(v1, v1) * v1

    if np.linalg.norm(orth_v1) > 0 and np.linalg.norm(orth_v2) > 0:
        vk = w1 / np.linalg.norm(orth_v1) * orth_v1 + w2 / np.linalg.norm(orth_v2) * orth_v2
    elif np.inner(v1, v2) > 0:
        vk = w1 * v1 + w2 * v2
    else:
        raise ValueError

    return vk / np.linalg.norm(vk)


def estimate_alpha(grad, xk, gk, pk, vk):
    if np.linalg.norm(vk) == 0:
        return 0

    # TODO: `1e-6` is set based on experimental results.
    eps = 1e-6 / np.linalg.norm(vk)
    bv = (grad(xk + eps * vk) - gk) / eps

    if np.linalg.norm(pk) > 0:
        eps = 1e-6 / np.linalg.norm(pk)
        bp = (grad(xk + eps * pk) - gk) / eps
    else:
        bp = 0

    return np.inner(gk - bp, bv) / np.inner(bv, bv)


def fix_pk(grad, xk, gk, pk, version="A"):
    if version == "A":
        try:
            vk = estimate_vk(grad, xk, gk, pk)
            alpha = estimate_alpha(grad, xk, gk, pk, vk=vk)
        except ValueError:
            vk = np.zeros_like(xk)
            alpha = 0.
    elif version == "B":
        vk = gk
        alpha = estimate_alpha(grad, xk, gk, pk, vk=gk)
    else:
        raise ValueError("Invalid argument `version`.")

    return pk - alpha * vk


def fast_bfgs(func, grad, x0, m=8, secant=False, version="A", tol=1e-5, max_iters=1000):
    xk = x0
    gk = grad(xk)
    logs = [[*xk, func(xk), np.linalg.norm(gk)]]

    # compute x1, s0, y0
    pk = -gk
    tau = step_length(func, grad, xk, alpha=1.0, pk=pk, c2=0.9)
    sk = tau * pk
    xk1 = xk + sk
    gk1 = grad(xk1)
    yk = gk1 - gk

    # update xk, gk, sks, ak, k
    xk = xk1
    gk = gk1
    sks = [sk / np.sqrt(np.abs(np.inner(sk, yk)))]
    lk = np.array([[1]])
    k = 0
    logs.append([*xk, func(xk), np.linalg.norm(gk)])

    while True:
        # compute pk
        pk = -np.array(sks).T @ (lk @ (np.array(sks) @ gk))

        if m < x0.__len__() or sks.__len__() <= x0.__len__():
            # fix pk
            pk = fix_pk(grad, xk, gk, pk, version=version)

        # compute xk1, sk, yk
        tau = step_length(func, grad, xk, alpha=1.0, pk=pk, c2=0.9)
        sk = tau * pk
        xk1 = xk + sk
        gk1 = grad(xk1)
        yk = gk1 - gk

        if termination_criterion(TERMINATION_VERSION, grad, xk, xk1, tol):
            logs.append([*xk1, func(xk1), np.linalg.norm(gk1)])
            break
        if np.linalg.norm(yk) == 0:
            logs = logs[:-1]
            break
        if k >= max_iters:
            break

        # update xk, gk, sks, yk, ak, k
        xk = xk1
        gk = gk1
        sks.append(sk / np.sqrt(np.abs(np.inner(sk, yk))))
        yk = yk / np.sqrt(np.abs(np.inner(sk, yk)))
        lk = update_lk(m, yk, sks, lk, secant=secant)
        if len(sks) > m:
            sks = sks[1:]
        k += 1

        logs.append([*xk, func(xk), np.linalg.norm(gk)])

    return xk, np.array(logs)


def unit_test():
    print('\n' + '#' * 32 + "\tLIGHTWEIGHT TEST\t" + '#' * 32)

    # Rosenbrock's function
    class Rosenbrock:
        @classmethod
        def function(cls, x):
            return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

        @classmethod
        def gradients(cls, x):
            return np.array([200 * (x[1] - x[0] ** 2) * (-2 * x[0]) + 2 * (x[0] - 1), 200 * (x[1] - x[0] ** 2)])

    # Styblinski & Tang's function(1990)
    class Styblinski:
        @classmethod
        def function(cls, x):
            return np.mean(x ** 4 - 16 * x ** 2 + 5 * x)

        @classmethod
        def gradients(cls, x):
            return (4 * x ** 3 - 32 * x + 5) / x.__len__()

    m = 8

    # Solving Rosenbrock's function.
    print('-' * 32 + "\tSolving {}-dim Rosenbrock's function\t".format(2) + '-' * 32)
    func = Rosenbrock.function
    grad = Rosenbrock.gradients
    x0 = 1.1 * np.ones(shape=(2, ))

    xk, log = gradient_descent(func, grad, x0)
    fk, gk, k = log[-1, -2], log[-1, -1], log.__len__()
    print("gradient descent: fk = {:.2e}, ||gk|| = {:.2e}, iters = {}".format(fk, gk, k))

    xk, log = bfgs(func, grad, x0)
    fk, gk, k = log[-1, -2], log[-1, -1], log.__len__()
    print("bfgs: fk = {:.2e}, ||gk|| = {:.2e}, iters = {}".format(fk, gk, k))

    xk, log = l_bfgs(func, grad, m, x0)
    fk, gk, k = log[-1, -2], log[-1, -1], log.__len__()
    print("l-bfgs(m={}): fk = {:.2e}, ||gk|| = {:.2e}, iters = {}".format(m, fk, gk, k))

    for secant in [False, True]:
        for version in ["A", "B"]:
            xk, log = fast_bfgs(func, grad, x0, m=m, version=version)
            fk, gk, k = log[-1, -2], log[-1, -1], log.__len__()
            print("fast-bfgs(secant={}, version={}, m={}): fk = {:.2e}, ||gk|| = {:.2e}, iters = {}".format(
                secant, version, m, fk, gk, k))

    # Solving Styblinski & Tang's function.
    for n in [128, 256, 512, 1024]:
        print('-' * 32 + "\tSolving {}-dim Styblinski & Tang's function\t".format(n) + '-' * 32)
        func = Styblinski.function
        grad = Styblinski.gradients
        x0 = -2.5 * np.ones(shape=(n, )) + 0.1 * np.random.rand(n)

        xk, log = gradient_descent(func, grad, x0)
        fk, gk, k = log[-1, -2], log[-1, -1], log.__len__()
        print("gradient descent: fk = {:.2e}, ||gk|| = {:.2e}, iters = {}".format(fk, gk, k))

        xk, log = bfgs(func, grad, x0)
        fk, gk, k = log[-1, -2], log[-1, -1], log.__len__()
        print("bfgs: fk = {:.2e}, ||gk|| = {:.2e}, iters = {}".format(fk, gk, k))

        xk, log = l_bfgs(func, grad, m, x0)
        fk, gk, k = log[-1, -2], log[-1, -1], log.__len__()
        print("l-bfgs(m={}): fk = {:.2e}, ||gk|| = {:.2e}, iters = {}".format(m, fk, gk, k))

        for secant in [False, True]:
            for version in ["A", "B"]:
                xk, log = fast_bfgs(func, grad, x0, m=m, version=version)
                fk, gk, k = log[-1, -2], log[-1, -1], log.__len__()
                print("fast-bfgs(secant={}, version={}, m={}):fk = {:.2e}, ||gk|| = {:.2e}, iters = {}".format(
                    secant, version, m, fk, gk, k))


# unit_test()
