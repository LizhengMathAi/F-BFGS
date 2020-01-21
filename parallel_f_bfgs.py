import time
import numpy as np
import tensorflow as tf

import f_bfgs as serial

# -------- [ Styblinski & Tang's function(1990),Including Communication Time ] -------- #
# +-------------------------------------------------------------------------------------+
# | m  | n    | L-BFGS(pk)             | ver-A(pk)              | ver-A(ak)             |
# +----+------+------------------------+------------------------+-----------------------+
# | 2  | 2    | 0.000643440403206678 s | 0.001328898015130064 s | 0.00068024826851051 s |
# | 2  | 2^10 | 0.000637397763251686 s | 0.001398700622349506 s | 0.00075107511221635 s |
# | 2  | 2^15 | 0.000711795877852328 s | 0.001563036840041653 s | 0.00124819411304288 s |
# | 4  | 2^10 | 0.000865752087948107 s | 0.001373611609681555 s | 0.00070530892753868 s |
# | 4  | 2^15 | 0.000948097782627204 s | 0.001498389444391258 s | 0.00126336941506343 s |
# | 8  | 2^10 | 0.001217196573851513 s | 0.001376778155618916 s | 0.00075274881159551 s |
# | 8  | 2^15 | 0.001415753512412077 s | 0.001544760880314796 s | 0.00142903176754277 s |
# | 16 | 2^10 | 0.001928193470458312 s | 0.001386631868152194 s | 0.00076677980936153 s |
# | 16 | 2^15 | 0.002266560895225004 s | 0.001661728610752058 s | 0.00149819757442852 s |
# | 32 | 2^10 | 0.003334326275731831 s | 0.001368592610047278 s | 0.00190438375780167 s |
# | 32 | 2^15 | 0.003630589070046370 s | 0.001728885076980301 s | 0.00254183436517931 s |
# +-------------------------------------------------------------------------------------+


#  --------------------------- line-search ---------------------------
def wolfe(f_alpha, fk, alpha, gpk):
    """
    :param f_alpha: func(xk + alpha * pk)
    :param fk:      func(xk)
    :param alpha:
    :param gpk:     grad(xk) @ pk
    """
    c1 = 1e-4
    return f_alpha <= fk + c1 * alpha * gpk


def strong_wolfe(f_alpha, fk, alpha, gpk, gpk_alpha):
    """
    :param f_alpha:   func(xk + alpha * pk)
    :param fk:        func(xk)
    :param alpha:
    :param gpk:       grad(xk) @ pk
    :param gpk_alpha: grad(xk + alpha * pk) @ pk
    :return:
    """
    c2 = 0.9
    return wolfe(f_alpha, fk, alpha, gpk) and abs(gpk_alpha) <= c2 * abs(gpk)


def step_length(sess, fk, gpk, alpha_ph, max_iters=20):
    """
    :param sess:
    :param fk:        tf.Tensor
    :param gpk:       tf.Tensor
    :param alpha_ph:  tf.Placeholder
    :param max_iters:
    :return:
    """
    alpha = 1.
    l = 0.0
    h = 1.0

    fk_val, gpk_val = sess.run([fk, gpk], feed_dict={alpha_ph: 0.})
    f_alpha_val, gpk_alpha_val = sess.run([fk, gpk], feed_dict={alpha_ph: alpha})

    for i in range(max_iters):
        if strong_wolfe(f_alpha_val, fk_val, alpha, gpk_val, gpk_alpha_val):
            return alpha

        half = (l + h) / 2

        f_l_val, gpk_l_val = sess.run([fk, gpk], feed_dict={alpha_ph: l})
        f_h_val, = sess.run([fk], feed_dict={alpha_ph: h})
        alpha = - gpk_l_val * (h ** 2) / (2 * (f_h_val - f_l_val - gpk_l_val * h))
        if alpha < l or alpha > h:
            alpha = half

        f_alpha_val, gpk_alpha_val = sess.run([fk, gpk], feed_dict={alpha_ph: alpha})
        if gpk_alpha_val > 0:
            h = alpha
        elif gpk_alpha_val <= 0:
            l = alpha

    return alpha


class QuasiNewton:
    xk = None
    alpha = None
    gk_record = None
    pk_record = None

    @classmethod
    def init_misc(cls, xk_init, gk_init=None):
        with tf.variable_scope("init", reuse=tf.AUTO_REUSE):
            n = xk_init.__len__()

            xk = tf.constant(xk_init, dtype=tf.float64)
            xk = tf.get_variable('xk', initializer=xk)

            alpha = tf.placeholder(tf.float64, shape=(), name="alpha")

            if gk_init is None:
                gk_init = tf.constant(np.zeros(shape=(n,)), dtype=tf.float64)
            else:
                gk_init = tf.constant(gk_init, dtype=tf.float64)
            gk_record = tf.get_variable('gk', initializer=gk_init)

            pk_init = tf.constant(np.zeros(shape=(n,)), dtype=tf.float64)
            pk_record = tf.get_variable('pk', initializer=pk_init)

        return xk, alpha, gk_record, pk_record

    def compute_misc(self, func):
        with tf.name_scope("compute"):
            # (self.alpha, self.pk_record) -> sk
            sk = self.alpha * self.pk_record
            # (self.xk, self.alpha, self.pk_record) -> xk_shift
            xk_shift = self.xk + sk
            # (self.xk, self.alpha, self.pk_record) -> fk
            fk = func(xk_shift)
            # (self.xk, self.alpha, self.pk_record) -> gk
            gk = tf.gradients(fk, [xk_shift])[0]
            # (self.xk, self.alpha, self.pk_record, self.gk_record) -> yk
            yk = gk - self.gk_record
            # (self.xk, self.alpha, self.pk_record, self.pk_record) -> gpk
            gpk = tf.reduce_sum(gk * self.pk_record)

        return sk, fk, gk, yk, gpk

    def solve(self, tol=1e-5, max_iters=1000):
        return 0


#  --------------------------- bfgs ---------------------------
class BFGS(QuasiNewton):
    def __init__(self, func, x0):
        self.xk, self.alpha, self.gk_record, self.pk_record = self.init_misc(x0, gk_init=None)

        n = x0.__len__()

        hk_init = tf.constant(np.eye(n), dtype=tf.float64)
        self.hk = tf.get_variable('hk', initializer=hk_init)

        sk, self.fk, self.gk, yk, self.gpk = self.compute_misc(func)

        with tf.name_scope("update"):
            # (self.hk, self.xk, self.alpha, self.pk_record) -> pk_update
            pk = -tf.einsum("ij,j->i", self.hk, self.gk_record)
            self.pk_update = tf.assign(self.pk_record, pk)

            # (self.hk, self.xk, self.alpha, self.pk_record, self.gk_record) -> hk_update
            temp = tf.eye(n, dtype=tf.float64) - tf.einsum('i,j->ij', sk, yk) / tf.einsum('i,i->', sk, yk)
            temp = tf.matmul(tf.matmul(temp, self.hk), tf.transpose(temp))
            temp = temp + tf.einsum('i,j->ij', sk, sk) / tf.einsum('i,i->', sk, yk)
            hk_update = tf.assign(self.hk, temp)
            with tf.control_dependencies([hk_update]):
                # (self.gk_record, self.xk, self.alpha, self.pk_record) -> gk_update
                gk_update = tf.assign(self.gk_record, self.gk)
                with tf.control_dependencies([gk_update]):
                    # (self.xk, self.alpha, self.pk_record) -> xk_update
                    self.xk_update = tf.assign_add(self.xk, sk)

    def solve(self, tol=1e-5, max_iters=1000):
        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            k = 0
            sess.run(tf.assign(self.gk_record, self.gk), feed_dict={self.alpha: 0.})
            fk, norm_gk = sess.run([self.fk, tf.linalg.norm(self.gk_record)], feed_dict={self.alpha: 0.})
            logs = [[k, fk, norm_gk]]

            while logs.__len__() < max_iters:
                # update pk
                sess.run(self.pk_update, feed_dict={self.alpha: 0.})
                # compute tau
                tau = step_length(sess, self.fk, self.gpk, self.alpha)
                # update hk, gk, xk
                sess.run(self.xk_update, feed_dict={self.alpha: tau})

                k += 1
                fk, norm_gk = sess.run([self.fk, tf.linalg.norm(self.gk_record)], feed_dict={self.alpha: 0.})
                logs.append([k, fk, norm_gk])
                if norm_gk < tol:
                    break

            return sess.run(self.xk), np.array(logs)


#  --------------------------- l-bfgs ---------------------------
class LBFGS(QuasiNewton):
    @staticmethod
    def serial_init(func, grad, x0, m=8):
        xk = x0
        gk = grad(xk)
        I = np.identity(xk.size)

        sks = []
        yks = []

        while len(sks) < m:
            # compute search direction
            pk = serial.two_loop_recursion(sks, yks, I, gk)

            # obtain step length by line search
            tau = serial.step_length(func, grad, xk, alpha=1.0, pk=pk, c2=0.9)

            # update x
            xk1 = xk + tau * pk
            gk1 = grad(xk1)

            # define sk and yk for convenience
            sk = xk1 - xk
            yk = gk1 - gk

            xk = xk1
            gk = gk1
            sks.append(sk)
            yks.append(yk)

        return xk, gk, np.array(sks), np.array(yks)

    def two_loop_recursion(self, m, q):
        a = [None for _ in range(m)]
        b = [None for _ in range(m)]

        for i in reversed(range(m)):
            s = self.sks[i]
            y = self.yks[i]
            a[i] = tf.reduce_sum(s * q) / tf.reduce_sum(y * s)
            q = q - a[i] * y

        r = q

        for i in range(m):
            s = self.sks[i]
            y = self.yks[i]
            b[i] = tf.reduce_sum(y * r) / tf.reduce_sum(y * s)
            r = r + s * (a[i] - b[i])

        return -r

    def __init__(self, func, grad, x0, m=8):
        xm, gm, skm, ykm = self.serial_init(func, grad, x0, m=m)
        self.xk, self.alpha, self.gk_record, self.pk_record = self.init_misc(xm, gk_init=gm)

        sks_init = tf.constant(skm, dtype=tf.float64)
        self.sks = tf.get_variable('sks', initializer=sks_init)

        yks_init = tf.constant(ykm, dtype=tf.float64)
        self.yks = tf.get_variable('yks', initializer=yks_init)

        sk, self.fk, self.gk, yk, self.gpk = self.compute_misc(func)

        with tf.name_scope("update"):
            # (self.sks, self.yks, self.xk, self.alpha, self.pk_record) -> pk
            pk = self.two_loop_recursion(m, self.gk)
            self.pk_update = tf.assign(self.pk_record, pk)

            # (self.sks, self.alpha, self.pk_record) -> sks_update
            new_sks = tf.concat([self.sks[1:], tf.expand_dims(sk, axis=0)], axis=0)
            sks_update = tf.assign(self.sks, new_sks)
            # (self.yks, self.xk, self.alpha, self.pk_record, self.gk_record) -> yks_update
            new_yks = tf.concat([self.yks[1:], tf.expand_dims(yk, axis=0)], axis=0)
            yks_update = tf.assign(self.yks, new_yks)

            with tf.control_dependencies([sks_update, yks_update]):
                # (self.gk_record, self.xk, self.alpha, self.pk_record) -> gk_update
                gk_update = tf.assign(self.gk_record, self.gk)
                with tf.control_dependencies([gk_update]):
                    # (self.xk, self.sks) -> xk_update
                    self.xk_update = tf.assign_add(self.xk, self.sks[-1])

    def solve(self, tol=1e-5, max_iters=1000):
        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            k = self.sks.get_shape().as_list()[0]
            sess.run(tf.assign(self.gk_record, self.gk), feed_dict={self.alpha: 0.})
            fk, norm_gk = sess.run([self.fk, tf.linalg.norm(self.gk_record)], feed_dict={self.alpha: 0.})
            logs = [[k, fk, norm_gk]]

            while logs.__len__() < max_iters:
                # update pk
                sess.run(self.pk_update, feed_dict={self.alpha: 0.})
                # compute tau
                tau = step_length(sess, self.fk, self.gpk, self.alpha)
                # update sks, yks, gk, xk
                sess.run(self.xk_update, feed_dict={self.alpha: tau})

                k += 1
                fk, norm_gk = sess.run([self.fk, tf.linalg.norm(self.gk_record)], feed_dict={self.alpha: 0.})
                logs.append([k, fk, norm_gk])
                if norm_gk < tol:
                    break

            return sess.run(self.xk), np.array(logs)

    def time_spy(self, max_iters=10000):
        """
        m = 2, n = 2^15
        pk: 0.0007480820504921296 s.
        m = 4, n = 2^15
        pk: 0.0009806187874461488 s.
        m = 8, n = 2^15
        pk: 0.001430450427243652 s.
        m = 16, n = 2^15
        pk: 0.0022952255427205435 s.
        m = 32, n = 2^15
        pk: 0.0035891839327109195 s.
        """
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            sess.run(tf.assign(self.gk_record, self.gk), feed_dict={self.alpha: 0.})

            pk_time = []
            for _ in range(max_iters):
                start_time = time.time()
                sess.run(self.pk_update, feed_dict={self.alpha: 0.})
                pk_time.append(time.time() - start_time)
            pk_time = (np.sum(pk_time) - np.max(pk_time) - np.min(pk_time)) / (max_iters - 2)
            print("pk: {} s.".format(pk_time))


#  --------------------------- fast-bfgs ---------------------------
class FastBFGS(QuasiNewton):
    @staticmethod
    def serial_init(func, grad, x0, m=8, acculate=False, version="A"):
        xk = x0
        gk = grad(xk)
        logs = [[*xk, func(xk), np.linalg.norm(gk)]]

        # compute x1, s0, y0
        pk = -gk
        tau = serial.step_length(func, grad, xk, alpha=1.0, pk=pk, c2=0.9)
        sk = tau * pk
        xk1 = xk + sk
        gk1 = grad(xk1)
        yk = gk1 - gk

        # update xk, gk, sks, lk, k
        xk = xk1
        gk = gk1
        sks = [sk / np.sqrt(np.abs(np.inner(sk, yk)))]
        lk = np.array([[1]])
        logs.append([*xk, func(xk), np.linalg.norm(gk)])

        while sks.__len__() < m:
            # compute pk
            pk = -np.array(sks).T @ (lk @ (np.array(sks) @ gk))

            if m < x0.__len__() or sks.__len__() <= x0.__len__():
                # fix pk
                pk = serial.fix_pk(grad, xk, gk, pk, version=version)

            # compute xk1, sk, yk
            tau = serial.step_length(func, grad, xk, alpha=1.0, pk=pk, c2=0.9)
            sk = tau * pk
            xk1 = xk + sk
            gk1 = grad(xk1)
            yk = gk1 - gk

            # update xk, gk, sks, yk, lk, k
            xk = xk1
            gk = gk1
            sks.append(sk / np.sqrt(np.abs(np.inner(sk, yk))))
            yk = yk / np.sqrt(np.abs(np.inner(sk, yk)))
            lk = serial.update_lk(m, yk, sks, lk, acculate)

        return xk, gk, np.array(sks), lk

    def update_lk(self, sess, tau):
        """
        lk <- tk @ lk @ tk.T + [[0, 0], [0, 1]].
        """
        ys, mat, rhs = sess.run(self.misc_ak, feed_dict={self.alpha: tau})

        m = self.lk_val.shape[0]
        p = np.zeros(shape=(m, m))
        p[:-1, 1:] = np.eye(m - 1)
        p[-1] = ys

        mat + 1e-6 * np.eye(mat.shape[0]) if np.linalg.det(mat) == 0 else mat
        coeff = np.linalg.solve(mat, rhs)
        p[:, 0] += coeff

        self.lk_val = p @ self.lk_val @ p.T
        self.lk_val[-1, -1] += 1

    def compute_pk(self):
        with tf.name_scope("pk"):
            pk = tf.einsum("ij,j->i", self.sks, self.gk_record)
            pk = tf.einsum("ij,j->i", self.lk, pk)
            pk = tf.einsum("ji,j->i", self.sks, pk)
            pk = -pk

        with tf.name_scope("fix_pk"):
            with tf.name_scope("estimate_vk"):
                v1 = self.gk_record

                with tf.name_scope("bg"):
                    # TODO: `1e-6` is set based on experimental results.
                    eps = 1e-6 / tf.linalg.norm(self.gk_record)
                    bg = (self.grad(self.xk + eps * self.gk_record) - self.gk_record) / eps

                with tf.name_scope("b2g"):
                    pred = tf.greater(tf.linalg.norm(pk), 0)
                    eps = 1e-6 / tf.linalg.norm(pk)
                    bp = (self.grad(self.xk + eps * pk) - self.gk_record) / eps
                    eps = 1e-6 / tf.linalg.norm(bp)
                    b2p = (self.grad(self.xk + eps * bp) - self.gk_record) / eps
                    b2p = tf.cond(pred, lambda: b2p, lambda: tf.zeros_like(b2p))

                v2 = bg - b2p

                orth_v1 = v1 - tf.reduce_sum(v1 * v2) / tf.reduce_sum(v2 * v2) * v2
                orth_v2 = v2 - tf.reduce_sum(v1 * v2) / tf.reduce_sum(v1 * v1) * v1

                vk = tf.cond(
                    tf.greater(tf.linalg.norm(orth_v1) * tf.linalg.norm(orth_v2), 0),
                    lambda: 0.5 / tf.linalg.norm(orth_v1) * orth_v1 + 0.5 / tf.linalg.norm(orth_v2) * orth_v2,
                    lambda: 0.5 * v1 + 0.5 * v2)

                vk = vk / tf.linalg.norm(vk)

            with tf.name_scope("estimate_alpha"):
                # TODO: `1e-6` is set based on experimental results.
                eps = 1e-6 / tf.linalg.norm(vk)
                bv = (self.grad(self.xk + eps * vk) - self.gk_record) / eps

                alpha = tf.reduce_sum((self.gk_record - bp) * bv) / tf.reduce_sum(tf.square(bv))

        return tf.assign(self.pk_record, pk - alpha * vk)

    def __init__(self, func, grad, x0, m=8):
        self.grad = grad

        xm, gm, skm, self.lk_val = self.serial_init(func, grad, x0, m=m)
        self.xk, self.alpha, self.gk_record, self.pk_record = self.init_misc(xm, gk_init=gm)

        sks_init = tf.constant(skm, dtype=tf.float64)
        self.sks = tf.get_variable('sks', initializer=sks_init)

        self.lk = tf.placeholder(shape=(m, m), dtype=tf.float64)

        sk, self.fk, self.gk, yk, self.gpk = self.compute_misc(func)
        # (self.xk, self.alpha, self.pk_record, self.gk_record) -> tilde_sk
        tilde_sk = sk * tf.rsqrt(tf.abs(tf.reduce_sum(sk * yk)))
        # (self.xk, self.alpha, self.pk_record, self.gk_record) -> tilde_yk
        tilde_yk = yk * tf.rsqrt(tf.abs(tf.reduce_sum(sk * yk)))

        ys = -tf.einsum("ij,j->i", self.sks, tilde_yk)
        new_sks = tf.concat([self.sks[1:], tf.expand_dims(tilde_sk, axis=0)], axis=0)
        mat = tf.matmul(new_sks, new_sks, transpose_b=True)
        rhs = tf.einsum("ij,j->i", new_sks, self.sks[0])
        self.misc_ak = [ys, mat, rhs]

        with tf.name_scope("update"):
            # (self.sks, self.lk, self.xk, self.alpha, self.pk_record) -> pk
            self.pk_update = self.compute_pk()

            # (self.sks, self.xk, self.alpha, self.pk_record, self.gk_record) -> sks_update
            sks_update = tf.assign(self.sks, new_sks)

            with tf.control_dependencies([sks_update]):
                # (self.gk_record, self.xk, self.alpha, self.pk_record) -> gk_update
                gk_update = tf.assign(self.gk_record, self.gk)
                with tf.control_dependencies([gk_update]):
                    # (self.xk, self.sks) -> xk_update
                    self.xk_update = tf.assign_add(self.xk, self.alpha * self.pk_record)
        self.temp = [self.xk[:3]]

    def solve(self, tol=1e-5, max_iters=1000):
        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            sess.run(tf.assign(self.gk_record, self.gk), feed_dict={self.alpha: 0.})
            fk, norm_gk = sess.run([self.fk, tf.linalg.norm(self.gk_record)], feed_dict={self.alpha: 0.})

            k = self.lk_val.shape[0] - 1
            logs = [[k, fk, norm_gk]]

            while logs.__len__() < max_iters:
                # update pk
                sess.run(self.pk_update, feed_dict={self.alpha: 0., self.lk: self.lk_val})
                # compute tau
                tau = step_length(sess, self.fk, self.gpk, self.alpha)
                # update lk
                self.update_lk(sess, tau)
                # update sks, gk, xk
                sess.run(self.xk_update, feed_dict={self.alpha: tau})

                k += 1
                fk, norm_gk = sess.run([self.fk, tf.linalg.norm(self.gk_record)], feed_dict={self.alpha: 0.})
                logs.append([k, fk, norm_gk])
                if norm_gk < tol:
                    break

            return sess.run(self.xk), np.array(logs)

    def time_spy(self, max_iters=10000):
        """
        # Styblinski & Tang's function(1990)
        m = 2, n = 2^15
        pk: 0.0015379705055162034 s.
        lk: 0.0012625766577876122 s.
        m = 4, n = 2^15
        pk: 0.0015346092995607177 s.
        lk: 0.001283996795315484 s.
        m = 8, n = 2^15
        pk: 0.0014974982243915825 s.
        lk: 0.0013846234813597851 s.
        m = 16, n = 2^15
        pk: 0.00167377812072119 s.
        lk: 0.0015173963986293963 s.
        m = 32, n = 2^15
        pk: 0.0016935241773238681 s.
        lk: 0.0024556207475625986 s.
        """
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            sess.run(tf.assign(self.gk_record, self.gk), feed_dict={self.alpha: 0.})

            pk_time = []
            for _ in range(max_iters):
                start_time = time.time()
                sess.run(self.pk_update, feed_dict={self.alpha: 0., self.lk: self.lk_val})
                pk_time.append(time.time() - start_time)
            pk_time = (np.sum(pk_time) - np.max(pk_time) - np.min(pk_time)) / (max_iters - 2)
            print("pk: {} s.".format(pk_time))

            ak_time = []
            for _ in range(max_iters):
                start_time = time.time()
                self.update_lk(sess, 1.)
                ak_time.append(time.time() - start_time)
            ak_time = (np.sum(ak_time) - np.max(ak_time) - np.min(ak_time)) / (max_iters - 2)
            print("lk: {} s.".format(ak_time))


#  --------------------------- solvers collection ---------------------------
def bfgs(func, x0, tol=1e-5, max_iters=1000):
    return BFGS(func, x0=x0).solve(tol=tol, max_iters=max_iters)


def l_bfgs(func, grad, x0, m, tol=1e-5, max_iters=1000):
    return LBFGS(func, grad, m=m, x0=x0).solve(tol=tol, max_iters=max_iters)


def fast_bfgs(func, grad, x0, m, tol=1e-5, max_iters=1000):
    return FastBFGS(func, grad, m=m, x0=x0).solve(tol=tol, max_iters=max_iters)


#  --------------------------- test ---------------------------
# Styblinski & Tang's function(1990)
class Styblinski:
    @classmethod
    def function(cls, x):
        if isinstance(x, np.ndarray):
            return np.mean(x ** 4 - 16 * x ** 2 + 5 * x)
        else:
            return tf.reduce_mean(x ** 4 - 16 * x ** 2 + 5 * x)

    @classmethod
    def gradients(cls, x):
        if isinstance(x, np.ndarray):
            n = x.__len__()
        else:
            n = x.get_shape().as_list()[0]
        return (4 * x ** 3 - 32 * x + 5) / n


# m, n = 32, 2 ** 10
# x0 = -2.5 * np.ones(shape=(n, )) + np.linspace(0, 1, n) - 0.5
#
# # ------------------------------ test parallel mode ------------------------------
# xk, logs = BFGS(Styblinski.function, x0=x0).solve()
# xk, logs = LBFGS(Styblinski.function, Styblinski.gradients, m=m, x0=x0).solve(tol=1e-8)
# xk, logs = serial.fast_bfgs(Styblinski.function, Styblinski.gradients, x0, m, tol=1e-8)
# print(logs[:, -2:])
# xk, logs = fast_bfgs(Styblinski.function, Styblinski.gradients, x0, m, tol=1e-8)
# print(logs)
# # ------------------------------ test time spy ------------------------------
# LBFGS(Styblinski.function, Styblinski.gradients, x0=x0, m=m).time_spy()
# FastBFGS(Styblinski.function, Styblinski.gradients, x0=x0, m=m).time_spy()

# # ------------------------------ make table ------------------------------
data = np.array([
    [4, 2**10, 0.000865752087948107, 0.001373611609681555, 0.00070530892753868],
    [8, 2**10, 0.001217196573851513, 0.001376778155618916, 0.00075274881159551],
    [16, 2**10, 0.001928193470458312, 0.001386631868152194, 0.00076677980936153],
    [32, 2**10, 0.003334326275731831, 0.001368592610047278, 0.00190438375780167],
    [4, 2**15, 0.000948097782627204, 0.001498389444391258, 0.00126336941506343],
    [8, 2**15, 0.001415753512412077, 0.001544760880314796, 0.00142903176754277],
    [16, 2**15, 0.002266560895225004, 0.001661728610752058, 0.00149819757442852],
    [32, 2**15, 0.003630589070046370, 0.001728885076980301, 0.00254183436517931]
])


data[:, 2:] -= np.array([0.000643440403206678, 0.001328898015130064, 0.0006802482685105137])


import pandas as pd


df = pd.DataFrame(data=data, columns=['m', 'n', "L_BFGS", "pk", "lk"])
print(df)

#    m        n    L_BFGS        pk        lk
#
#  4.0   1024.0  0.000222  0.000045  0.000025
#  8.0   1024.0  0.000574  0.000048  0.000073
# 16.0   1024.0  0.001285  0.000058  0.000087
# 32.0   1024.0  0.002691  0.000028  0.001599
#
#  4.0  32768.0  0.000305  0.000169  0.000583
#  8.0  32768.0  0.000772  0.000216  0.000749
# 16.0  32768.0  0.001623  0.000333  0.000818
# 32.0  32768.0  0.002987  0.000400  0.001862
