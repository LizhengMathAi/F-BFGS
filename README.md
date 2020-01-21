# F-BFGS

## Introduction
Codes for paper "A Dynamic Subspace Based BFGS Method for Large Scale Optimization Problem".

## Algorithms
See the function `unit_test` in `f_bfgs.py` for addition examples.

To run various methods with termination criterion $\|\nabla f_k\|_2 < tol$, simply run functions in `f_bfgs.py`.
``` python
import numpy as np
import f_bfgs


def function(x):
    pass


def gradients(x):
    pass


x0 = np.random.rand(1024)
tol = 1e-5

# gradient descent method
xk, log = f_bfgs.gradient_descent(function, gradient, x0, tol=tol)
# Newton method
xk, log = f_bfgs.newton(function, gradient, x0, tol=tol)
# BFGS method
xk, log = f_bfgs.bfgs(function, gradient, x0, tol=tol)
# L-BFGS method
xk, log = f_bfgs.l_bfgs(function, gradient, m=8, x0=x0, tol=tol)
# Fast-BFGS method(version A)
xk, log = f_bfgs.fast_bfgs(function, gradient, m=8, x0=x0, secant=False, version="A", tol=tol)
# Fast-BFGS method(version B)
xk, log = f_bfgs.fast_bfgs(function, gradient, m=8, x0=x0, secant=False, version="B", tol=tol)
```
The return value `log` is the log of iteration.

|        | x_1 | x_2 | ... | x_n | $f(x_k)$ | $\| \nabla f_k \|_2$ |
| ---    | --- | --- | --- | --- | ---      | ---                  |
| step 0 | ... | ... | ... | ... | ...      | ...                  |
| step 1 | ... | ... | ... | ... | ...      | ...                  |
| ...    | ... | ... | ... | ... | ...      | ...                  |

To run various methods with termination criterion $\frac{\|\nabla f_{k+1}\|_2}{\|\nabla f_k\|_2} < tol$, Setting `f_bfgs.TERMINATION_VERSION = 1` after `import f_bfgs`.
``` python
import numpy as np
import f_bfgs


f_bfgs.TERMINATION_VERSION = 1


def function(x):
    pass


def gradients(x):
    pass
```


## CUTE Collection and Experiments
See the function `unit_test` and `memory_experiments` in `cute_problems.py` for addition examples.

The structure of `CUTE Collection` in `cute_problems.py` is 
* class Problem
    - @classmethod def function
    - @classmethod def gradients
    - @classmethod def gradients_check
    - @classmethod def gen_x0
    - @classmethod def solve
* class CUTE.ARWHEAD(Problem)
* class CUTE.BDQRTIC(Problem)
* class CUTE.BDEXP(Problem)
* class CUTE.COSINE(Problem)
* class CUTE.DIXMAANE(Problem)
* class CUTE.DIXMAANF(Problem)
* class CUTE.DIXMAANG(Problem)
* class CUTE.DQRTIC(Problem)
* class CUTE.EDENSCH(Problem)
* class CUTE.ENGVAL1(Problem)
* class CUTE.EG2(Problem)
* class CUTE.EXTROSNB(Problem)
* class CUTE.FLETCHER(Problem)
* class CUTE.FREUROTH(Problem)
* class CUTE.GENROSE(Problem)
* class CUTE.HIMMELBG(Problem)
* class CUTE.HIMMELH(Problem)
* class CUTE.LIARWHD(Problem)
* class CUTE.NONDIA(Problem)
* class CUTE.NONDQUAR(Problem)
* class CUTE.NONSCOMP(Problem)
* class CUTE.POWELLSG(Problem)
* class CUTE.SCHMVETT(Problem)
* class CUTE.SINQUAD(Problem)
* class CUTE.SROSENBR(Problem)
* class CUTE.TOINTGSS(Problem)
* class CUTE.TQUARTIC(Problem)
* class CUTE.WOODS(Problem)

Here are some examples of solving problems in CUTE collection.
```python
import numpy as np
import f_bfgs
import cute_problems


function = cute_problems.CUTE.ARWHEAD.function
gradients = cute_problems.CUTE.ARWHEAD.gradients
x0 = cute_problems.CUTE.ARWHEAD.gen_x0(1024)

# gradient descent method
xk, log = f_bfgs.gradient_descent(function, gradient, x0)
# Newton method
xk, log = f_bfgs.newton(function, gradient, x0)
# BFGS method
xk, log = f_bfgs.bfgs(function, gradient, x0)
# L-BFGS method
xk, log = f_bfgs.l_bfgs(function, gradient, m=8, x0=x0)
# Fast-BFGS method(version A)
xk, log = f_bfgs.fast_bfgs(function, gradient, m=8, x0=x0, secant=False, version="A")
# Fast-BFGS method(version B)
xk, log = f_bfgs.fast_bfgs(function, gradient, m=8, x0=x0, secant=False, version="B")
```


## Parallel Algorithms
We provided the parallel mode for BFGS, L-BFGS and Fast-BFGS(version A). See the comments in `parallel_f_bfgs.py` for details.