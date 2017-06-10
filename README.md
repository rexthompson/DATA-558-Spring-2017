Least Squares Regression with Elastic Net Regularization
=================

This repository contains my own Python implementation of least-squares regression with elastic net regularization.

Elastic Net
----------------

The elastic net is a hybrid approach between the ever-popular ℓ1 (LASSO) and squared ℓ2 (Ridge) regularization penalties. It is capable of mitigating some of LASSO's weaknesses (correlated variables, p > n) while still maintaining its desirable variable selection capabilities.

The elastic net least-squares minimization problem writes as follows:

<img src=https://github.com/rexthompson/DATA-558-Spring-2017/blob/master/images/ElasticNet.png alt="Objective Function" width="500" height="60" />

where α ∈ [0, 1], with the two extremes equating to the Ridge and LASSO problems, respectively.

Because of the ℓ1 component, the objective function above is non-differentiable and therefore cannot be minimized by gradient descent. Instead, we leverage the subgradient of the absolute value function to define a soft-thresholding operator which is used to minimize the objective function one coordinate at a time. This process is known as coordinate descent.

This Repository
----------------

This repository contains Python code for solving the minimization problem described above. Specifically, I provide a function called `coorddescent` (in `src/coorddescent.py`) which solves the coordinate descent algorithm described above in one of two ways:

* **cyclic:** proceeds sequentially through each coordinate, returning to the first coordinate after all coordinates have been updated; repeats until stopping criterion achieved
* **random:** proceeds randomly through the coordinates; repeats until stopping criterion achieved

`src/coorddescent.py` also includes several supplemental functions which are either called by `coorddescent` or are useful for visualizing how the algorithm arrives at its solution. I also include a cross-validation function (`coorddescentCV`).

Please refer to the following examples in which I demonstrate the functionality of the code in this repository:

* [**Demo 1**](https://github.com/rexthompson/DATA-558-Spring-2017/blob/master/Demo%201%20-%20Simulated%20Data.ipynb)**:** Coordinate descent on a simulated dataset
* [**Demo 2**](https://github.com/rexthompson/DATA-558-Spring-2017/blob/master/Demo%202%20-%20Real%20World%20Data.ipynb)**:** Coordinate descent on a "real-world" dataset
* [**Demo 3**](https://github.com/rexthompson/DATA-558-Spring-2017/blob/master/Demo%203%20-%20Comparison%20to%20Scikit-learn.ipynb)**:** Comparison of my functions to those from scikit-learn

I wrote most of this code for the take-home portion of my DATA 558 Midterm in Spring 2017. I subsequently enhanced and cleaned the code prior to releasing it on GitHub as part of the Polished Code Release assignment for the course.

Installation
-----------

To use the code in this repository:

* clone the repository
* navigate to the main directory (i.e. that which contains this README.md file)
* launch python
* enter `import src.coorddescent` (or `... as cd` or whatever shorthand you prefer)

The functions from `coorddescent.py` should now be available to you in Python by typing `src.coorddescent.<function_name>` (or `cd.<function_name>` if you use the shorthand recommended in the last bullet above).

This code was developed in Python 3.6.0; functionality is not guaranteed for older versions. You may need to install the following dependencies if they do not already exist on your machine:

* copy
* matplotlib.pyplot
* numpy
* pandas
* sklearn.metrics.mean_squared_error

#### References

Hastie, Trevor J., Robert John Tibshirani, and Martin J. Wainwright. "4.2." _Statistical Learning with Sparsity: The Lasso and Generalizations._ Boca Raton: CRC, Taylor & Francis Group, 2015. N. pag. Print.
