Least Squares Regression with Elastic Net Regularization
=================

This repository contains my own implementation of least-squares regression with elastic net regularization.

I wrote most of this code for the take-home portion of my DATA 558 Midterm in Spring 2017. I subsequently enhanced and cleaned the code prior to releasing it on GitHub as part of the final Polished Code Release assignment for the course.

---------------

Elastic net is a hybrid approach between the ever-popular ℓ1 (LASSO) and ℓ2 (Ridge) regularization methods; the minimization problem writes as follows:

min F(β) = 1/n * ||Y − XTβ||2^2 +λ α||β||1 +(1−α)||β||2^2)
β∈Rd

where α ∈ (0, 1), with the two extremes equating to the Ridge and LASSO problems, respectively.

Because of the ℓ1 component (i.e. λ α||β||1), the objective function above is non-differentiable and therefore cannot be minimized by gradient descent. Instead, we leverage the subgradient of the absolute value function to define a soft-thresholding operator which we use to minimize the objective function with respect to one coordinate at a time. This process is known as coordinate descent.

I provide a function called `coorddescent` which solves the coordinate descent algorithm described above. The function includes two :

* **cyclic:** proceeds sequentially through each coordinate, returning to the first coordinate after all coordinates have been updated
* **random:** proceeds randomly through the coordinates

This repository contains the following:

```
src/coorddescent.py
data/Hitters.csv
Demo 1: Simulated dataset
Demo 2: Real-world dataset
Demo 3: Comparison to scikit-learn
```
