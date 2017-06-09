Least Squares Regression with Elastic Net Regularization
=================

This repository contains my own implementation of least-squares regression with elastic net regularization.

Elastic net is a hybrid approach between the ever-popular ℓ1 (LASSO) and squared ℓ2 (Ridge) regularization penalties. It is capable of mitigating the LASSO's difficulties with correlated variables while still maintaining the desirable dimension reduction capabilities of the LASSO.

The elastic net least-squares minimization problem writes as follows:

min F(β) = 1/n * ||Y − XTβ||2^2 + λα||β||1 + (1 − α)||β||2^2)
β∈Rd

where α ∈ [0, 1], with the two extremes equating to the Ridge and LASSO problems, respectively.

Because of the ℓ1 component (i.e. λα||β||1), the objective function above is non-differentiable and therefore cannot be minimized by gradient descent. Instead, we leverage the subgradient of the absolute value function (from the ℓ1 term) to define a soft-thresholding operator which is used to minimize the objective function one coordinate at a time. This process is known as coordinate descent.

---

This repository contains Python code for solving the minimization problem above. Specifically, I provide a function called `coorddescent` which solves the coordinate descent algorithm above in one of two ways:

* **cyclic:** proceeds sequentially through each coordinate, returning to the first coordinate after all coordinates have been updated; repeats until stopping criterion achieved
* **random:** proceeds randomly through the coordinates; repeats until stopping criterion achieved

I also include the following sub-functions:



I provide the following examples which demonstrate the capabilities of the code:

```
Demo 1: Simulated dataset
Demo 2: Real-world dataset
Demo 3: Comparison to scikit-learn
```

I wrote most of this code for the take-home portion of my DATA 558 Midterm in Spring 2017. I subsequently enhanced and cleaned the code prior to releasing it on GitHub as part of the final Polished Code Release assignment for the course.


Installation...?
