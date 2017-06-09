DATA 558 - Least Squares Regression with Elastic Net Regularization
=================

Polished code release for DATA 558, University of Washington, Spring 2017

This repository contains my own implementation of least-squares regression with elastic net regularization. I originally wrote this code for the take-home portion of my DATA 558 Midterm in Spring 2017. I subsequently enhanced the code and released it on GitHub as part of the final assignment for the course.

Elastic Net is a hybrid between the popular ℓ2 (Ridge) and ℓ1 (LASSO) regularization; the minimization problem writes as follows:

min F(β) = 1/n * ||Y − XTβ||^2 +λ α||β||1 +(1−α)||β||^2 2)
β∈Rd

where α ∈ (0, 1), with the two extremes producing Ridge and Lasso, respectively.

Because of the ℓ1 component, this function is non-differentiable and the solution must be achieved through use of the subgradient. As a result, we are unable to use gradient descent and must instead leverage coordinate descent.

I provide functions to solve this problem. Examples include:

* **Demo 1:** Simulated dataset
* **Demo 2:** Real-world dataset
* **Demo 3:** Comparison to scikit-learn

