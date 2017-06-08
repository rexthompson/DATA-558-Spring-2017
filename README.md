# DATA-558-Spring-2017
Polished code release for UW MSDS DATA 558, Spring 2017


Derivation
=======================================

First I write the objective function as $F(\beta) = f(\beta) + g(\beta)$, where

$$
f(\beta) = \frac{1}{n}\sum_{i=1}^n\big(y_i - x_i^T\beta\big)^2 + \lambda(1-\alpha)||\beta||_2^2
$$

and

$$
g(\beta) = \lambda\alpha||\beta||_1
$$

$f(\beta)$ is differentiable, so I follow the same steps as my answer in the in-class portion of the midterm to find the derivative (i.e. gradient) of $f(\beta)$:

$$
\frac{\delta f}{\delta\beta} = -\frac{2}{n}X(Y-X^T\beta) + 2\lambda(1-\alpha)\beta
$$

As shown above, the only difference between this and prior derivations is the $(1-\alpha)$ term. I then follow the same process as the Lasso derivation notes to write the equation as the partial derivative with respect to $\beta_j$ only, and then I simplify:

$$
\begin{aligned}
\frac{\delta f}{\delta\beta_j} &= -\frac{2}{n}x_j(Y-X_{-j}^T\beta_{-j} - x_j^T\beta_j) + 2\lambda(1-\alpha)\beta_j \\
      &= -\frac{2}{n}x_j(Y-X_{-j}^T\beta_{-j}) + \frac{2}{n}x_jx_j^T\beta_j + 2\lambda(1-\alpha)\beta_j \\
      &= -\frac{2}{n}x_jR^{-j} + \frac{2}{n}\beta_jz_j + 2\lambda(1-\alpha)\beta_j \\
      &= -\frac{2}{n}x_jR^{-j} + \frac{2}{n}\beta_jz_j + \frac{2}{n}\lambda(1-\alpha)n\beta_j \\
      &= -\frac{2}{n}x_jR^{-j} + \frac{2}{n}\big(z_j + \lambda(1-\alpha)n\big)\beta_j
\end{aligned}
$$

where

$$
R^{-j} = (Y - X_{-j}^T\beta_{-j})
$$

and

$$
z_j = x_jx_j^T = ||x_j||_2^2 = \sum_{i=1}^n x_{ij}^2
$$

$g(\beta)$ is not differentiable, so I follow the same procedure as the Lasso derivation notes to find the subdifferential of $g(\beta)$:

$$
\frac{\delta}{\delta\beta_j} = \delta\{|\beta_1| + \dots + |\beta_j| + \dots + |\beta_d|\} = \delta\{|\beta_j|\}
$$

Like in the notes, I can say that $s_j \in \delta|\beta_j| \iff$ for all $j = 1, \dots, d$:

$$
s_j = \begin{cases}
sign(\beta_j) &\text{if } \beta \ne 0 \\
[-1, +1] &\text{if } \beta = 0
\end{cases}
$$

Thus, there exists some $s_j$ such that:

$$
-\frac{2}{n}x_jR^{-j} + \frac{2}{n}\big(z_j + \lambda(1-\alpha)n\big)\beta_j + \lambda\alpha s_j = 0
$$

Using this result, I break the equation above into 3 cases:

#### Case I: $\beta_j > 0$

$$
-\frac{2}{n}x_jR^{-j} + \frac{2}{n}\big(z_j + \lambda(1-\alpha)n\big)\beta_j = - \lambda\alpha \\
 \frac{2}{n}\big(z_j + \lambda(1-\alpha)n\big)\beta_j = \frac{2}{n}x_jR^{-j} - \lambda\alpha
$$

and so we have

$$
\beta_j = \frac{\frac{2}{n}x_jR^{-j} - \lambda\alpha}{\frac{2}{n}\big(z_j + \lambda(1-\alpha)n\big)}
$$

Since this holds when $\beta_j > 0$, I can also write that it holds when:

$$
\begin{aligned}
\frac{\frac{2}{n}x_jR^{-j} - \lambda\alpha}{\frac{2}{n}\big(z_j + \lambda(1-\alpha)n\big)} &> 0 \\
\frac{2}{n}x_jR^{-j} - \lambda\alpha &> 0 \\
\frac{2}{n}x_jR^{-j} &> \lambda\alpha \\
\end{aligned}
$$

#### Case II: $\beta_j < 0$

$$
-\frac{2}{n}x_jR^{-j} + \frac{2}{n}\big(z_j + \lambda(1-\alpha)n\big)\beta_j = \lambda\alpha \\
 \frac{2}{n}\big(z_j + \lambda(1-\alpha)n\big)\beta_j = \frac{2}{n}x_jR^{-j} + \lambda\alpha
$$

and so we have

$$
\beta_j = \frac{\frac{2}{n}x_jR^{-j} + \lambda\alpha}{\frac{2}{n}\big(z_j + \lambda(1-\alpha)n\big)}
$$

Since this holds when $\beta_j < 0$, I can also write that it holds when:

$$
\begin{aligned}
\frac{\frac{2}{n}x_jR^{-j} + \lambda\alpha}{\frac{2}{n}\big(z_j + \lambda(1-\alpha)n\big)} &< 0 \\
\frac{2}{n}x_jR^{-j} + \lambda\alpha &< 0 \\
\frac{2}{n}x_jR^{-j} &< -\lambda\alpha \\
\end{aligned}
$$

#### Case III: $\beta_j = 0$

$$
\begin{aligned}
-\frac{2}{n}x_jR^{-j} + \frac{2}{n}\big(z_j + \lambda(1-\alpha)n\big)\beta_j &= -\lambda\alpha[-1, +1] \\
\frac{2}{n}x_jR^{-j} - \frac{2}{n}\big(z_j + \lambda(1-\alpha)n\big)\beta_j &= \lambda\alpha[-1, +1] \\
\frac{2}{n}x_jR^{-j} - \frac{2}{n}\big(z_j + \lambda(1-\alpha)n\big)\beta_j &= [-\lambda\alpha, +\lambda\alpha] \\
\frac{2}{n}x_jR^{-j} &= [-\lambda\alpha, +\lambda\alpha] \\
\Big|\frac{2}{n}x_jR^{-j}\Big| &\le \lambda\alpha \\
\end{aligned}
$$

#### Summary

I can write the solution to the minimization problem more consicely as follows:

$$
\beta_j = \begin{cases}
(c_j + \lambda\alpha)/w_j &\text{if } c_j < -\lambda\alpha \\
0 &\text{if } |c_j| \le \lambda\alpha \\
(c_j - \lambda\alpha)/w_j &\text{if } c_j > \lambda\alpha \\
\end{cases}
$$

where

$$
c_j = \frac{2}{n}x_jR^{-j}
$$

and

$$
w_j = \frac{2}{n}\big(z_j + \lambda(1-\alpha)n\big)
$$

and where $R^{-j}$ and $z_j$ are as defined above.