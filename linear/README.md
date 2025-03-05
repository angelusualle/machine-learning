# Linear models
Linear models are relationships expressed in a linear function (or more commonly affine).

A linear function can be defined as any function such that:
$$
f(ax + by) = a*f(x) + b*f(y), \\ \forall x, y
$$

Or that scalar multiplication and addition give you same result if the function is applied before or after.

While an affine function is any function that by adding a scalar, is linear.

$$
f(x) - c = g(x), \\ g(x) \in Linear
$$

In linear algebra form the function could be expressed as:
$$
\vec{a}\cdot\vec{x} + bias
$$
Where:
$$
a \in R^m \\
x \in R^m \\
bias \in R
$$

This bias is the scalar offset that makes it affine.

To compute an entire batch of samples one could use a matrix $\mathbf{X}$, where a row is a sample, and each column is a dimension of the sample, with an extra dimension added for the bias (always set to 1). This is equivalent to an affine function on a batch of data:

$$
\vec{y} = \mathbf{X}*\vec{b}
$$

Where:
$$
X \in R^{n,m} \\
\vec{b} \in R^m \\
\vec{y} \in R^m
$$

## Assumptions:
The data can truly be modeled this way, this this globally linear relationship, which is not often the case.
