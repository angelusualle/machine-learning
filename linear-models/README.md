# Linear models
Linear models are relationships expressed in a linear function (or more commonly affine).

Assume here * is matrix multiplication.

A linear function can be defined as any function such that:

$$ f(ax + by) = a*f(x) + b*f(y),  \newline \forall x, y $$

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


$$
X \in R^{n,m} \\
\vec{b} \in R^m \\
\vec{y} \in R^m
$$

## Assumptions:
The data can truly be modeled this way, this this globally linear relationship (as described above).

## Additional modifications:
The data itself can be transformed prior so the a linear model is more applicable.

When encoding categorical variables for linear regression, using a method often called 'dummy encoding' (or 'one-hot encoding with a dropped category') is essential. This involves creating k-1 binary (0/1) features for a categorical variable with k unique categories. This k-1 approach specifically avoids the 'dummy variable trap.' The trap occurs if you create k binary features for k categories (full one-hot encoding without dropping one), because then one of those binary columns becomes perfectly predictable from the others (e.g., if a value isn't in category A or B, and there are only 3 categories, it must be in C), leading to perfect multicollinearity. By using k-1 features, this perfect multicollinearity is broken, making the model suitable for linear regression. The coefficients for these k-1 dummy variables are then interpreted as the average effect or difference in the outcome variable compared to the baseline (the omitted category), holding all other variables constant.

Multiplying features to capture interactions or polynomial expansions that multiply features with themselves.