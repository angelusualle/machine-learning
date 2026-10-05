# Regression
Regression is where we are trying to predict a continous value of a data sample. The decision is what value we should predict.

For regression we will use a simple intuitive loss function of squared error:

$$
L(y, \hat{y}) = (y - \hat{y})^2
$$

This means that we take how far off we are and square that, into a simple positive metric, though it tends to punish outliers more heavily, than other loss functions. The average loss or risk is then the mean squared error.

Defining loss this way has a benefit, as we shall see shortly, under additive 0 meaned Gaussian noise, the bayes optimal rule is the easily understood conditional mean, that is for common regression problems (with this normal noise), the best decision is to output the conditional mean.

It's godo to mention that there are others like absolute error, that lead to an bayes optimal rule of outputing the median, however the absolute error is not differentiable, which is more complex to optimize (veresus a gradient based parameter optimization).

