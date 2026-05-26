# Supervised Learning: Statistical Inference & Algorithms

This repository is a comprehensive collection of notes, mathematical proofs, and at-scale applications focused on Supervised Learning. 

In this module, we explore the quest to approximate the Bayes Optimal Rule using labeled datasets. This repository bridges the gap between pure theory and applied engineering. We document the underlying statistical theory, and then immediately apply it to real-world data problems. 

We perform reproducible experiments, develop custom implementations, and utilize modern, best-in-class approaches to scale these algorithms. Whether analyzing data at a low scale for deep interpretability or deploying high-capacity models on high-scale infrastructure, our goal is to master both the mathematical bounds of these models and their practical, real-world execution.

## Decision Theory & Supervised Learning

Decision theory is a mathematical framework for making decisions. A way formulated in this framework to make optimal decisions is to minimize the amount of mistakes on average.

Making least mistakes on average can be expressed mathematically as minimizing the global expected risk (often simply referred to as risk):

$$min_f E_{X,Y}[L(f(x), Y)]$$

Where $f$ is the function (or rule) that produces a decision or an action, and L, the loss function defines the cost of an error. The global expectation of this loss is the risk.

The Bayes Optimal Decision Rule is whatever rule $f$, that makes the least mistakes on average and depends on how we define error.

If we look at point wise conditional risk (whose optimum across all points will optimize the entire function), then we can express the Bayes Optimal Rule $f(x)$, as minimizing the average loss (conditional expected risk) given a data point across all data points:

$$\min_{f(x)} E_{Y|x}[L(f(x), Y)] \quad \forall x$$

$$\text{Conditional Expected Risk} = E[L(f(x), Y)|x]$$

If we could minimize this for every possible point, we would mathematically minimize the true expected risk above. However, since we do not have access to every possible data point in the population, nor do we know the true conditional distribution for the points we *do* have, we must approximate the True Risk by averaging the loss over our finite sample of training examples. This is called Empirical Risk, and it is the best we can do with the data we have:

$$\hat{R} = \frac{1}{N}\sum_{i}^N L(f(x_i), y_i)$$


In the supervised learning setting, the decision we make is estimating the output and the risk is our prediction error. If the output is discrete, the task is called classification; if it is continuous, it is called regression.

Since we’re optimizing $\hat{R}(f)$ the sample average as a proxy for true risk, we face the danger of **overfitting**. The model may "memorize" the specific noise and idiosyncrasies of the training set rather than learning the underlying patterns. Mathematically, this results in a low Empirical Risk but a high True Risk.

### Bridging the gap between Emprical and True Risk
To bridge this gap between empriical and true risk, we try to estimate the true risk by running our trained model on a dataset it hasn't seen (out-of-sample). The simplest way to is to hold back a fraction of the data at random and run our our model against it to get our expected True Risk, however this can be brittle as the fraction of the data may be non representative of the general data distribution (we might get easier or harder examples).

To mitigate this, there is a K-fold validation strategy, which requires quick easy fitting but gives you the best estimate of the true risk by segementing the data into different folds and for each fold, training on the rest of the folds and measuring performance on the target fold. Then averaging the risk across the folds.

TODO: check with AI

## Modeling Assumptions

In both classification and regression, every model carries a specific set of assumptions—known as Inductive Bias—about how the features relate to the label.

For regression, we can actually decompose our expected risk (squared error) into three distinct terms by looking at it from an estimator's perspective (taking the expectation over samples of the full dataset $D$, and assuming additive noise $\epsilon$ with $0$ mean and finite variance $\sigma^2$):

$$
\begin{aligned}
\text{Expected Risk} &= E_{D,\epsilon}[(y - \hat{f}(x))^2] \\
&= E_{D,\epsilon}[(f(x) + \epsilon - \hat{f}(x))^2] \\
&= E_{D,\epsilon}[((f(x) - \hat{f}(x)) + \epsilon)^2] \\
&= E_{D,\epsilon}[(f(x) - \hat{f}(x))^2 + 2 \cdot \epsilon \cdot (f(x) - \hat{f}(x)) + \epsilon^2] \\
&= E_{D,\epsilon}[(f(x) - \hat{f}(x))^2] + 2 \cdot E_{D,\epsilon}[\epsilon] \cdot E_{D,\epsilon}[f(x) - \hat{f}(x)] + E_{D,\epsilon}[\epsilon^2] \\
&= E_{D,\epsilon}[(f(x) - \hat{f}(x))^2] + \sigma^2 \\
&= E_{D,\epsilon}[(f(x) - \bar{f}(x) + \bar{f}(x) - \hat{f}(x))^2] + \sigma^2 \\
&= E_{D,\epsilon}[((f(x) - \bar{f}(x)) + (\bar{f}(x) - \hat{f}(x)))^2] + \sigma^2 \\
&= E_{D,\epsilon}[(f(x) - \bar{f}(x))^2 + 2 \cdot (f(x) - \bar{f}(x))(\bar{f}(x) - \hat{f}(x)) + (\bar{f}(x) - \hat{f}(x))^2] + \sigma^2 \\
&= E_{D,\epsilon}[(f(x) - \bar{f}(x))^2] + 2 \cdot (f(x) - \bar{f}(x))E_D[\bar{f}(x) - \hat{f}(x)] + E_{D,\epsilon}[(\bar{f}(x) - \hat{f}(x))^2] + \sigma^2 \\
&= \text{Bias}^2 + \text{Variance} + \sigma^2
\end{aligned}
$$

*(Note: The cross-term cancels out because the expected value of our estimator $\hat{f}(x)$ across all datasets is exactly the average estimator $\bar{f}(x)$, making $E_D[\bar{f}(x) - \hat{f}(x)] = 0$.)*

1. **Bias Squared** $\left( E_{D,\epsilon}[(f(x) - \bar{f}(x))^2] \right)$: The expectation of the squared difference between the average estimator over the entire dataset and the true function. This measures how wrong our model is on average, regardless of which specific dataset sample it was trained on.
2. **Variance** $\left( E_{D,\epsilon}[(\bar{f}(x) - \hat{f}(x))^2] \right)$: The average squared difference between a given estimator from a specific sample of the dataset and the overall average estimator. This is a measure of how sensitive our estimator is to the specific dataset sample it is trained on.
3. **Irreducible Error** $\left( \sigma^2 \right)$: The variance of the noise intrinsic to the problem. No model can eliminate this error, regardless of its complexity.

This decomposition is illustrative of a fundamental tradeoff in model design. Models with strong assumptions tend to have higher bias but lower variance — they are less flexible, but more stable across different training samples. When those assumptions approximately hold, this stability makes them a more efficient use of available data. Models with weaker assumptions have lower bias but higher variance — they are more flexible and can capture complex patterns, but their predictions are highly sensitive to which data they were trained on. Increasing the amount of training data reduces variance, effectively 'taming' highly flexible models. However, no amount of data can ever compensate for a model that is fundamentally too rigid (high bias) to capture the true underlying pattern.

These two failure modes have names: high bias corresponds to underfitting — the model is too rigid to capture the true relationship — while high variance corresponds to overfitting — the model is too sensitive to the specific training sample. Both can be diagnosed by comparing training error to generalization error (validation error). A large gap between the two suggests overfitting; a model that performs poorly on both suggests underfitting. Understanding which problem you have is the first step to addressing it.

In the absence of a prior fundamental understanding of the structure of our data (which may lead us to pick a more data-efficient model) it’s better to choose a higher variance model that we know will overfit, and then apply regularization to it to tame it.

As an illustrative example, one technique to increase the variance of an estimator (and increase its ability to overfit) is basis expansion: let's take linear regression which can only fit hyperplanes to data, and add second-degree polynomial terms:

$$f(x) = \beta_0 + \sum_{i=1}^p \beta_i x_i + \sum_{i=1}^p \sum_{j=i+1}^p \beta_{ij} x_i x_j + \sum_{i=1}^p \beta_{ii} x_i^2$$

We have statistically added several degrees of freedom that can now be used to draw more complicated shapes for the objective function.

It’s called basis expansion because it adds vectors to the “basis” which forms the underlying space in a geometric sense.

Mathematically basis expansion is viewed as basis functions that take the input data and make a feature out of it, those m basis functions $h_m(X)$ features can then have the models applied to them like in the original space (for example linear regression):

$$f(x) = \sum_{m=1}^M \beta_m \cdot h_m(X)$$

Note that while Basis Expansion allows us to achieve a vastly better fit and reduce bias, it comes at a severe cost to model interpretability. By engineering higher-degree polynomial or interaction terms we introduce multicollinearity, we can’t look at a feature and get its isolated marginal effect on the target. It’s a trade in transparency with predictive power.

Another implication of this increased flexibility is a drastic increase in the model's sensitivity to the training data. As a fundamental rule of statistical learning, as models become more flexible, they require a significantly higher sample density to constrain them and prevent overfitting.

However, expanding our basis functions triggers a geometric paradox: as the dimensionality of our feature space grows, the volume of that space grows exponentially, which means our sampling density plummets. Even with a massive dataset, if we assume a uniform distribution, the data points become incredibly sparse and isolated. In high-dimensional space, the concept of a 'local neighborhood' collapses because almost every point is far away from every other point.

This phenomenon is known as the Curse of Dimensionality. It perfectly explains why basis expansion leads to overfitting: just when our highly flexible model needs dense data the most, the high-dimensional space isolates the data points. Unconstrained in this vast empty space, the model violently contorts its decision boundary to perfectly memorize the noise of individual samples rather than learning a generalized underlying rule.

Because of this, even if a highly flexible model perfectly maximizes the likelihood—achieving zero error on the training dataset—it is almost certainly not the optimal model. We must remember that we are operating under the Empirical Risk Minimization (ERM) framework. We are merely trying to approximate the True Risk using a limited, finite sample of data. Achieving zero Empirical Risk often just means we have perfectly memorized the sample noise. This is exactly why an out-of-sample test set is absolutely critical: it provides an unbiased estimate of the True Expected Risk on unseen data, allowing us to measure how well the model actually generalizes.

A solution to reduce the variance or flexibility is regularization (typically via a continuous penalty function) - the process of introducing supplemental information into a learning algorithm to prevent overfitting and reduce generalization error. Mathematically, it is achieved by applying a penalty to the complexity of the model, forcing the optimization process to trade off between minimizing empirical risk (fitting the training data) and restricting the effective capacity of the hypothesis space.

Regularization can also be viewed as a non-uniform prior in the Bayesian Inference framework’s Maximum A Posteriori (MAP) estimate.

The Maximum A Posteriori (MAP) estimate comes from Bayes' rule:

$$\theta_{MAP} = \text{argmax}_\theta (P(Y|\theta, X) \cdot P(\theta))$$

So we can look at the MLE as a special case of the MAP estimate where there is a uniform prior (no prior belief about theta), and therefore is equivalent to just optimizing likelihood.

So regularization actually adds in a prior belief, let's examine the prior belief that the parameters are $\sim N(0, \sigma^2)$:

$$
\begin{aligned}
\theta_{MAP} &= \text{argmax}_\theta \left( \prod_{i=1}^N P(y_i | \vec{x}_i, \theta) \cdot P(\theta) \right) \\
\theta_{MAP} &= \text{argmax}_\theta \left( \prod_{i=1}^N P(y_i | \vec{x}_i, \theta) \cdot \frac{1}{(2\pi)^{\frac{p}{2}}|\Sigma|^{\frac{1}{2}}} e^{-\frac{1}{2}\theta^T \Sigma^{-1} \theta} \right) \\
\theta_{MAP} &= \text{argmax}_\theta \left( \sum_{i=1}^N \ln(P(y_i | \vec{x}_i, \theta)) - \frac{1}{2\sigma^2} \theta^T \theta + \text{constant} \right)
\end{aligned}
$$

Where $\frac{1}{2\sigma^2}(\theta)^T(\theta)$ is being added to the quantity that is minimized, this is equivalent to $\lambda \cdot \sum_{i=1}^p \theta_i^2$ for $p$ parameters (we assume an isotropic gaussian, so the covariance matrix is $\sigma^2I$). This is called L2 regularization.

Similarly if one assumes a Laplacian prior, we can get the expression for L1 regularization.

Another common solution (and not mutually exclusive) to the curse of dimensionality is instead of expanding space and using simpler algorithms, using algorithms that assume non isotropic neighborhoods - for example decision trees that slice up the space ignoring unhelpful dimensions.

Or we could learn the representation from the data - deep learning can be viewed as learning projections that simple layers at the end of the network exploit. So complex non-linear modeling can often be reduced to finding the right spatial transformation followed by a simple, isotropic linear rule.

### Approach to specific modeling techniques:

In the subsequent supervised-learning sections, we will explore various modeling techniques. For each algorithm, our primary task will be to explicitly identify its underlying statistical assumptions and utilize mathematical mechanisms (such as Conditional Mutual Information or correlation tests) to validate their applicability to our data.

Ultimately, the goal of supervised learning remains constant: we seek a data-efficient model that provides the closest possible approximation to the Bayes Optimal Rule. We measure this success by the model's ability to minimize generalization error on unseen data, while deliberately navigating the fundamental trade-offs between predictive capacity and model interpretability.

---
## Repository Structure

This repository separates the foundational theory (covered in this document) from the specific modeling implementations. 

```text
supervised-learning/
├── README.md
├── classification/
│   └── linear_models/
│       ├── generative_models/
│       └── discriminative_models/
└── regression/
    └── linear_models/
        └── linear_regression/
```