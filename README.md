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

### Bridging the Gap: Training, Validation, and Testing
To bridge the gap between Empirical Risk (training error) and True Risk (generalization error), we must evaluate our model on data it has never seen. However, a critical distinction must be made between evaluating a model to *tune* it, and evaluating a model to *audit* it.

**The Model Selection Trap (Data Leakage)**
If we simply split our data into a "Train" and "Test" set, and repeatedly adjust our hyperparameters (like polynomial degree or regularization strength) to maximize the score on the Test set, we introduce **optimism bias**. By picking the hyperparameters that perform best on that specific Test set, information about the Test data "leaks" into the model. The Test set is effectively acting as training data for the hyperparameters, meaning its error rate is no longer an unbiased estimate of True Risk.

To prevent this, we must divide our data into three distinct functional roles:
1. **The Training Set:** Used by the optimization algorithm to learn the model's weights/parameters.
2. **The Validation Set:** Used by the modeler to evaluate different algorithms, tune hyperparameters, and perform early stopping.
3. **The Test Set:** A locked vault of data used exactly *once* at the very end of the project to provide a mathematically unbiased estimate of True Risk.

#### Validation Strategies
When constructing the Validation Set, we face a tradeoff between computational cost and statistical variance.

* **The Static Validation Split:** We divide our accessible data (excluding the locked Test Set) into a static Train and Validation fraction. While computationally cheap, this single-split approach has high variance. It can be incredibly brittle because the random fraction may not be perfectly representative of the general data distribution. 
* **K-Fold Cross-Validation:** To mitigate variance, we segment the accessible training data into $K$ mutually exclusive subsets (folds). For each fold $k$, we train the model from scratch on the remaining $K-1$ folds and measure its performance on the target fold $k$. We then average the validation risk across all $K$ folds. This maximizes data efficiency and provides a highly stable evaluation metric for hyperparameter tuning. While it comes at a computational cost (requiring $K$ separate model fits—infeasible for massive deep learning models), it is the gold standard for model selection.

#### The Standard Operating Procedure (SOP): From Validation to Production
Cross-validation is strictly an evaluation procedure for the Validation phase. The standard engineering pipeline proceeds as follows:

1. **The Global Holdout:** Split the full dataset into a Working Set (e.g., 80%) and a final Test Set (20%). Lock the Test Set away.
2. **The Scouts (Validation):** Perform K-Fold Cross-Validation exclusively on the 80% Working Set. The $K$ models trained here act merely as "scouts" to estimate how a specific set of hyperparameters will generalize.
3. **Hyperparameter Tuning:** Adjust hyperparameters and repeat Step 2 until the Cross-Validation risk is minimized. Once the optimal configuration is found, discard all $K$ temporary scout models.
4. **The Final Train:** Using those winning hyperparameters, train **one single production model** on 100% of the Working Set. A fundamental rule of statistical learning is that more data reduces variance; giving the final model the entire Working Set yields the most stable decision boundary.
5. **The Final Audit (Testing):** Evaluate this newly trained production model on the locked Test Set from Step 1. This provides the final, unbiased estimate of generalization error to report to stakeholders.

K-Fold evaluates the model-building **procedure** (Algorithm + Hyperparameters + Data Size). By averaging the $K$ models, you get an unbiased estimate of how well your *learning strategy* generalizes. Once proven, you apply that exact strategy to 100% of the training data.

 **The Small Data Exception:** While the 3-way split is the industry standard, it breaks down when data is severely limited (e.g., a medical study with 150 patients) or when there is absolutely zero model selection occurring. If you are not tuning hyperparameters, selecting features, or comparing algorithms, there is no selection process to introduce optimism bias. In these regimes, we drop the Test Set, rely exclusively on K-Fold Cross-Validation for our official unbiased estimate of True Risk, and deploy the final model trained on 100% of the data.
 
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

We can look at learning algorithms as optimizing a Likelihood function (Maximum Likelihood Framework), that is that the best estimate for parameters of a model are those that maximize the likelihood of seeing the data.

To understand from a theoretical view why likelihood works well, we can look at two proven facts: the Cramér-Rao theorem, and the fact that the likelihood-optimized estimator converges to the bound provided by that theorem. 

Information is generally defined as the measure of a reduction of uncertainty. **Fisher Information** ($\mathcal{I}$) is formally defined as the variance of the score (the gradient) of the log-likelihood:

$$\mathcal{I}(\theta) = \text{Var}\left[\frac{\partial L}{\partial \theta}\right] = -E\left[\frac{\partial^2 L}{\partial \theta^2}\right]$$

Equality can be shown with calculus (Second Bartlett Identity).

Because the gradient is a measure of the log-likelihood's sensitivity to parameter changes, its variance tells us how fast that sensitivity changes.

A likelihood function with high curvature means the gradient changes rapidly around the true value of $\theta$. This indicates that the data is highly informative and allows for a precise estimate. A flat likelihood, where the gradient is stable and doesn't change much, provides little information.

And the Cramér-Rao theorem states:

$$\text{Var}(\hat{\theta}) \geq \frac{1}{\mathcal{I}(\theta)}$$

That is, the variance of any estimator is lower bounded by the inverse of information. It has been proven that the likelihood-optimized estimator converges to that bound, so it wastes no information, and is therefore optimal.

On top of MLE, regularization can also be viewed as a non-uniform prior in the Bayesian Inference framework’s Maximum A Posteriori (MAP) estimate.

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

In the subsequent supervised-learning sections, we will explore various modeling techniques. For each algorithm, our primary tasks will be to explicitly identify its underlying statistical assumptions and utilize mathematical mechanisms to validate their applicability to our data for interpertability, then show how to navigate the bias variance trade off to opitimize generalization performance.

Ultimately, the goal of supervised learning remains constant: we seek a data-efficient model that provides the closest possible approximation to the Bayes Optimal Rule.
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