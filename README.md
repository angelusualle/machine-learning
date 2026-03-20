# Supervised Learning: Statistical Inference & Algorithms

This repository is a comprehensive collection of notes, mathematical proofs, and at-scale applications focused on Supervised Learning. 

Machine Learning is fundamentally about learning useful structures from data. In this module, we explore the quest to approximate the Bayes Optimal Rule using labeled datasets. However, this repository bridges the gap between pure theory and applied engineering. We rigorously document the underlying statistical theory, and then immediately apply it to real-world data problems. 

We perform reproducible experiments, develop custom implementations, and utilize modern, best-in-class approaches to scale these algorithms. Whether analyzing data at a low scale for deep interpretability or deploying high-capacity models on high-scale infrastructure, our goal is to master both the mathematical bounds of these models and their practical, real-world execution.

## Decision Theory & Supervised Learning Foundations

Decision theory is a mathematical framework for making decisions. A way formulated in this framework to make optimal decisions is to minimize the amount of mistakes on average.

Making least mistakes on average can be expressed mathematically as minimizing expected risk:

$$min_f E_{X,Y}[L(f(x), Y)]$$

Where $f$ is the function (or rule) that produces a decision or an action, and L, the loss function defines the cost of an error. The expectation of this loss is the risk.

The Bayes Optimal Decision Rule is whatever rule $f$, that makes the least mistakes on average and depends on how we define error.

If we look at point wise conditional risk (whose optimum across all points will optimize the entire function), then we can express the Bayes Optimal Rule $f(x)$, as minimizing the average loss (conditional expected risk) given a data point across all data points:

$$min_{f(x)} E_{Y|x}[L(f(x), Y)]$$
$$Conditional\ Expected\ Risk = E[L(f(x), Y)|x]$$

If we could minimize this for every possible point we would minimize the true expected risk above, however since we lack the full conditional distribution for each point, we approximate the True Risk by averaging the loss over our specific training examples. This is called Empirical Risk, and it's the best we can do with the data we have.

$$\hat{R} = \frac{1}{N}\sum_{i}^N L(f(x_i), y_i)$$

Since we’re optimizing $\hat{R}(f)$ the sample average as a proxy for true risk, we face the danger of **overfitting**. The model may "memorize" the specific noise and idiosyncrasies of the training set rather than learning the underlying patterns. Mathematically, this results in a low Empirical Risk but a high True Risk, and we usually split the dataset to train on one subset and measure performance on the set that's held out as a better estimate of the true risk.

### Empirical Risk Minimization (ERM)
If we knew the true conditional distribution for every point, we could perfectly minimize the True Expected Risk. However, in reality, we lack this full distribution. Therefore, we must approximate the True Risk by averaging the loss over our finite training sample. This is called the **Empirical Risk**:
$$R_{emp}(f) = \frac{1}{N} \sum_{i=1}^N L(f(x_i), y_i)$$

Because we are optimizing $R_{emp}(f)$ as a proxy for True Risk, we face the constant danger of **overfitting**. A highly flexible model may "memorize" the specific noise and idiosyncrasies of the training set (achieving zero Empirical Risk) rather than learning the generalized structure. This is why an out-of-sample test set is critical: it provides an unbiased estimate of the True Expected Risk on unseen data.

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