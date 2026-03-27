# Linear Decision Boundaries

Within the broader class of classification models, this folder covers algorithms that separate data using a linear decision boundary.

Mathematically, a decision boundary is the exact region in the feature space where the model is perfectly unsure—where the inferred classification score (rank preserving transformation of posterior) is equal between two classes:

$$
\mathcal{B}_{ij} = \{ \vec{x} \mid g_i(\vec{x}) = g_j(\vec{x}) \}
$$

Where $g_i$ is the classification rule for class $i$. 

We call these linear boundaries because they form flat hyperplanes through space, although the functions themselves are technically **affine transformations**. An affine function transforms the input vector via scaling (the weights) and translation (the bias): $g(\vec{x}) = \vec{w} \cdot \vec{x} + b$.


Ultimately, an algorithm is classified as a 'linear model' if, after resolving the point of equality between classes, the resulting mathematical boundary simplifies down to a purely affine combination of the input features.
