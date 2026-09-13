# How BoostForge Works

BoostForge implements the essential mathematics behind an **XGBoost-style
second-order gradient boosting system** without calling XGBoost, LightGBM,
CatBoost, or scikit-learn estimators.

## 1. Additive boosting model

The raw prediction after `t` trees is

\[
\hat y_i^{(t)} = \hat y_i^{(t-1)} + \eta f_t(x_i)
\]

where `eta` is the learning rate and `f_t` is the new tree.

## 2. Second-order Taylor approximation

Instead of fitting residuals with ordinary least squares, BoostForge expands the
objective around the current prediction:

\[
\mathcal{L}^{(t)}
\approx
\sum_i
\left[
g_i f_t(x_i)
+
\frac{1}{2}h_i f_t(x_i)^2
\right]
+
\Omega(f_t)
\]

with first derivatives `g_i` and second derivatives `h_i`.

For binary logistic loss:

\[
g_i = p_i-y_i, \qquad
h_i = p_i(1-p_i)
\]

For squared-error regression:

\[
g_i = \hat y_i-y_i, \qquad
h_i = 1
\]

## 3. Optimal leaf weight

For a leaf containing samples `I`, define

\[
G=\sum_{i\in I}g_i,\qquad H=\sum_{i\in I}h_i
\]

The regularized Newton step stored in that leaf is

\[
w^*=-\frac{G}{H+\lambda}
\]

## 4. Split gain

A candidate split is accepted when its gain is positive:

\[
Gain =
\frac{1}{2}
\left[
\frac{G_L^2}{H_L+\lambda}
+
\frac{G_R^2}{H_R+\lambda}
-
\frac{G^2}{H+\lambda}
\right]
-\gamma
\]

This means the tree is not minimizing plain variance. It is directly optimizing
the current boosting objective.

## 5. Regularization implemented

BoostForge includes:

- `reg_lambda`: L2 regularization on leaf weights
- `gamma`: minimum gain required before a split is useful
- `min_child_weight`: minimum Hessian mass in a child
- `max_depth`: structural regularization
- `subsample`: stochastic row sampling
- `colsample_bytree`: stochastic feature sampling

## 6. Complexity

The educational splitter evaluates multiple thresholds for every candidate
feature. With `max_bins=None`, the method approaches exact greedy split search.
With finite `max_bins`, quantile thresholds reduce the candidate count and make
experiments significantly faster.

The project prioritizes **clarity of the algorithm** over production-level speed.
