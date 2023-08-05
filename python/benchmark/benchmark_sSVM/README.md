## Benchmark repository for smoothed SVMs

Smoothed SVMs solve the following optimization problem:
```math
  \min_{\mathbf{\beta} \in \mathbb{R}^d} \frac{C}{n} \sum_{i=1}^n V( y_i \mathbf{\beta}^\intercal \mathbf{x}_i ) + \frac{1}{2} \| \mathbf{\beta} \|_2^2
```
where $V(\cdot)$ is the smoothed hinge loss, $\mathbf{x}_i \in \mathbb{R}^d$ is a feature vector, and $y_i \in \{-1, 1\}$ is a binary label. Smoothed SVM can be rewritten as a ReHLine optimization with
```math
\mathbf{S} \leftarrow -\sqrt{C} \mathbf{y}/n, \quad
\mathbf{T} \leftarrow \sqrt{C} \mathbf{1}_n/n, \quad
\mathbf{\tau} \leftarrow \sqrt{C},
```
where $\mathbf{1}_n = (1, \cdots, 1)^\intercal$ is the $n$-length one vector, $\mathbf{X} \in \mathbb{R}^{n \times d}$ is the feature matrix, and $\mathbf{y} = (y_1, \cdots, y_n)^\intercal$ is the response vector.
### Benchmarking solvers

The solvers can be benchmarked using the command below:

```bash
benchopt run . -d classification_data
```