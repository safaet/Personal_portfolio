import { useState, useMemo } from "react";

const CATEGORIES = {
  math: { label: "Math Foundations", color: "#a78bfa", bg: "#1e1a2e", topics: ["linear-algebra","calculus","probability","statistics"] },
  core: { label: "Core ML", color: "#34d399", bg: "#0f2018", topics: ["machine-learning","deep-learning"] },
  applied: { label: "Applied AI", color: "#fb923c", bg: "#1f1208", topics: ["nlp","computer-vision","llm"] },
  systems: { label: "AI Systems", color: "#38bdf8", bg: "#071822", topics: ["rag","agentic-ai","mlops"] }
};

const TOPICS = [
  {
    id: "linear-algebra",
    name: "Linear Algebra",
    cat: "math",
    hours: 40,
    depth: "Deep",
    why: "Embeddings ARE vectors. Transformers ARE matrix multiplications. LoRA IS low-rank matrix decomposition. PCA IS SVD. You cannot understand any modern AI paper without LA fluency.",
    subtopics: [
      "Vectors: dot product, norm, angle, projection onto subspace",
      "Matrices: multiplication, transpose, inverse, determinant, rank",
      "Linear transformations: geometric intuition (rotation, scaling, shear)",
      "Eigenvalues & eigenvectors: Av = λv, power iteration, spectral theorem",
      "SVD: W = UΣVᵀ — the most important decomposition in ML",
      "PCA: principal components = right singular vectors of centered data",
      "Matrix norms: Frobenius, spectral, nuclear (used in regularization)",
      "Positive definite matrices: Gram matrices, covariance matrices",
      "Sparse matrices: COO, CSR formats (critical for GNNs, embeddings)",
      "Tensor operations: broadcasting rules, Einstein summation (einsum)"
    ],
    exercises: [
      "Implement matrix multiply in pure Python (no NumPy) — understand cache-friendly row-major access",
      "Derive: why does PCA = SVD on centered data? Prove on paper starting from covariance matrix C = XᵀX/(n-1)",
      "Implement power iteration to find top eigenvector without eigendecomposition library",
      "Compute SVD of a 3×3 matrix by hand using the characteristic polynomial",
      "Use np.einsum to implement: batch matrix multiply, outer product, trace, bilinear forms",
      "Implement PageRank using repeated matrix-vector multiplication — eigenvalue interpretation",
      "Prove: for unit vectors, cosine similarity = dot product. Build intuition for embedding spaces"
    ],
    project: "Image compression with PCA: load a grayscale image (512×512), compute SVD, reconstruct with k=5,20,50,100 singular values, plot compression ratio vs reconstruction PSNR. Write a 200-word analysis of the tradeoff.",
    interview_qs: [
      "Explain SVD and its connection to PCA. When would you use each?",
      "Why does LoRA work? What assumption does it make about weight updates?",
      "What does it mean for a matrix to be positive semi-definite? When does this appear in ML?",
      "How does attention use matrix operations? Trace dimensions for a single-head attention block.",
      "What is the rank of a matrix and why does it matter for model compression?",
      "Explain matrix broadcasting in PyTorch with an example showing a shape error you'd make."
    ],
    resources: [
      { name: "3Blue1Brown — Essence of Linear Algebra", url: "https://www.3blue1brown.com/topics/linear-algebra", type: "video", priority: 1 },
      { name: "MIT 18.06 — Gilbert Strang (free lectures)", url: "https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/", type: "course", priority: 1 },
      { name: "Mathematics for Machine Learning (Deisenroth)", url: "https://mml-book.github.io/", type: "book", priority: 2 },
      { name: "fast.ai Numerical Linear Algebra", url: "https://github.com/fastai/numerical-linear-algebra", type: "course", priority: 2 },
      { name: "Strang — Introduction to Linear Algebra (book)", url: "https://math.mit.edu/~gs/linearalgebra/", type: "book", priority: 3 }
    ],
    code: `import numpy as np

# SVD decomposition — most important LA tool in ML
A = np.random.randn(100, 50)  # e.g., 100 samples, 50 features
U, S, Vt = np.linalg.svd(A, full_matrices=False)
# U: (100,50) — left singular vectors (sample space)
# S: (50,)    — singular values (importance ranking)
# Vt:(50,50)  — right singular vectors (feature space, PCA directions)

# PCA via SVD on centered data
X = np.random.randn(200, 50)
X_centered = X - X.mean(axis=0)
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
# PCA components = rows of Vt
X_reduced = X_centered @ Vt[:10].T  # project to 10 dims

# LoRA insight: weight update ΔW = B @ A
# B ∈ R^{d×r}, A ∈ R^{r×k} — both low rank r << min(d,k)
# Assumption: ΔW lies in a low-dimensional subspace of weight space
# This is exactly what SVD tells us: most "useful" directions
# of a matrix are captured by top-k singular vectors
d, k, r = 4096, 4096, 16
B = np.random.randn(d, r) * 0.02  # init near zero
A = np.random.randn(r, k)          # init random
delta_W = B @ A  # rank-r matrix — only 2*4096*16 = 131K params vs 16.7M`
  },
  {
    id: "calculus",
    name: "Calculus",
    cat: "math",
    hours: 30,
    depth: "Medium",
    why: "Backpropagation IS the chain rule applied to computational graphs. Every optimizer — SGD, Adam, RMSProp — moves in the direction of the negative gradient. You need calculus to debug training failures.",
    subtopics: [
      "Derivatives: power rule, product rule, quotient rule, chain rule",
      "Partial derivatives: ∂f/∂xᵢ holding other variables constant",
      "Gradient: ∇f = [∂f/∂x₁, ..., ∂f/∂xₙ] — direction of steepest ascent",
      "Jacobian: ∂y/∂x matrix for vector-valued functions",
      "Hessian: ∂²f/∂xᵢ∂xⱼ — second-order curvature (used in Newton's method)",
      "Chain rule for composition: ∂L/∂x = (∂L/∂y)(∂y/∂x)",
      "Computational graphs: forward pass (compute values) + backward pass (compute gradients)",
      "Automatic differentiation: reverse-mode AD (what PyTorch uses)",
      "Taylor series: f(x+ε) ≈ f(x) + εf'(x) — basis for gradient descent intuition",
      "Convexity: local minimum = global minimum for convex functions"
    ],
    exercises: [
      "Derive backprop for a 2-layer network: L = CE(softmax(W₂ · ReLU(W₁ · x + b₁) + b₂), y). Compute ∂L/∂W₁ step by step.",
      "Implement micrograd: a scalar-valued autograd engine. Support +, *, ReLU, exp. Build a 2-layer MLP on XOR.",
      "Show why vanishing gradients occur in sigmoid networks: |σ'(x)| ≤ 0.25 always. What's the max gradient through 10 sigmoid layers?",
      "Derive the Adam update rule from first principles: m = β₁m + (1-β₁)g, v = β₂v + (1-β₂)g². What does bias correction fix?",
      "Compute the Jacobian of softmax analytically: ∂sᵢ/∂zⱼ = sᵢ(δᵢⱼ - sⱼ)"
    ],
    project: "Build a scalar autograd engine (Karpathy's micrograd style) that supports the 10 most common operations used in neural nets. Train a 2-layer MLP on a toy 2D classification problem. Visualize the decision boundary and loss curve.",
    interview_qs: [
      "Explain backpropagation mathematically — what is it computing?",
      "Why do vanishing gradients happen in deep sigmoid networks? How does ReLU fix it partially?",
      "What does the learning rate control geometrically? What happens if it's too large?",
      "Explain the Adam optimizer. What problem does it solve that SGD doesn't?",
      "What is gradient clipping and when do you use it?",
      "Why is the Hessian rarely used in deep learning despite being theoretically optimal?"
    ],
    resources: [
      { name: "3Blue1Brown — Essence of Calculus", url: "https://www.3blue1brown.com/topics/calculus", type: "video", priority: 1 },
      { name: "Karpathy micrograd (build backprop from scratch)", url: "https://github.com/karpathy/micrograd", type: "code", priority: 1 },
      { name: "The Matrix Calculus You Need for Deep Learning (Parr & Howard)", url: "https://explained.ai/matrix-calculus/", type: "paper", priority: 1 },
      { name: "Khan Academy — Multivariable Calculus", url: "https://www.khanacademy.org/math/multivariable-calculus", type: "course", priority: 2 }
    ],
    code: `# Implement reverse-mode AD for scalar values (micrograd-style)
import math

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            # Chain rule: ∂L/∂self = ∂L/∂out * ∂out/∂self = out.grad * other.data
            self.grad  += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'relu')
        def _backward():
            # ∂ReLU/∂x = 1 if x > 0 else 0
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()  # propagate gradients backward`
  },
  {
    id: "probability",
    name: "Probability",
    cat: "math",
    hours: 35,
    depth: "Deep",
    why: "Cross-entropy loss IS negative log-likelihood. LLM temperature sampling IS a categorical distribution. VAE uses KL divergence. Bayesian inference IS the mathematical foundation of learning from data.",
    subtopics: [
      "Random variables: discrete vs continuous, PMF vs PDF vs CDF",
      "Key distributions: Bernoulli, Categorical, Gaussian, Poisson, Beta, Dirichlet",
      "Expectation E[X], variance Var(X), covariance Cov(X,Y)",
      "Conditional probability: P(A|B) = P(A∩B)/P(B)",
      "Bayes' theorem: P(θ|D) ∝ P(D|θ)P(θ) — posterior, likelihood, prior",
      "MLE: θ* = argmax_θ Σ log p(xᵢ|θ) — why we minimize cross-entropy",
      "MAP: add log prior to MLE objective — connection to L2 regularization",
      "KL divergence: KL(P||Q) = Σ P log(P/Q) — asymmetric, always ≥ 0",
      "Entropy: H(X) = -Σ p(x) log p(x) — average surprise",
      "Information theory: mutual information I(X;Y) = H(X) - H(X|Y)"
    ],
    exercises: [
      "Derive: cross-entropy loss = negative log-likelihood under categorical distribution. Show every step.",
      "Prove KL(P||Q) ≥ 0 using Jensen's inequality. When does equality hold?",
      "Derive: L2 regularization (weight decay) = MAP estimation with Gaussian prior. Show the math.",
      "Compute analytically: KL divergence between N(μ₁,σ₁²) and N(μ₂,σ₂²).",
      "Simulate: sample 10000 points from a mixture of 3 Gaussians. Fit with EM algorithm from scratch.",
      "Explain temperature sampling: p_T(x) = softmax(logits/T). Show: T→0 = greedy, T→∞ = uniform."
    ],
    project: "Implement Naive Bayes classifier from scratch for text classification. Use Laplace smoothing. Evaluate on a Bengali/English news dataset. Compare to sklearn's NaiveBayes — results should be identical.",
    interview_qs: [
      "Why do we minimize cross-entropy for classification? Derive its probabilistic interpretation.",
      "Explain KL divergence intuitively. Why is it asymmetric?",
      "What is the connection between L2 regularization and a Gaussian prior?",
      "Explain temperature in LLM sampling. What does T=0 give you? T=2?",
      "What is the bias-variance tradeoff in probabilistic terms?",
      "Explain why the ELBO is used in VAE training. What two terms does it balance?"
    ],
    resources: [
      { name: "Probability Theory: The Logic of Science (Jaynes — free PDF)", url: "https://bayes.wustl.edu/etj/prob/book.pdf", type: "book", priority: 1 },
      { name: "StatQuest — Probability & Statistics (YouTube)", url: "https://www.youtube.com/@statquest", type: "video", priority: 1 },
      { name: "Seeing Theory (visual probability)", url: "https://seeing-theory.brown.edu/", type: "interactive", priority: 2 },
      { name: "Deep Learning Book Ch.3 — Probability (Goodfellow et al.)", url: "https://www.deeplearningbook.org/contents/prob.html", type: "book", priority: 1 }
    ],
    code: `import numpy as np
import scipy.stats as stats

# Cross-entropy = negative log-likelihood under Categorical distribution
def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    CE(p, q) = -Σ p(x) log q(x)
    For one-hot labels: CE = -log q(correct_class)
    This is MLE: maximize log p(D|θ) = minimize negative log p(D|θ)
    """
    eps = 1e-12
    return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))

# KL divergence — asymmetric "distance" between distributions
def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(P||Q) = Σ P(x) log(P(x)/Q(x)) ≥ 0 always"""
    eps = 1e-12
    return np.sum(p * np.log((p + eps) / (q + eps)))

# Temperature sampling from logits
def temperature_sample(logits: np.ndarray, temperature: float) -> int:
    """
    T → 0: argmax (deterministic/greedy)
    T = 1: standard categorical sampling
    T → ∞: uniform random (maximum entropy)
    """
    scaled = logits / max(temperature, 1e-6)
    probs = np.exp(scaled - scaled.max())  # numerical stability
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)

# Bayes theorem: P(θ|D) ∝ P(D|θ) * P(θ)
# L2 regularization derivation:
# log P(θ|D) = log P(D|θ) + log P(θ) + const
# If P(θ) = N(0, σ²), then log P(θ) = -||θ||² / (2σ²) + const
# → maximizing posterior = minimizing CE loss + λ||θ||²
# Therefore: L2 reg = MAP with Gaussian prior (λ = 1/(2σ²))`
  },
  {
    id: "statistics",
    name: "Statistics",
    cat: "math",
    hours: 25,
    depth: "Medium",
    why: "Is your new model actually better, or did you just get lucky on the test set? Hypothesis testing answers this. A/B testing model versions, computing confidence intervals on metrics — all statistics.",
    subtopics: [
      "Descriptive stats: mean, median, mode, variance, std, skewness, kurtosis",
      "Sampling distributions: central limit theorem (CLT), law of large numbers",
      "Hypothesis testing: null hypothesis, p-value, Type I/II errors, power",
      "t-test (paired, independent), chi-square test, ANOVA",
      "Confidence intervals: CI = x̄ ± z*(σ/√n)",
      "A/B testing: sample size calculation, multiple testing correction (Bonferroni, FDR)",
      "Bootstrap: resampling-based uncertainty estimation",
      "Bias-variance decomposition: E[(y-ŷ)²] = Bias² + Variance + Noise",
      "Correlation vs causation: Simpson's paradox, confounders",
      "Nonparametric tests: Mann-Whitney, Wilcoxon, Spearman correlation"
    ],
    exercises: [
      "Run a paired t-test on two model eval runs: are the improvements statistically significant at α=0.05?",
      "Compute bootstrap confidence intervals for F1 score without assuming any distribution",
      "Simulate Type I error inflation with 20 simultaneous tests. Apply Bonferroni correction.",
      "Compute required sample size for an A/B test: 80% power, α=0.05, effect size=5% relative improvement",
      "Manually verify bias-variance decomposition on a polynomial regression problem"
    ],
    project: "Statistical evaluation framework for ML experiments: given two model eval results (each with 50 test-set scores), compute paired t-test, bootstrap 95% CI, effect size (Cohen's d), and produce a formatted report. Use this on any two of your existing projects.",
    interview_qs: [
      "How do you determine if model A is significantly better than model B?",
      "What is the bias-variance tradeoff? How does it manifest in practice?",
      "Explain the bootstrap method. When would you use it over a parametric test?",
      "What is multiple testing correction and when do you need it in ML?",
      "What's wrong with selecting a model based on its highest test set performance across many runs?"
    ],
    resources: [
      { name: "StatQuest — Statistics (YouTube, Josh Starmer)", url: "https://www.youtube.com/@statquest", type: "video", priority: 1 },
      { name: "Think Stats 2e (free online, Python-based)", url: "https://greenteapress.com/wp/think-stats-2e/", type: "book", priority: 1 },
      { name: "Statistics for Machine Learning (Jason Brownlee)", url: "https://machinelearningmastery.com/statistics_for_machine_learning/", type: "book", priority: 2 }
    ],
    code: `import numpy as np
from scipy import stats

# Paired t-test: are improvements statistically significant?
def compare_models(scores_a: np.ndarray, scores_b: np.ndarray, alpha=0.05):
    """
    H₀: models are equivalent (mean difference = 0)
    H₁: model B is better
    """
    diff = scores_b - scores_a
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    cohen_d = diff.mean() / diff.std()
    
    print(f"Mean improvement: {diff.mean():.4f} ± {diff.std():.4f}")
    print(f"t-statistic: {t_stat:.3f}, p-value: {p_value:.4f}")
    print(f"Effect size (Cohen's d): {cohen_d:.3f}")
    print(f"Significant at α={alpha}: {p_value < alpha}")
    return p_value < alpha

# Bootstrap confidence intervals (model-free)
def bootstrap_ci(metric_scores: np.ndarray, n_bootstrap=10000, ci=0.95):
    """No distribution assumptions needed."""
    boot_means = [
        np.mean(np.random.choice(metric_scores, len(metric_scores)))
        for _ in range(n_bootstrap)
    ]
    alpha = (1 - ci) / 2
    lower = np.percentile(boot_means, alpha * 100)
    upper = np.percentile(boot_means, (1 - alpha) * 100)
    return lower, upper

# Bias-variance decomposition (conceptual)
# E[(y - ŷ)²] = Bias(ŷ)² + Var(ŷ) + σ²_noise
# High bias = underfitting, high variance = overfitting
# Key insight: training loss ≪ val loss → high variance (overfit)`
  },
  {
    id: "machine-learning",
    name: "Machine Learning",
    cat: "core",
    hours: 65,
    depth: "Deep",
    why: "Classical ML is the foundation every senior ML engineer is tested on. XGBoost still wins tabular competitions. Understanding bias-variance, regularization, and error analysis applies to ALL models including LLMs.",
    subtopics: [
      "Linear regression: OLS, gradient descent derivation, closed-form (Xᵀx)⁻¹Xᵀy",
      "Logistic regression: sigmoid + cross-entropy, MLE derivation",
      "Decision trees: information gain, Gini impurity, pruning, depth vs bias-variance",
      "Ensemble methods: Bagging (random forests), Boosting (gradient boosting, XGBoost, LightGBM)",
      "SVMs: margin maximization, kernel trick, soft-margin formulation",
      "Unsupervised: k-means (EM interpretation), DBSCAN, hierarchical clustering",
      "Dimensionality reduction: PCA, t-SNE, UMAP (for visualization)",
      "Regularization: L1 (Lasso, sparsity), L2 (Ridge, weight decay), ElasticNet, Dropout",
      "Model evaluation: precision, recall, F1, AUC-ROC, calibration, confusion matrix",
      "Bias-variance tradeoff, learning curves, error analysis workflow",
      "Feature engineering, feature selection, handling imbalanced data",
      "Cross-validation: k-fold, stratified, time-series splits"
    ],
    exercises: [
      "Implement linear regression from scratch: gradient descent AND closed-form. Verify they converge to the same solution.",
      "Implement decision tree with information gain. Grow to depth 5 on Iris dataset. Compare to sklearn.",
      "Debug a gradient boosting model: learning curves show high variance. Tune max_depth, n_estimators, subsample.",
      "Implement k-means from scratch. Show it converges to different solutions with different initializations. Fix with k-means++.",
      "Build complete error analysis: collect misclassified examples, categorize failure modes, propose targeted fixes.",
      "Implement cross-validation without data leakage for time-series data."
    ],
    project: "End-to-end Kaggle tabular competition: data exploration → feature engineering (target encoding, date features) → XGBoost baseline → LightGBM → stacking ensemble → submission. Document each step with metric improvement. Target: top 30% on a competition.",
    interview_qs: [
      "When would you use XGBoost vs a neural network for a tabular problem?",
      "Explain the gradient boosting algorithm. What exactly is being boosted?",
      "What is the kernel trick in SVMs? Give an example.",
      "How do you handle class imbalance? Compare SMOTE, class weights, threshold moving.",
      "You have high training accuracy but low validation accuracy. Walk me through your diagnosis.",
      "Explain L1 vs L2 regularization. Why does L1 produce sparse weights?",
      "How does random forest differ from bagging on decision trees?"
    ],
    resources: [
      { name: "Hands-On Machine Learning (Géron) — Ch.1-9", url: "https://homl.info/", type: "book", priority: 1 },
      { name: "Andrew Ng ML Specialization (Coursera)", url: "https://www.coursera.org/specializations/machine-learning-introduction", type: "course", priority: 1 },
      { name: "sklearn documentation — User Guide", url: "https://scikit-learn.org/stable/user_guide.html", type: "docs", priority: 2 },
      { name: "Kaggle Learn — ML courses", url: "https://www.kaggle.com/learn", type: "course", priority: 2 }
    ],
    code: `import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb

# Gradient Boosting: F_m(x) = F_{m-1}(x) + η * h_m(x)
# Each tree h_m fits the negative gradient (residuals) of the loss
# For log-loss: pseudo-residuals = y - p(x) (actual - predicted probability)

# Error analysis workflow — the most important skill in practical ML
def error_analysis(model, X_val, y_val, feature_names):
    y_pred = model.predict(X_val)
    errors = X_val[y_pred != y_val]
    error_labels = y_val[y_pred != y_val]
    pred_labels = y_pred[y_pred != y_val]
    
    # Categorize errors
    print("=== Error Analysis Report ===")
    print(f"Total errors: {len(errors)} / {len(y_val)} ({len(errors)/len(y_val)*100:.1f}%)")
    
    # Feature distributions in errors vs correct predictions
    correct = X_val[y_pred == y_val]
    for i, feat in enumerate(feature_names):
        error_mean = errors[:, i].mean()
        correct_mean = correct[:, i].mean()
        if abs(error_mean - correct_mean) > 0.5 * correct[:, i].std():
            print(f"  Feature '{feat}': errors={error_mean:.2f}, correct={correct_mean:.2f} ← investigate")

# XGBoost with proper early stopping
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

params = {
    "max_depth": 6,        # controls complexity (variance)
    "learning_rate": 0.05, # shrinkage — lower = more robust
    "subsample": 0.8,      # row subsampling (prevent overfitting)
    "colsample_bytree": 0.8,
    "min_child_weight": 5, # minimum samples per leaf (controls variance)
    "objective": "binary:logistic",
    "eval_metric": "auc"
}
model = xgb.train(params, dtrain, num_boost_round=1000,
                  evals=[(dval, "val")], early_stopping_rounds=50,
                  verbose_eval=50)`
  },
  {
    id: "deep-learning",
    name: "Deep Learning",
    cat: "core",
    hours: 75,
    depth: "Deep",
    why: "The technical foundation of everything in AI. If you cannot build and debug a neural network from scratch, you are dependent on black boxes. This is what separates ML Engineers from ML users.",
    subtopics: [
      "Feedforward networks: forward pass, backprop derivation, universal approximation theorem",
      "Activation functions: ReLU, GELU, SiLU (SwiGLU used in LLaMA), sigmoid, tanh — derivatives and dead neuron problem",
      "Optimizers: SGD + momentum, AdaGrad, RMSProp, Adam, AdamW — mathematical derivation",
      "Regularization: Dropout (inverted dropout), BatchNorm (covariate shift), LayerNorm, WeightNorm",
      "CNNs: convolutions (stride, padding, dilation), pooling, receptive field, residual connections (ResNet)",
      "RNNs/LSTMs: vanishing gradients, gating mechanisms, BPTT, gradient clipping",
      "Attention & Transformers: self-attention, multi-head attention, positional encoding, encoder-decoder",
      "Training tricks: learning rate schedules (cosine annealing, warmup), gradient clipping, mixed precision (fp16/bf16)",
      "Loss functions: MSE, cross-entropy, focal loss, contrastive loss, triplet loss",
      "Hyperparameter tuning: batch size, learning rate, depth, width — practical heuristics"
    ],
    exercises: [
      "Implement ResNet-18 from scratch in PyTorch. No nn.ResNet. Every layer by hand. Train on CIFAR-10 to >90%.",
      "Implement LayerNorm from scratch. Verify numerically against nn.LayerNorm on the same input.",
      "Implement the AdamW optimizer from scratch. Compare trajectory vs torch.optim.AdamW on a toy problem.",
      "Debug a failing training run: introduce 3 common bugs (wrong loss, no grad, data leakage) and fix each.",
      "Reproduce the training instability in a deep network without LayerNorm. Add it back and show stability.",
      "Implement mixed precision training: use torch.cuda.amp.autocast. Measure speed and memory improvement."
    ],
    project: "Train a transformer language model (char-level or BPE token-level) on a Bengali text corpus from scratch using PyTorch. Implement: multi-head attention, positional encoding, feedforward block, training loop with cosine LR schedule, and text generation. Target: coherent character sequences after 1000 steps.",
    interview_qs: [
      "Explain BatchNorm and LayerNorm. Why did LLMs switch to LayerNorm?",
      "Walk me through the forward and backward pass of a single transformer block. What are all the matrix shapes?",
      "Why does Adam converge faster than SGD? What are its failure modes?",
      "You're training a deep network and loss is NaN. Walk me through your debugging process.",
      "What is gradient clipping? When is it necessary and what threshold do you use?",
      "Explain residual connections. Why do they help with training deep networks?"
    ],
    resources: [
      { name: "Karpathy — Neural Networks: Zero to Hero (YouTube)", url: "https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ", type: "video", priority: 1 },
      { name: "Deep Learning Book (Goodfellow, Bengio, Courville — free)", url: "https://www.deeplearningbook.org/", type: "book", priority: 1 },
      { name: "Fast.ai Practical Deep Learning (Part 1 & 2)", url: "https://course.fast.ai/", type: "course", priority: 1 },
      { name: "PyTorch documentation + tutorials", url: "https://pytorch.org/tutorials/", type: "docs", priority: 2 }
    ],
    code: `import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    """Implement from scratch — understand every line."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask=None):
        B, T, C = x.shape  # batch, seq_len, d_model
        # Project and split into heads
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        # (B, n_heads, T, d_k) for each
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, V)  # (B, n_heads, T, d_k)
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out), attn  # return weights for visualization`
  },
  {
    id: "nlp",
    name: "NLP",
    cat: "applied",
    hours: 60,
    depth: "Deep",
    why: "Your Bengali NLP specialization IS your competitive moat. NLP is the direct path to LLM Engineering. Every LLM application — chatbots, RAG, summarization — is NLP at its core.",
    subtopics: [
      "Text preprocessing: tokenization, normalization, stopword removal, stemming vs lemmatization",
      "Tokenization algorithms: BPE (GPT-2/LLaMA), WordPiece (BERT), SentencePiece, Unigram",
      "Word embeddings: Word2Vec (skip-gram, CBOW), GloVe, FastText — training objectives",
      "Sequence models: LSTM, bidirectional LSTM, seq2seq with attention",
      "BERT architecture: masked language modeling, next sentence prediction, [CLS] token",
      "GPT architecture: causal LM, autoregressive generation, KV cache",
      "Fine-tuning paradigms: feature extraction vs full fine-tuning vs PEFT",
      "NLP tasks: text classification, NER, question answering, summarization, translation",
      "Evaluation: BLEU, ROUGE-L, BERTScore, perplexity, human evaluation",
      "Bengali NLP: morphological richness, Unicode normalization, BanglaBERT, mBERT, XLM-RoBERTa",
      "Multilingual models: language identification, cross-lingual transfer, zero-shot evaluation"
    ],
    exercises: [
      "Implement BPE tokenization from scratch: start with character vocab, merge most frequent pairs, build vocab of size 1000.",
      "Train Word2Vec skip-gram from scratch on a Bengali text corpus. Verify: nearest neighbors of 'রাজধানী' include 'ঢাকা'.",
      "Fine-tune BanglaBERT for 5 different NLP tasks. Compare convergence speed across tasks.",
      "Implement beam search decoding from scratch for seq2seq generation.",
      "Build a Bengali text classifier: scrape 1000 news headlines, label 5 categories, fine-tune, deploy as API."
    ],
    project: "Bengali Named Entity Recognition system: fine-tune XLM-RoBERTa-large on a Bengali NER dataset, evaluate with seqeval (entity-level F1), deploy as FastAPI endpoint with confidence scores per entity, push model to HuggingFace Hub. Write a technical blog post.",
    interview_qs: [
      "Explain BPE tokenization. What problem does it solve vs word-level tokenization?",
      "What is the difference between BERT and GPT architectures? When would you use each?",
      "Explain the attention mask in BERT. How does masked language modeling work?",
      "What is transfer learning in NLP? Why does fine-tuning a pretrained model work so well?",
      "How would you evaluate a summarization model? What are ROUGE's limitations?",
      "What challenges arise in low-resource NLP for Bengali? How do multilingual models help?"
    ],
    resources: [
      { name: "HuggingFace NLP Course (free, hands-on)", url: "https://huggingface.co/learn/nlp-course/", type: "course", priority: 1 },
      { name: "Speech and Language Processing (Jurafsky & Martin — free PDF)", url: "https://web.stanford.edu/~jurafsky/slp3/", type: "book", priority: 1 },
      { name: "BanglaBERT (BUET NLP paper)", url: "https://huggingface.co/csebuetnlp/banglabert", type: "model", priority: 1 },
      { name: "ACL Anthology — Bengali NLP papers", url: "https://aclanthology.org/", type: "research", priority: 2 }
    ],
    code: `# BPE Tokenization from scratch
from collections import Counter, defaultdict

def get_vocab(corpus: list[str]) -> dict:
    """Initialize vocab with character-level tokens + </w> word boundary."""
    vocab = Counter()
    for word in corpus:
        vocab[' '.join(list(word)) + ' </w>'] += 1
    return vocab

def get_bigram_stats(vocab: dict) -> dict:
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def merge_vocab(pair: tuple, vocab: dict) -> dict:
    new_vocab = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word in vocab:
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = vocab[word]
    return new_vocab

def train_bpe(corpus: list[str], num_merges: int):
    vocab = get_vocab(corpus)
    merges = []
    for i in range(num_merges):
        pairs = get_bigram_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        merges.append(best)
        if i % 100 == 0:
            print(f"Merge {i}: {best[0]}+{best[1]} → {''.join(best)} (freq={pairs[best]})")
    return vocab, merges`
  },
  {
    id: "computer-vision",
    name: "Computer Vision",
    cat: "applied",
    hours: 50,
    depth: "Medium",
    why: "Vision is the second-largest ML application domain. Multimodal LLMs (LLaVA, GPT-4V, Gemini) require CV understanding. CLIP and ViT underpin every image-text system. Strong signal for research roles.",
    subtopics: [
      "Image representation: pixels, channels, tensors (C,H,W), normalization conventions",
      "Convolutions: kernel, stride, padding, dilation, depthwise separable convolutions",
      "Pooling: max pooling, average pooling, global average pooling, adaptive pooling",
      "Architectures: AlexNet → VGG → ResNet → EfficientNet → ConvNeXt (evolution)",
      "Residual connections: skip connections prevent vanishing gradients, enable 1000+ layer networks",
      "Object detection: R-CNN family, YOLO (anchor-free), DETR (transformer-based)",
      "Semantic segmentation: FCN, U-Net, Mask R-CNN, Segment Anything Model (SAM)",
      "Vision Transformers (ViT): patch embedding, class token, no inductive bias",
      "CLIP: contrastive image-text pretraining, zero-shot classification",
      "Transfer learning: when to freeze, when to fine-tune, which layers to unfreeze",
      "Data augmentation: random crop, flip, color jitter, mixup, cutmix, AutoAugment"
    ],
    exercises: [
      "Implement a conv2d operation from scratch (no nn.Conv2d). Verify output matches PyTorch on a 3×3 image.",
      "Implement ResNet-50 block with residual connection. Handle the projection shortcut for dimension mismatch.",
      "Fine-tune CLIP for a Bengali image-text matching task (if Bengali captions exist, or create 100 yourself).",
      "Run YOLO on a webcam feed or video file. Measure inference FPS. Understand the speed-accuracy tradeoff.",
      "Implement test-time augmentation (TTA): average predictions over 5 augmented versions of each test image."
    ],
    project: "Document digitization system: given an image of a handwritten or printed Bengali form, detect fields (bounding boxes with YOLO), crop each field, run OCR (Tesseract + fine-tuned model). Output structured JSON. This is a real-world Bangladesh use case with commercial value.",
    interview_qs: [
      "Explain the difference between semantic segmentation, instance segmentation, and panoptic segmentation.",
      "Why did ViT outperform CNNs on large-scale pretraining? What advantage does self-attention provide?",
      "Explain CLIP training. How is the contrastive loss computed?",
      "What is the receptive field of a network and why does it matter?",
      "You have 100 labeled images for a custom detection task. What approach would you take?",
      "What is depthwise separable convolution? Why is it used in MobileNet?"
    ],
    resources: [
      { name: "CS231n — CNNs for Visual Recognition (Stanford, free)", url: "https://cs231n.github.io/", type: "course", priority: 1 },
      { name: "Practical Deep Learning for Coders (fast.ai)", url: "https://course.fast.ai/", type: "course", priority: 1 },
      { name: "Ultralytics YOLO documentation", url: "https://docs.ultralytics.com/", type: "docs", priority: 2 },
      { name: "Papers With Code — CV benchmarks", url: "https://paperswithcode.com/area/computer-vision", type: "research", priority: 2 }
    ],
    code: `import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

# ResNet BasicBlock — understand residual connections
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Projection shortcut: align dimensions when stride > 1
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # Key insight: gradients flow directly through shortcut
        # ∂L/∂x = ∂L/∂out * (1 + ∂F/∂x) — the "1" prevents vanishing
        out += self.shortcut(x)
        return self.relu(out)

# CLIP zero-shot classification
from transformers import CLIPProcessor, CLIPModel
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
image = Image.open("test.jpg")
labels = ["a cat", "a dog", "a car"]
inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)
probs = outputs.logits_per_image.softmax(dim=1)
print(dict(zip(labels, probs[0].tolist())))`
  },
  {
    id: "llm",
    name: "LLM Engineering",
    cat: "applied",
    hours: 70,
    depth: "Deep",
    why: "This IS your primary target role. Every company building AI products needs LLM engineers. Fine-tuning, alignment, serving, evaluation — the full LLM lifecycle is a distinct engineering discipline.",
    subtopics: [
      "Transformer architecture deep dive: KV cache, RoPE positional encoding, GQA, SwiGLU",
      "Tokenization: BPE/SentencePiece, vocabulary size tradeoffs, tokenization for non-Latin scripts",
      "Pretraining: next token prediction, data curation, training at scale (gradient checkpointing, ZeRO)",
      "Instruction tuning: SFT (supervised fine-tuning), data formatting (ChatML, Alpaca template)",
      "RLHF: reward model training, PPO, RLAIF, constitutional AI",
      "DPO: direct preference optimization — simpler than RLHF, same goal",
      "PEFT: LoRA, QLoRA, IA³, prefix tuning — when to use which",
      "Inference optimization: quantization (INT8, INT4, GPTQ, AWQ), speculative decoding, vLLM",
      "Context management: sliding window attention, RAG for long context, chunking strategies",
      "Prompt engineering: system prompts, few-shot, CoT, self-consistency, structured outputs",
      "Evaluation: perplexity, human eval, LLM-as-judge, MT-bench, MMLU, HellaSwag"
    ],
    exercises: [
      "Implement KV cache from scratch: show how it reduces computation from O(T²) to O(T) for autoregressive generation.",
      "Fine-tune Qwen2.5-0.5B on a 500-sample Bengali instruction dataset using QLoRA. Evaluate before/after.",
      "Run DPO training: create 100 Bengali preference pairs, train, show reward margin increasing.",
      "Benchmark vLLM vs HuggingFace generate: measure throughput (tokens/sec) and latency for batch size 1 vs 16.",
      "Implement grouped query attention (GQA) and compare memory usage to multi-head attention.",
      "Build a structured output system: force LLM to output valid JSON schema using Instructor or Outlines."
    ],
    project: "Full LLM fine-tuning pipeline with evaluation: (1) Collect 1000 Bengali instruction samples. (2) QLoRA fine-tune LLaMA/Qwen. (3) Run DPO on 200 preference pairs. (4) Evaluate with ROUGE + LLM-as-judge. (5) Deploy with vLLM. (6) Track everything in wandb. Push model to HuggingFace Hub with a model card.",
    interview_qs: [
      "Explain the KV cache. Why is it necessary and how much memory does it save?",
      "What is the difference between RLHF and DPO? When would you choose DPO?",
      "Explain QLoRA. What is NF4 quantization and why is it better than INT4 for weights?",
      "How does vLLM achieve high throughput? What is PagedAttention?",
      "What is temperature, top-p, and top-k sampling? How do they interact?",
      "How would you evaluate a fine-tuned LLM beyond perplexity?"
    ],
    resources: [
      { name: "Karpathy — Let's reproduce GPT-2 (YouTube)", url: "https://youtu.be/l8pRSuU81PU", type: "video", priority: 1 },
      { name: "HuggingFace TRL documentation (SFT, DPO, PPO)", url: "https://huggingface.co/docs/trl/", type: "docs", priority: 1 },
      { name: "LLM University by Cohere", url: "https://docs.cohere.com/docs/llmu", type: "course", priority: 2 },
      { name: "vLLM documentation", url: "https://docs.vllm.ai/", type: "docs", priority: 2 },
      { name: "Chip Huyen — Designing Machine Learning Systems", url: "https://www.oreilly.com/library/view/designing-machine-learning/9781098107963/", type: "book", priority: 2 }
    ],
    code: `from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig
import torch

# KV Cache intuition — why it matters
# Without cache: each new token requires recomputing attention over ALL previous tokens
# Complexity: O(T²) per generation step
# With cache: store K,V for all previous tokens, only compute for new token
# Complexity: O(T) per step after initial prefill

# QLoRA configuration
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",         # NormalFloat4: 4-bit quantization
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,    # quantize the quantization constants too
)
# Memory: fp16 model = 2 bytes/param, NF4 = 0.5 bytes/param → 4x compression

# DPO training
dpo_config = DPOConfig(
    beta=0.1,              # KL penalty strength — higher = stay close to reference
    max_length=512,
    max_prompt_length=256,
    learning_rate=5e-7,    # much lower than SFT
    per_device_train_batch_size=2,
    report_to="wandb",
)
# DPO loss: -log σ(β * (log π_θ(y_w|x) - log π_θ(y_l|x)) 
#                  - β * (log π_ref(y_w|x) - log π_ref(y_l|x)))
# No explicit reward model needed — the reference model IS the implicit reward`
  },
  {
    id: "rag",
    name: "RAG Systems",
    cat: "systems",
    hours: 45,
    depth: "Deep",
    why: "RAG is the most deployed LLM application pattern in 2024-25. Every enterprise building on LLMs needs engineers who can build, evaluate, and optimize RAG systems. Your FastAPI background makes you the perfect RAG engineer.",
    subtopics: [
      "Embeddings: dense (sentence-transformers), sparse (BM25), hybrid search",
      "Chunking strategies: fixed, recursive, semantic, parent-child, late chunking",
      "Vector stores: FAISS (local), ChromaDB (persistent), Weaviate, Qdrant, pgvector",
      "Retrieval: bi-encoder (fast, approximate) vs cross-encoder reranking (slow, accurate)",
      "Advanced RAG: HyDE, multi-query retrieval, step-back prompting, FLARE",
      "Context assembly: how to format retrieved chunks for the LLM context window",
      "RAG evaluation: RAGAS metrics (faithfulness, answer relevancy, context precision, recall)",
      "Agentic RAG: LLM decides whether and what to retrieve",
      "Graph RAG: Microsoft's approach — construct knowledge graph, traverse for retrieval",
      "Production RAG: caching, async retrieval, streaming, monitoring, cost optimization"
    ],
    exercises: [
      "Compare dense vs sparse (BM25) vs hybrid retrieval on a 500-doc corpus. Measure recall@5 for each.",
      "Implement HyDE: generate a hypothetical answer, embed it instead of the question. Compare recall@5.",
      "Build reranking pipeline: retrieve top-20 with FAISS, rerank with cross-encoder, compare NDCG vs retrieve-only.",
      "Run RAGAS evaluation on your RAG system. Tune chunk_size and top_k until faithfulness > 0.85.",
      "Implement semantic caching: if query embedding is within cosine similarity 0.95 of a cached query, return cached answer."
    ],
    project: "Domain-specific RAG system with full evaluation: (1) Choose domain (Bengali law texts, medical PDFs, or ML papers). (2) Build ingestion pipeline (PyMuPDF + chunking + ChromaDB). (3) Add reranking. (4) Evaluate with RAGAS on 50 QA pairs. (5) Build Streamlit UI. (6) Deploy. (7) Write technical blog post with your RAGAS scores.",
    interview_qs: [
      "Explain the difference between bi-encoder and cross-encoder retrieval. Why use both?",
      "What is HyDE and when does it help retrieval performance?",
      "Walk me through the RAGAS faithfulness metric. How is it computed?",
      "What chunking strategy would you use for a 100-page legal PDF? Justify your choice.",
      "How would you scale a RAG system from 1000 to 10 million documents?",
      "What is the difference between RAG and fine-tuning for knowledge injection? When to use each?"
    ],
    resources: [
      { name: "RAGAS paper and documentation", url: "https://docs.ragas.io/", type: "docs", priority: 1 },
      { name: "LlamaIndex documentation (advanced RAG patterns)", url: "https://docs.llamaindex.ai/", type: "docs", priority: 1 },
      { name: "RAG survey paper (Gao et al. 2023, arxiv 2312.10997)", url: "https://arxiv.org/abs/2312.10997", type: "paper", priority: 1 },
      { name: "ChromaDB documentation", url: "https://docs.trychroma.com/", type: "docs", priority: 2 }
    ],
    code: `from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
import numpy as np

# Two-stage retrieval: bi-encoder → cross-encoder reranking
class RAGRetriever:
    def __init__(self):
        # Stage 1: Fast approximate retrieval (inner product ≈ cosine)
        self.bi_encoder = SentenceTransformer('BAAI/bge-small-en-v1.5')
        # Stage 2: Accurate reranking (slower but better)
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.db = chromadb.PersistentClient("./chroma")
        self.collection = self.db.get_or_create_collection(
            "docs", metadata={"hnsw:space": "cosine"}
        )

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        # Stage 1: get top-20 candidates quickly
        q_emb = self.bi_encoder.encode([query], normalize_embeddings=True)
        results = self.collection.query(
            query_embeddings=q_emb.tolist(), n_results=20
        )
        candidates = results['documents'][0]
        
        # Stage 2: rerank with cross-encoder
        pairs = [[query, doc] for doc in candidates]
        scores = self.cross_encoder.predict(pairs)  # [-inf, +inf]
        
        # Return top-k after reranking
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [{"doc": candidates[i], "score": float(scores[i])} 
                for i in top_indices]

# RAGAS evaluation
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset

def evaluate_rag(rag_system, test_pairs: list[dict]) -> dict:
    data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for pair in test_pairs:
        retrieved = rag_system.retrieve(pair["question"])
        answer = rag_system.generate(pair["question"], retrieved)
        data["question"].append(pair["question"])
        data["answer"].append(answer)
        data["contexts"].append([r["doc"] for r in retrieved])
        data["ground_truth"].append(pair["answer"])
    return evaluate(Dataset.from_dict(data),
                    metrics=[faithfulness, answer_relevancy, context_precision])`
  },
  {
    id: "agentic-ai",
    name: "Agentic AI",
    cat: "systems",
    hours: 40,
    depth: "Deep",
    why: "Agentic AI is the fastest-growing segment in 2025. Companies are moving from chatbots to autonomous agents that take actions. LangGraph is rapidly becoming the production standard. This is your Month 3 core skill.",
    subtopics: [
      "ReAct pattern: Reasoning → Action → Observation → loop until done",
      "Tool use: function calling schemas, tool routing, parallel tool execution",
      "LangGraph: StateGraph, nodes, edges, conditional branching, checkpointing",
      "Memory types: in-context (prompt), external (vector DB), episodic (conversation history), semantic (knowledge)",
      "Planning: task decomposition, hierarchical planning, Tree of Thought (ToT)",
      "Multi-agent patterns: supervisor, hierarchical, swarm, debate",
      "Human-in-the-loop: interrupt nodes, approval workflows, feedback incorporation",
      "Agent evaluation: task completion rate, efficiency (steps to completion), safety",
      "Failure modes: hallucinated tool calls, infinite loops, context overflow, tool misuse",
      "Production agents: rate limiting, cost tracking, fallback strategies, observability (LangSmith)"
    ],
    exercises: [
      "Implement ReAct from scratch without LangChain: LLM generates Thought/Action/Observation in a loop.",
      "Build a LangGraph agent with 5 nodes. Add a conditional edge that routes to different tools based on query type.",
      "Implement agent memory: persist conversation + tool results to SQLite, reload on next session.",
      "Build a self-correcting agent: if a tool fails, agent retries with modified parameters up to 3 times.",
      "Measure agent cost: log token counts per step, compute total cost per task across 20 tasks."
    ],
    project: "Research paper agent: given a research question, agent (1) searches arXiv API, (2) downloads and reads relevant papers via PDF tool, (3) identifies methodology patterns, (4) drafts a literature review, (5) checks citations. Uses LangGraph with persistent state + LangSmith observability. Deploy as a FastAPI endpoint.",
    interview_qs: [
      "Explain the ReAct pattern. What problem does it solve over one-shot prompting?",
      "What is LangGraph and why is it better than LangChain's AgentExecutor for production?",
      "How do you prevent an agent from running in an infinite loop?",
      "What are the different types of agent memory? When would you use a vector DB vs in-context memory?",
      "Design a multi-agent system for code review. What agents would you create and why?",
      "How do you evaluate an agentic system? What metrics matter?"
    ],
    resources: [
      { name: "LangGraph documentation and tutorials", url: "https://langchain-ai.github.io/langgraph/", type: "docs", priority: 1 },
      { name: "LangSmith (agent observability)", url: "https://docs.smith.langchain.com/", type: "docs", priority: 1 },
      { name: "ReAct paper (arxiv 2210.03629)", url: "https://arxiv.org/abs/2210.03629", type: "paper", priority: 1 },
      { name: "CrewAI documentation", url: "https://docs.crewai.com/", type: "docs", priority: 2 }
    ],
    code: `from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # append-only via reducer
    query: str
    tool_results: dict
    iteration: int
    done: bool

def reasoning_node(state: AgentState) -> dict:
    """LLM decides: use a tool or give final answer."""
    response = llm_with_tools.invoke(state["messages"])
    if response.tool_calls:
        return {"messages": [response], "done": False}
    return {"messages": [response], "done": True}

def tool_node(state: AgentState) -> dict:
    """Execute tools and return observations."""
    last_msg = state["messages"][-1]
    results = {}
    for tc in last_msg.tool_calls:
        result = execute_tool(tc["name"], tc["args"])
        results[tc["id"]] = result
    tool_messages = [
        ToolMessage(content=str(v), tool_call_id=k)
        for k, v in results.items()
    ]
    return {"messages": tool_messages, "tool_results": results,
            "iteration": state["iteration"] + 1}

def should_continue(state: AgentState) -> str:
    if state["done"]:
        return "end"
    if state["iteration"] >= 10:  # prevent infinite loops
        return "end"
    return "tools"

graph = StateGraph(AgentState)
graph.add_node("reason", reasoning_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("reason")
graph.add_conditional_edges("reason", should_continue,
                             {"tools": "tools", "end": END})
graph.add_edge("tools", "reason")  # cycle back
app = graph.compile(checkpointer=MemorySaver())`
  },
  {
    id: "mlops",
    name: "MLOps",
    cat: "systems",
    hours: 45,
    depth: "Medium-Deep",
    why: "An ML model that isn't deployed doesn't exist. MLOps is what separates a student project from a production system. Your FastAPI/Docker background makes this learnable in 4-6 weeks — faster than any pure ML person.",
    subtopics: [
      "Experiment tracking: wandb, MLflow — logging metrics, params, artifacts, models",
      "Hyperparameter optimization: wandb Sweeps, Optuna — Bayesian vs grid vs random",
      "Model registry: version control for models, promotion stages (dev → staging → prod)",
      "Containerization: Docker multi-stage builds for ML, optimizing image size, layer caching",
      "Model serving: FastAPI + uvicorn, batching requests, async inference, gRPC vs REST",
      "Scalable serving: vLLM (LLMs), TorchServe, Triton Inference Server, BentoML",
      "CI/CD for ML: GitHub Actions, automated eval gates, data validation (Great Expectations)",
      "Data versioning: DVC, LakeFS, Delta Lake — reproducible pipelines",
      "Feature stores: Feast, Tecton — online vs offline features",
      "Monitoring: data drift (Evidently), model performance drift, alerting",
      "Infrastructure: cloud GPU (GCP, AWS, Lambda Labs), cost optimization strategies"
    ],
    exercises: [
      "Set up a wandb sweep with Bayesian optimization over 4 hyperparameters. Show optimal config vs random baseline.",
      "Write a GitHub Actions workflow that runs eval on every PR and blocks merge if metric regresses.",
      "Dockerize your RAG API: multi-stage build, pre-download model weights, health check endpoint. Image size < 2GB.",
      "Implement request batching in FastAPI: accumulate requests for 50ms, batch process, return results.",
      "Set up Evidently monitoring: detect when embedding distribution drifts after 1000 queries."
    ],
    project: "Production ML pipeline: training (DVC for data) → evaluation (automated GitHub Actions) → Docker build → push to registry → deploy to Railway/Render → monitor with Evidently → alert on drift. Wire it all together with a single `make deploy` command that runs the entire pipeline.",
    interview_qs: [
      "What is the difference between a model registry and an artifact store?",
      "How do you detect and handle data drift in production?",
      "Explain blue-green deployment for an ML model. Why is it safer than a direct update?",
      "What is a feature store and when do you need one?",
      "How would you design a CI/CD pipeline for an LLM application?",
      "You have a model serving 1000 req/sec that is CPU-bound. What are your options?"
    ],
    resources: [
      { name: "Weights & Biases documentation", url: "https://docs.wandb.ai/", type: "docs", priority: 1 },
      { name: "Full Stack Deep Learning (2022 course, free)", url: "https://fullstackdeeplearning.com/course/2022/", type: "course", priority: 1 },
      { name: "Designing ML Systems (Chip Huyen)", url: "https://www.oreilly.com/library/view/designing-machine-learning/9781098107963/", type: "book", priority: 1 },
      { name: "MLOps community (blog + podcast)", url: "https://mlops.community/", type: "blog", priority: 2 }
    ],
    code: `import wandb
from github import Github  # PyGithub

# wandb Bayesian sweep
sweep_config = {
    "method": "bayes",
    "metric": {"name": "val/f1", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-3},
        "batch_size": {"values": [8, 16, 32]},
        "lora_r": {"values": [8, 16, 32]},
        "warmup_ratio": {"distribution": "uniform", "min": 0.0, "max": 0.1}
    },
    "early_terminate": {
        "type": "hyperband",  # terminate bad runs early
        "min_iter": 3
    }
}

# GitHub Actions eval gate (Python side)
def check_regression(new_metrics: dict, baseline_path: str, threshold: float = 0.02):
    """Block PR if metric regresses by more than threshold."""
    with open(baseline_path) as f:
        baseline = json.load(f)
    
    for metric in ["f1", "rouge_l", "faithfulness"]:
        if metric in new_metrics and metric in baseline:
            delta = new_metrics[metric] - baseline[metric]
            if delta < -threshold:
                print(f"REGRESSION: {metric} dropped {delta:.4f} (threshold: -{threshold})")
                exit(1)  # non-zero exit = GitHub Actions failure
    print("All metrics within threshold — PR approved")

# Async batching for inference API
from asyncio import Queue, sleep
import time

request_queue = Queue()
BATCH_TIMEOUT_MS = 50

async def batch_inference_worker():
    while True:
        batch = []
        deadline = time.monotonic() + BATCH_TIMEOUT_MS / 1000
        while time.monotonic() < deadline:
            try:
                item = request_queue.get_nowait()
                batch.append(item)
            except:
                await sleep(0.005)
        if batch:
            results = model.batch_predict([b["input"] for b in batch])
            for item, result in zip(batch, results):
                item["future"].set_result(result)`
  }
];

const TYPE_COLORS = { video:"#fb923c", course:"#34d399", book:"#a78bfa", docs:"#38bdf8", paper:"#f472b6", code:"#facc15", research:"#fb923c", interactive:"#34d399", model:"#38bdf8", blog:"#a78bfa" };

export default function App() {
  const [activeTopic, setActiveTopic] = useState("linear-algebra");
  const [activeSection, setActiveSection] = useState("subtopics");
  const [completedTopics, setCompletedTopics] = useState(new Set());

  const topic = TOPICS.find(t => t.id === activeTopic);
  const cat = CATEGORIES[topic.cat];
  const totalHours = TOPICS.reduce((a, t) => a + t.hours, 0);
  const doneHours = TOPICS.filter(t => completedTopics.has(t.id)).reduce((a, t) => a + t.hours, 0);

  const SECTIONS = [
    { id: "subtopics", label: "Subtopics" },
    { id: "why", label: "Why It Matters" },
    { id: "depth", label: "Depth" },
    { id: "exercises", label: "Exercises" },
    { id: "project", label: "Mini Project" },
    { id: "interview", label: "Interview Q&A" },
    { id: "resources", label: "Resources" },
    { id: "code", label: "Code" },
  ];

  const toggleDone = (id) => {
    setCompletedTopics(prev => {
      const s = new Set(prev);
      s.has(id) ? s.delete(id) : s.add(id);
      return s;
    });
  };

  const pct = Math.round((doneHours / totalHours) * 100);

  return (
    <div style={{ background: "var(--color-background-tertiary)", minHeight: "100vh", fontFamily: "var(--font-sans)" }}>
      {/* Top bar */}
      <div style={{ background: "var(--color-background-primary)", borderBottom: "0.5px solid var(--color-border-tertiary)", padding: "12px 16px", position: "sticky", top: 0, zIndex: 50 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "8px" }}>
          <div>
            <div style={{ fontSize: 15, fontWeight: 500, color: "var(--color-text-primary)" }}>AI Fundamentals Micro-Plan</div>
            <div style={{ fontSize: 12, color: "var(--color-text-tertiary)" }}>12 topics · {totalHours}h total · 3h/day ≈ 6.5 months</div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <div style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>{doneHours}h / {totalHours}h</div>
            <div style={{ width: 100, height: 6, background: "var(--color-background-tertiary)", borderRadius: 99, overflow: "hidden" }}>
              <div style={{ width: `${pct}%`, height: "100%", background: "#a78bfa", borderRadius: 99, transition: "width 0.3s" }} />
            </div>
            <div style={{ fontSize: 12, fontWeight: 500, color: "#a78bfa" }}>{pct}%</div>
          </div>
        </div>
      </div>

      <div style={{ display: "flex", height: "calc(100vh - 61px)" }}>
        {/* Sidebar */}
        <div style={{ width: 220, flexShrink: 0, background: "var(--color-background-primary)", borderRight: "0.5px solid var(--color-border-tertiary)", overflowY: "auto", padding: "8px 0" }}>
          {Object.entries(CATEGORIES).map(([catId, catInfo]) => (
            <div key={catId} style={{ marginBottom: 8 }}>
              <div style={{ padding: "4px 12px 2px", fontSize: 10, fontWeight: 500, color: catInfo.color, letterSpacing: "0.08em", textTransform: "uppercase" }}>
                {catInfo.label}
              </div>
              {TOPICS.filter(t => t.cat === catId).map(t => {
                const isActive = t.id === activeTopic;
                const isDone = completedTopics.has(t.id);
                return (
                  <button key={t.id} onClick={() => setActiveTopic(t.id)}
                    style={{ width: "100%", display: "flex", alignItems: "center", gap: 8, padding: "7px 12px", background: isActive ? `${catInfo.color}14` : "transparent", border: "none", cursor: "pointer", textAlign: "left", borderLeft: isActive ? `2px solid ${catInfo.color}` : "2px solid transparent" }}>
                    <span style={{ fontSize: 12, color: isActive ? catInfo.color : "var(--color-text-secondary)", fontWeight: isActive ? 500 : 400, flex: 1 }}>{t.name}</span>
                    <span style={{ fontSize: 10, color: "var(--color-text-tertiary)", flexShrink: 0 }}>{t.hours}h</span>
                    {isDone && <span style={{ fontSize: 12, color: catInfo.color }}>✓</span>}
                  </button>
                );
              })}
            </div>
          ))}
        </div>

        {/* Main */}
        <div style={{ flex: 1, overflowY: "auto" }}>
          {/* Topic header */}
          <div style={{ padding: "16px 20px 12px", borderBottom: "0.5px solid var(--color-border-tertiary)", background: "var(--color-background-primary)", display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 8, position: "sticky", top: 0, zIndex: 40 }}>
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                <h2 style={{ margin: 0, fontSize: 18, fontWeight: 500, color: "var(--color-text-primary)" }}>{topic.name}</h2>
                <span style={{ background: `${cat.color}20`, color: cat.color, fontSize: 10, padding: "2px 8px", borderRadius: 99, fontWeight: 500 }}>{cat.label}</span>
                <span style={{ background: "var(--color-background-secondary)", color: "var(--color-text-secondary)", fontSize: 10, padding: "2px 8px", borderRadius: 99 }}>{topic.hours}h</span>
                <span style={{ background: "var(--color-background-secondary)", color: "var(--color-text-secondary)", fontSize: 10, padding: "2px 8px", borderRadius: 99 }}>Depth: {topic.depth}</span>
              </div>
              {/* Section tabs */}
              <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                {SECTIONS.map(s => (
                  <button key={s.id} onClick={() => setActiveSection(s.id)}
                    style={{ fontSize: 12, padding: "3px 10px", borderRadius: 99, border: activeSection === s.id ? `1px solid ${cat.color}` : "0.5px solid var(--color-border-tertiary)", background: activeSection === s.id ? `${cat.color}18` : "transparent", color: activeSection === s.id ? cat.color : "var(--color-text-secondary)", cursor: "pointer", fontWeight: activeSection === s.id ? 500 : 400 }}>
                    {s.label}
                  </button>
                ))}
              </div>
            </div>
            <button onClick={() => toggleDone(topic.id)}
              style={{ padding: "6px 14px", borderRadius: 8, border: `1px solid ${completedTopics.has(topic.id) ? cat.color : "var(--color-border-secondary)"}`, background: completedTopics.has(topic.id) ? `${cat.color}22` : "transparent", color: completedTopics.has(topic.id) ? cat.color : "var(--color-text-secondary)", cursor: "pointer", fontSize: 12, fontWeight: 500 }}>
              {completedTopics.has(topic.id) ? "✓ Done" : "Mark done"}
            </button>
          </div>

          {/* Section content */}
          <div style={{ padding: "16px 20px" }}>

            {activeSection === "subtopics" && (
              <div>
                <div style={{ fontSize: 13, color: "var(--color-text-secondary)", marginBottom: 12 }}>Core concepts you must master in this topic.</div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 8 }}>
                  {topic.subtopics.map((s, i) => (
                    <div key={i} style={{ background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-tertiary)", borderRadius: 8, padding: "10px 12px", display: "flex", gap: 10, alignItems: "flex-start" }}>
                      <span style={{ color: cat.color, fontWeight: 500, fontSize: 12, flexShrink: 0, marginTop: 1 }}>{String(i+1).padStart(2,"0")}</span>
                      <span style={{ fontSize: 13, color: "var(--color-text-primary)", lineHeight: 1.5 }}>{s}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {activeSection === "why" && (
              <div style={{ background: "var(--color-background-primary)", border: `1px solid ${cat.color}33`, borderLeft: `3px solid ${cat.color}`, borderRadius: "0 8px 8px 0", padding: "16px 20px" }}>
                <div style={{ fontSize: 11, color: cat.color, letterSpacing: "0.08em", fontWeight: 500, marginBottom: 8 }}>WHY THIS MATTERS FOR YOUR CAREER</div>
                <p style={{ margin: 0, fontSize: 14, color: "var(--color-text-primary)", lineHeight: 1.7 }}>{topic.why}</p>
              </div>
            )}

            {activeSection === "depth" && (
              <div>
                <div style={{ background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-tertiary)", borderRadius: 8, padding: "14px 16px", marginBottom: 12 }}>
                  <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", letterSpacing: "0.08em", marginBottom: 6 }}>REQUIRED DEPTH</div>
                  <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                    <span style={{ background: `${cat.color}22`, color: cat.color, padding: "4px 14px", borderRadius: 99, fontSize: 13, fontWeight: 500 }}>{topic.depth}</span>
                    <span style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>Estimated: {topic.hours} hours · {Math.ceil(topic.hours / 3)} days at 3h/day</span>
                  </div>
                </div>
                <div style={{ background: "var(--color-background-secondary)", borderRadius: 8, padding: "12px 16px" }}>
                  <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", letterSpacing: "0.08em", marginBottom: 8 }}>DEPTH GUIDE</div>
                  {[
                    ["Basic", "Can use the API/library without understanding internals", topic.depth === "Basic"],
                    ["Medium", "Understand the math, can debug failures, can explain to others", topic.depth === "Medium" || topic.depth === "Medium-Deep"],
                    ["Deep", "Can implement from scratch, read research papers, propose improvements", topic.depth === "Deep" || topic.depth === "Medium-Deep"]
                  ].map(([level, desc, active]) => (
                    <div key={level} style={{ display: "flex", gap: 10, marginBottom: 8, opacity: active ? 1 : 0.4 }}>
                      <span style={{ color: active ? cat.color : "var(--color-text-tertiary)", fontWeight: 500, fontSize: 13, width: 60, flexShrink: 0 }}>{level}</span>
                      <span style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>{desc}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {activeSection === "exercises" && (
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {topic.exercises.map((ex, i) => (
                  <div key={i} style={{ background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-tertiary)", borderRadius: 8, padding: "12px 14px", display: "flex", gap: 12 }}>
                    <span style={{ background: `${cat.color}22`, color: cat.color, borderRadius: 6, width: 24, height: 24, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, fontWeight: 500, flexShrink: 0 }}>{i+1}</span>
                    <p style={{ margin: 0, fontSize: 13, color: "var(--color-text-primary)", lineHeight: 1.6 }}>{ex}</p>
                  </div>
                ))}
              </div>
            )}

            {activeSection === "project" && (
              <div style={{ background: "var(--color-background-primary)", border: `1px solid ${cat.color}44`, borderRadius: 8, padding: "16px 18px" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
                  <i className="ti ti-rocket" style={{ fontSize: 18, color: cat.color }} aria-hidden="true"></i>
                  <span style={{ fontSize: 13, fontWeight: 500, color: cat.color }}>Mini Project</span>
                </div>
                <p style={{ margin: 0, fontSize: 13, color: "var(--color-text-primary)", lineHeight: 1.7 }}>{topic.project}</p>
              </div>
            )}

            {activeSection === "interview" && (
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {topic.interview_qs.map((q, i) => (
                  <div key={i} style={{ background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-tertiary)", borderRadius: 8, padding: "12px 14px", display: "flex", gap: 12 }}>
                    <i className="ti ti-message-question" style={{ fontSize: 16, color: cat.color, flexShrink: 0, marginTop: 1 }} aria-hidden="true"></i>
                    <p style={{ margin: 0, fontSize: 13, color: "var(--color-text-primary)", lineHeight: 1.6 }}>{q}</p>
                  </div>
                ))}
              </div>
            )}

            {activeSection === "resources" && (
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {topic.resources.sort((a,b) => a.priority - b.priority).map((r, i) => (
                  <a key={i} href={r.url} style={{ textDecoration: "none" }}>
                    <div style={{ background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-tertiary)", borderRadius: 8, padding: "10px 14px", display: "flex", alignItems: "center", gap: 10, cursor: "pointer" }}>
                      <span style={{ background: `${TYPE_COLORS[r.type] || "#888"}22`, color: TYPE_COLORS[r.type] || "#888", fontSize: 10, padding: "2px 8px", borderRadius: 99, flexShrink: 0, fontWeight: 500 }}>{r.type}</span>
                      <span style={{ fontSize: 13, color: "var(--color-text-primary)", flex: 1 }}>{r.name}</span>
                      {r.priority === 1 && <span style={{ fontSize: 10, color: cat.color, fontWeight: 500, flexShrink: 0 }}>★ PRIORITY</span>}
                      <i className="ti ti-external-link" style={{ fontSize: 14, color: "var(--color-text-tertiary)", flexShrink: 0 }} aria-hidden="true"></i>
                    </div>
                  </a>
                ))}
              </div>
            )}

            {activeSection === "code" && (
              <div>
                <div style={{ fontSize: 12, color: "var(--color-text-tertiary)", marginBottom: 8 }}>Production-quality reference implementation for this topic.</div>
                <pre style={{ background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-tertiary)", borderRadius: 8, padding: "14px 16px", fontSize: 12, color: "var(--color-text-primary)", overflowX: "auto", fontFamily: "var(--font-mono)", lineHeight: 1.65, margin: 0, whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
                  {topic.code}
                </pre>
              </div>
            )}
          </div>

          {/* Hours summary */}
          <div style={{ margin: "0 20px 20px", background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-tertiary)", borderRadius: 8, padding: "12px 16px" }}>
            <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", letterSpacing: "0.08em", marginBottom: 10 }}>ALL TOPICS — HOURS OVERVIEW</div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {TOPICS.map(t => {
                const c = CATEGORIES[t.cat];
                const isActive = t.id === activeTopic;
                const isDone = completedTopics.has(t.id);
                return (
                  <button key={t.id} onClick={() => setActiveTopic(t.id)}
                    style={{ padding: "4px 10px", borderRadius: 99, border: isActive ? `1px solid ${c.color}` : "0.5px solid var(--color-border-tertiary)", background: isDone ? `${c.color}22` : "transparent", cursor: "pointer", fontSize: 12, color: isActive ? c.color : isDone ? c.color : "var(--color-text-secondary)" }}>
                    {t.name} <span style={{ color: "var(--color-text-tertiary)" }}>{t.hours}h</span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
