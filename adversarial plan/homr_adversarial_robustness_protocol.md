# Comprehensive End-to-End Adversarial Robustness Protocol for the homr Architecture

## 1. Abstract and Objective Formulation

The `homr` (Homer's Optical Music Recognition) system addresses the complex transcription of Common Western Music Notation (CWMN) by partitioning the problem into a cascaded architecture. This consists of Phase I: structural layout parsing via semantic segmentation (`oemer` subsystem), and Phase II: semantic sequence generation via an autoregressive Transformer decoder (`Polyphonic-TrOMR` subsystem). 

The critical vulnerability in this pipeline is the discrete, non-differentiable bounding-box extraction bottleneck separating the two phases. Standard gradient-based adversarial analysis and robust optimization (adversarial training) are fundamentally blocked by this non-differentiable operator. 

This protocol establishes a mathematically rigorous methodology for adversarial testing and training by utilizing two parallel approaches:
1.  **Zero-Order Optimization (SPSA)** for exact structural vulnerability testing without architectural modification.
2.  **2D Spatial Soft-Argmax Continuous Relaxation** for end-to-end differentiable parameter optimization.

In addition, robustness is characterized over a 2D perturbation budget surface by sweeping both physical and mathematical perturbation magnitudes, $(\epsilon_{phys}, \epsilon_{math})$, and plotting downstream transcription error metrics (SER and CER) as response surfaces and slice curves.

### 1.1 Notation and Definitions

Prior to algorithmic formulation, we define the mathematical primitives utilized throughout this protocol:
* $X \in \mathbb{R}^{C \times H \times W}$: The input continuous spatial signal representing the scanned musical document, where $C$ is the channel dimension, $H$ is height, and $W$ is width.
* $Y = (y_1, y_2, \dots, y_T)$: The ground truth sequence of discrete musical tokens.
* $f_{seg}$: The Phase I layout parser, typically a UNet architecture mapping $X$ to unnormalized spatial activation heatmaps.
* $\text{extract}$: The discrete cropping operation isolating individual musical measures or primitives.
* $f_{seq}$: The Phase II autoregressive Transformer generator mapping isolated regions to the token sequence $\hat{Y}$.
* $\mathcal{L}(\hat{Y}, Y)$: The sequence generation loss, rigorously defined as the Negative Log-Likelihood (NLL) over the target sequence.

The composite architecture is defined as: $f(X) = f_{seq}(\text{extract}(f_{seg}(X)))$

The adversarial objective is to maximize the NLL loss within a constrained perturbation space $S$:
$$\max_{\delta \in S} \mathcal{L}(f(X + \delta), Y)$$

For protocol-wide evaluation we parameterize the perturbation family by two explicit controls,
$$X_{adv}(\epsilon_{phys},\epsilon_{math}) = \Pi_{[0,1]}\left(C(X;\epsilon_{phys}) + \delta^*(\epsilon_{math})\right),$$
and report task-level robustness as metric surfaces over this domain.

---

## 2. The Composite Attack Space and Spectral Noise

Adversarial perturbations against documents must model physical degradation (e.g., paper grain, non-uniform illumination) prior to mathematical exploitation. Physical degradation exhibits strong spatial correlation. Independent, Identically Distributed (IID) Gaussian noise fails to capture this topology.

We employ Spectral Noise Injection to simulate physical corruption $C(X)$. By modulating the frequency domain representation of white noise via a power-law filter, we generate correlated noise (e.g., pink noise, brown noise).

To make robustness claims reproducible, $\epsilon_{phys}$ is not fixed to a single value; it is evaluated on a predefined grid $\mathcal{E}_{phys}$ (Section 6), and combined with a matching grid $\mathcal{E}_{math}$.

### 2.1 Derivation of Spectral Noise

Let $Z$ be a matrix of IID standard normal random variables (white noise). 
Let $\mathcal{F}$ denote the 2-dimensional Fast Fourier Transform (FFT), mapping spatial domain to frequency domain: $\tilde{Z} = \mathcal{F}(Z)$. Let $u, v$ be the discrete spatial frequency coordinates and define the isotropic radial frequency by $f(u, v) = \sqrt{u^2 + v^2}$.

To avoid division by zero at the DC component we introduce a small numerical floor $\eta>0$ and define the stabilized frequency response
$$\bar{f}(u,v) = \max(f(u,v),\,\eta)\,,$$
so the filtered spectrum is computed as
$$\tilde{Z}_{\mathrm{filtered}}(u, v) = \tilde{Z}(u, v)\,\bar{f}(u,v)^{-\alpha}$$
The spatially correlated noise $N$ is recovered via the Inverse Fast Fourier Transform ($\mathcal{F}^{-1}$):
$$N = \operatorname{Re}\bigl(\mathcal{F}^{-1}(\tilde{Z}_{\mathrm{filtered}})\bigr)$$

### Algorithm 1: Spectral (Colored) Noise Injection

**Inputs:** * $X$: Original image tensor
* $\alpha$: Power-law decay factor ($\alpha=1$ for pink, $\alpha=2$ for brown noise)
* $\epsilon_{phys}$: Maximum physical noise magnitude threshold

**Procedure:**
1.  Initialize $Z \sim \mathcal{N}(0, I)$ with dimensions matching $X$.
2.  Compute frequency spectrum: $\tilde{Z} \leftarrow \text{FFT2D}(Z)$.
3.  Compute radial frequency grid $f(u,v) = \sqrt{u^2 + v^2}$ for all indices.
4.  Apply power-law filter with numerical floor: $\tilde{Z}_{filtered} \leftarrow \tilde{Z} \cdot (\bar{f}(u,v))^{-\alpha}$.
5.  Recover spatial noise: $N \leftarrow \mathrm{Real}(\text{InverseFFT2D}(\tilde{Z}_{filtered}))$.
6.  Normalize noise robustly: $N \leftarrow N / (\max(|N|) + \epsilon_{num})$ where $\epsilon_{num}\ll 1$ is a small constant to avoid division by zero.
7.  Apply thresholding to simulate ink bleed/fading non-linearities: $C(X) \leftarrow \text{Clip}(X + \epsilon_{phys} \cdot N, 0, 1)$.

**Output:** Physically corrupted tensor $C(X;\epsilon_{phys})$.


## 3. Phase A: Zero-Order Vulnerability Analysis

Standard backpropagation cannot pass through the discrete $\text{argmax}$ and slicing operations in the $\text{extract}$ function. To evaluate the exact architecture without modification, we utilize Simultaneous Perturbation Stochastic Approximation (SPSA).

### 3.1 Step-by-Step Derivation of SPSA Estimator

Let $\theta \in \mathbb{R}^d$ represent the input pixels to be perturbed; we seek the gradient vector $\nabla \mathcal{L}(\theta)$. Let $\Delta \in \{-1, 1\}^d$ be a vector of independent Rademacher random variables (each entry is $\pm1$ with probability $1/2$) and let $c>0$ be a finite-difference step size.

We approximate the objective function at positively and negatively perturbed points using a first-order multivariate Taylor series expansion centered at $\theta$:

1.  Positive perturbation expansion:
    $$\mathcal{L}(\theta + c\Delta) = \mathcal{L}(\theta) + (c\Delta)^T \nabla \mathcal{L}(\theta) + \mathcal{O}(c^2)$$
2.  Negative perturbation expansion:
    $$\mathcal{L}(\theta - c\Delta) = \mathcal{L}(\theta) - (c\Delta)^T \nabla \mathcal{L}(\theta) + \mathcal{O}(c^2)$$
3.  Subtract the negative expansion from the positive to isolate the first-order directional derivative:
    $$\mathcal{L}(\theta + c\Delta) - \mathcal{L}(\theta - c\Delta) = 2c\Delta^T \nabla \mathcal{L}(\theta) + \mathcal{O}(c^3)$$
4.  To estimate the partial derivative with respect to a specific parameter $\theta_i$, divide both sides by $2c\Delta_i$ (note $1/\Delta_i = \Delta_i$):
    $$\hat{g}_i = \frac{\mathcal{L}(\theta + c\Delta) - \mathcal{L}(\theta - c\Delta)}{2c\Delta_i}$$
In practice we recommend repeating the perturbation with $m$ independent draws $\{\Delta^{(j)}\}_{j=1}^m$ and averaging the resulting estimators to reduce variance:
    $$\hat{g} = \frac{1}{m}\sum_{j=1}^m \frac{\mathcal{L}(\theta + c\Delta^{(j)}) - \mathcal{L}(\theta - c\Delta^{(j)})}{2c} \;\Delta^{(j)}$$.

### Algorithm 2: SPSA-based Fast Gradient Sign Method (SPSA-FGSM)

**Inputs:**
* $X$: Corrupted input $C(X)$
* $Y$: Ground truth sequence
* $c$: Finite difference step size
* $\epsilon_{math}$: Maximum mathematical perturbation bound ($L_\infty$ norm)

**Procedure:**
1.  Sample perturbation vector $\Delta \sim \text{Rademacher}(d)$ matching dimensions of $X$.
2.  Construct positive perturbed input: $X^+ = \text{Clip}(X + c\Delta, 0, 1)$.
3.  Construct negative perturbed input: $X^- = \text{Clip}(X - c\Delta, 0, 1)$.
4.  Evaluate forward pass for positive loss: $L^+ = \mathcal{L}(f(X^+), Y)$.
5.  Evaluate forward pass for negative loss: $L^- = \mathcal{L}(f(X^-), Y)$.
6.  Compute scalar loss difference: $\Delta L = L^+ - L^-$.
7.  Estimate global gradient vector: $\hat{g} = \frac{\Delta L}{2c} \cdot \Delta$. In practice average this estimator over multiple independent $\Delta$ samples to reduce variance.
8.  Compute adversarial perturbation: $\delta = \epsilon_{math} \cdot \text{sign}(\hat{g})$.
9.  Generate adversarial example: $X_{adv} = \text{Clip}(X + \delta, 0, 1)$.

**Output:** Adversarial example $X_{adv}(\epsilon_{math})$ capable of attacking the non-differentiable architecture.


## 4. Phase B: Differentiable Pipeline Integration via 2D Spatial Soft-Argmax

For robust adversarial training, evaluating SPSA per iteration is computationally intractable due to the high dimensionality of $X$. We must construct an analytically differentiable pathway. A discrete $\text{argmax}$ produces coordinates $(x, y)$, but breaks gradients. We replace this with expected values over a probability distribution.

### 4.1 First Moments: Geometric Centers

Let $M \in \mathbb{R}^{H \times W}$ be the unnormalized 2D activation heatmap produced by $f_{seg}$ for a single target region (height $H$, width $W$). We index $M$ as $M_{y,x}$ where $y\in\{1,\dots,H\}$ and $x\in\{1,\dots,W\}$. Apply a numerically-stable 2D softmax with temperature $\tau$ to obtain a probability distribution $P$ over the spatial grid:
$$P_{x,y} = \frac{\exp\bigl(M_{y,x} / \tau\bigr)}{\sum_{i=1}^{H}\sum_{j=1}^{W} \exp\bigl(M_{i,j} / \tau\bigr)}$$

The center coordinates are the first moments of the distribution $P$ over the grid:
$$\hat{x}_c = \sum_{x=1}^{W}\sum_{y=1}^{H} x\cdot P_{x,y},\qquad \hat{y}_c = \sum_{x=1}^{W}\sum_{y=1}^{H} y\cdot P_{x,y}$$
Because the grid indices $x$ and $y$ are constants, the differentiability relies entirely on $P_{x,y}$, which natively propagates gradients back to $H$.

### 4.2 Second Moments: Bounding Box Dimensions

The box extents (width and height) are modeled using the second central moments (spatial variances):
$$ \sigma_x^2 = \sum_{x,y} x^2\,P_{x,y} - \hat{x}_c^2,\qquad \sigma_y^2 = \sum_{x,y} y^2\,P_{x,y} - \hat{y}_c^2 $$
To ensure numerical stability when variances are very small, add a small floor $\epsilon_{var}>0$ inside the square root. Let $k$ be a hyperparameter scale factor mapping standard deviation to explicit width/height:
$$\hat{w} = k\sqrt{\sigma_x^2 + \epsilon_{var}},\qquad \hat{h} = k\sqrt{\sigma_y^2 + \epsilon_{var}}$$

### Algorithm 3: Differentiable Spatial Extraction

**Inputs:**
* $H$: Phase I activation heatmaps.
* $X$: Raw image tensor.
* $\tau, k$: Temperature and scaling factors.

**Procedure:**
1.  Compute $P_{x,y} \leftarrow \text{Softmax2D}(H / \tau)$.
2.  Compute $\hat{x}_c, \hat{y}_c$ via First Moments (Eq. 4.1).
3.  Compute $\sigma_x^2, \sigma_y^2$ via Second Central Moments (Eq. 4.2).
4.  Compute dimensions $\hat{w}, \hat{h}$.
5.  Construct normalized bounding box coordinates: $[x_{min}, y_{min}, x_{max}, y_{max}]$, where $x_{min} = \hat{x}_c - \hat{w}/2$, etc.
6.  Generate a continuous sampling grid defined by the normalized bounds. Note: frameworks such as PyTorch expect sampling coordinates normalized to $[-1,1]$ and offer `align_corners` options; ensure consistent coordinate conventions when constructing the grid.
7.  Apply bilinear interpolation (e.g., `torch.nn.functional.grid_sample`) to map raw pixels $X$ onto the sampling grid, yielding the differentiable extracted tensor $X_{crop}$. Use careful handling of boundary modes (`padding_mode`) and `align_corners` to avoid subtle bias.

**Output:** Differentiable cropped tensor $X_{crop}$ mapped strictly via continuous operators.


## 5. Min-Max Robust Optimization (Adversarial Training)

We mathematically formalize the robust training as solving a saddle-point formulation over the concatenated network parameters $\Theta$:
$$\min_{\Theta} \mathbb{E}_{(X,Y),\,\epsilon_{phys}\sim\mathcal{P}_{phys}} \left[ \max_{\|\delta\|_\infty \leq \epsilon_{math}} \mathcal{L}(f_{\Theta}(C(X;\epsilon_{phys}) + \delta), Y) \right]$$

This requires an inner maximization algorithm to find the optimal perturbation $\delta^*$, followed by an outer minimization step updating $\Theta$. We utilize Projected Gradient Descent (PGD) as the inner maximizer, utilizing the continuous pipeline established in Phase B.

### Algorithm 4: Projected Gradient Descent (PGD) with Spatial Soft-Argmax

**Inputs:**
* $X$: Corrupted input $C(X)$.
* $Y$: Ground truth sequence.
* $f_{\Theta}^{relaxed}$: Continuous composite model with Phase B modifications.
* $\epsilon_{math}$: Perturbation bound ($L_\infty$).
* $\beta$: PGD step size.
* $N$: Number of iterations.

**Procedure:**
1.  Initialize $\delta_0 \sim \text{Uniform}(-\epsilon_{math}, \epsilon_{math})$ with bounds clipping.
2.  **For** $t = 0$ to $N-1$ **do**:
3.      Forward pass continuous model: $\hat{Y} = f_{\Theta}^{relaxed}(X + \delta_t)$.
4.      Compute NLL Loss: $L = \mathcal{L}(\hat{Y}, Y)$.
5.      Compute exact gradient via Autograd: $g = \nabla_{\delta} L$.
6.      Update perturbation: $\delta_{t+1} = \delta_t + \beta \cdot \text{sign}(g)$.
7.      Project onto $\epsilon_{math}$-ball: $\delta_{t+1} = \text{Clip}(\delta_{t+1}, -\epsilon_{math}, \epsilon_{math})$ (for $L_\infty$ projection). A practical default is to set $\beta = \epsilon_{math} / N$.
8.      Project valid image space: $\delta_{t+1} = \text{Clip}(X + \delta_{t+1}, 0, 1) - X$.
9.  **End For**

**Output:** Optimal mathematical perturbation $\delta^* = \delta_N$.

### Algorithm 5: End-to-End Robust Training Loop

**Inputs:** Dataset $D = \{(X^{(i)}, Y^{(i)})\}$, learning rate $\eta$, training perturbation sets $\mathcal{E}_{phys}^{train},\mathcal{E}_{math}^{train}$ (or sampling distributions).

**Procedure:**
1.  Initialize model parameters $\Theta$.
2.  **While** not converged **do**:
3.      Sample mini-batch $(X, Y)$ from $D$.
4.      Sample or cycle perturbation budgets $(\epsilon_{phys},\epsilon_{math})$ from $\mathcal{E}_{phys}^{train} \times \mathcal{E}_{math}^{train}$.
5.      Apply physical corruption: $X_C \leftarrow \text{Algorithm 1}(X;\epsilon_{phys})$.
6.      Execute Inner Maximization: $\delta^* \leftarrow \text{Algorithm 4}(X_C, Y, f_{\Theta}^{relaxed};\epsilon_{math})$.
7.      Construct Adversarial Composite: $X_{comp} = X_C + \delta^*$.
8.      Execute Outer Minimization Forward Pass: $L_{batch} = \mathcal{L}(f_{\Theta}^{relaxed}(X_{comp}), Y)$.
9.      Compute parameter gradients: $\nabla_{\Theta} L_{batch}$.
10.     Update parameters: $\Theta \leftarrow \Theta - \eta \nabla_{\Theta} L_{batch}$.
11. **End While**

**Output:** Adversarially robust parameter set $\Theta^*$.

## 6. Robustness Surface Evaluation Across $\epsilon_{phys}$ and $\epsilon_{math}$

This section defines the required evaluation protocol for reporting robustness. Instead of single-point attacks, evaluate the model over a Cartesian product of physical and mathematical perturbation magnitudes:
$$\mathcal{G}=\mathcal{E}_{phys}\times\mathcal{E}_{math},\quad
\mathcal{E}_{phys}=\{e^{(1)}_{phys},\dots,e^{(P)}_{phys}\},\quad
\mathcal{E}_{math}=\{e^{(1)}_{math},\dots,e^{(M)}_{math}\}.$$

Recommended initial grids (tune to data scaling and image normalization):
* $\mathcal{E}_{phys}=\{0.00,0.01,0.02,0.03,0.04,0.05\}$
* $\mathcal{E}_{math}=\{0.00,0.25/255,0.5/255,1/255,2/255,4/255\}$

### 6.1 Metrics: SER and CER

Let $\hat{Y}$ be model output and $Y$ reference tokens. Let $d_{lev}(\cdot,\cdot)$ denote Levenshtein edit distance.

Define Symbol Error Rate (SER) on symbolic token sequences:
$$\mathrm{SER}(\hat{Y},Y)=\frac{d_{lev}(\hat{Y},Y)}{|Y|}.$$ 

Define Character Error Rate (CER) on serialized text form (e.g., kern/MusicXML-derived plain sequence) with strings $\hat{S},S$:
$$\mathrm{CER}(\hat{S},S)=\frac{d_{lev}(\hat{S},S)}{|S|}.$$

Aggregate over dataset split $\mathcal{D}_{eval}$ for each grid point:
$$\overline{\mathrm{SER}}(e_p,e_m)=\frac{1}{|\mathcal{D}_{eval}|}\sum_{(X,Y)\in\mathcal{D}_{eval}}\mathrm{SER}(\hat{Y}_{e_p,e_m},Y),$$
$$\overline{\mathrm{CER}}(e_p,e_m)=\frac{1}{|\mathcal{D}_{eval}|}\sum_{(X,Y)\in\mathcal{D}_{eval}}\mathrm{CER}(\hat{S}_{e_p,e_m},S).$$

### Algorithm 6: Two-Epsilon Evaluation Sweep with SER/CER Logging

**Inputs:** Evaluation set $\mathcal{D}_{eval}$, trained model $f_\Theta$, grids $\mathcal{E}_{phys}, \mathcal{E}_{math}$, attack generator (SPSA or PGD).

**Procedure:**
1.  Initialize result tensors $R_{SER} \in \mathbb{R}^{P\times M}$ and $R_{CER} \in \mathbb{R}^{P\times M}$.
2.  **For** each $e_p \in \mathcal{E}_{phys}$ **do**:
3.      **For** each $e_m \in \mathcal{E}_{math}$ **do**:
4.          **For** each sample $(X,Y)$ in $\mathcal{D}_{eval}$ **do**:
5.              Compute $X_C \leftarrow \text{Algorithm 1}(X;e_p)$.
6.              Compute $X_{adv} \leftarrow \text{Attack}(X_C,Y;e_m)$.
7.              Infer prediction $\hat{Y}_{e_p,e_m}=f_\Theta(X_{adv})$.
8.              Convert prediction/reference to strings $(\hat{S},S)$ for CER.
9.              Accumulate per-sample SER and CER.
10.         **End For**
11.         Store averaged metrics in $R_{SER}[p,m]$ and $R_{CER}[p,m]$.
12.     **End For**
13. **End For**

**Output:** Robustness matrices $R_{SER}, R_{CER}$ for plotting and statistical reporting.

### 6.2 Plotting and Reporting Requirements

For both SER and CER, produce the following plots from $R_{SER}$ and $R_{CER}$:
1.  2D heatmaps with x-axis $\epsilon_{math}$, y-axis $\epsilon_{phys}$, color = error rate.
2.  Slice curves at fixed $\epsilon_{phys}$: plot error rate vs. $\epsilon_{math}$ for at least three levels (low/medium/high $\epsilon_{phys}$).
3.  Slice curves at fixed $\epsilon_{math}$: plot error rate vs. $\epsilon_{phys}$ for at least three levels (low/medium/high $\epsilon_{math}$).

Minimum reporting table columns per run:
* $(\epsilon_{phys}, \epsilon_{math})$
* mean SER, std SER
* mean CER, std CER
* sample count

Interpretation guidance:
* Monotonic increase with either epsilon indicates expected vulnerability scaling.
* Non-monotonic regions often indicate attack optimizer instability or saturation due to clipping; verify step size and iteration budget.
* A robust model should show lower surface area under both SER and CER response surfaces relative to baseline.

## 7. Implementation notes, hyperparameters and verification

To keep training and evaluation consistent:
* Use the same tokenization and normalization pipeline for clean and perturbed inputs before SER/CER computation.
* Report whether attack generation during evaluation is SPSA (exact non-differentiable path) or PGD (relaxed differentiable path).
* Cache per-grid predictions where feasible to avoid recomputation when regenerating plots/tables.
* Version control the exact epsilon grids, random seeds, and attack hyperparameters in experiment logs.


## 8. References and suggested reading

The repository currently has no bibliography file. Recommended references to cite and add to your bibliography:


Please replace these placeholders with formal bibliography entries in the format preferred by your project (BibTeX, RIS, etc.).
