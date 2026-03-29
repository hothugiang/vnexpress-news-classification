# KNOWLEDGE_PROFILE: CDH-CRS (Causal Disentangled Hypergraph-based CRS)

## 1. Core Concept & Problem Formulation
- **Domain**: Conversational Recommendation System (CRS).
- **Primary Goal**: Mitigate Popularity Bias ($Z$) via Causal Inference ($do(P)$) and Hypergraph Modeling.
- **Input**: Dialogue context $C = \{u_1, ..., u_T\}$, entities $E = \{e_1, ..., e_k\}$.
- **Output**: Predicted item $\hat{y} \in \mathcal{I}$ where $P(y \mid z_{inv})$ is maximized.
- **Causal Assumption**: $P(y \mid C, E) \neq P(y \mid do(P))$ due to popularity confounding.

## 2. Hypergraph Preference Modeling ($\mathcal{H} = (V, \mathcal{E})$)
- **Nodes ($V$)**: $V = \mathcal{E}_{entity} \cup \mathcal{I}$.
- **Hyperedge Construction**: Each conversation $n$ forms one hyperedge $e^{(n)}$ containing all its entities $\{e_1, ..., e_k\}$. (Zero-shot annotation).
- **Propagation Rule**:
  $$X' = \sigma(D_v^{-1} H W_e D_e^{-1} H^T X W_v)$$
  Where $H$ is the incidence matrix, $D_v$ and $D_e$ are degree matrices.

## 3. Architecture Components
- **Dialogue Encoder**: $h_C = \text{Encoder}(C) \in \mathbb{R}^d$.
- **Hypergraph Entity Encoder**: $h_E = \frac{1}{k} \sum_{i=1}^{k} \text{HypergraphConv}(v_{e_i})$.
- **Preference Representation**: $z = W_c h_C + W_e h_E$.
- **Disentanglement Mechanism**:
  - $z_{inv} = W_{inv} z$ (Invariant preference).
  - $z_{spur} = W_{spur} z$ (Spurious popularity-correlated component).
  - **Orthogonality Constraint**: $L_{orth} = |z_{inv}^T z_{spur}|_2$.

## 4. Optimization & Objectives
- **Causal Prediction Head**: $\hat{y} = \text{softmax}(W_r z_{inv})$.
- **Popularity Debiased Loss**:
  $$L_{rec} = - \sum \frac{1}{pop(y)^\gamma} \log P(y \mid z_{inv})$$
  Where $pop(y)$ is normalized interaction frequency.
- **Invariant Regularization ($L_{inv}$)**: Penalizes gradient variance across popularity quantiles: $L_{inv} = \sum_{k=1}^{K} |\nabla_{w|z_{inv}} L_k|^2$.
- **Total Loss**: $L = L_{rec} + \lambda_1 L_{orth} + \lambda_2 L_{inv}$.

## 5. Implementation Benchmarks vs. MSCRS
| Feature | MSCRS (Baseline) | CDH-CRS (Ours) |
| :--- | :--- | :--- |
| **Graph Logic** | Pairwise (Entity-Entity) | Hypergraph (Dialogue-centric) |
| **Prediction** | Correlation-based | Invariant Causal-based |
| **Embeddings** | Entangled | Disentangled (Inv/Spur) |
| **Debias** | None | Popularity Intervention |

## 6. Key Contributions
1. Shift from pairwise entities to **compositional hyperedges**.
2. Prediction based strictly on **invariant representations**.
3. Bias handling via **interventional reweighting**.
4. Full compatibility with Redial/Inspired datasets without hidden field assumptions.
