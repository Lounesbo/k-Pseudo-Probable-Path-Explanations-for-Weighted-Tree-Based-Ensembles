# k-Pseudo-Probable Path Explanations for Weighted Tree-Based Ensembles

*Bridging Interpretability and Computational Efficiency*

Louenas Bounia (LIPN, UMR CNRS 7030, Université Sorbonne Paris Nord) · **ECML PKDD 2026**

---

## Highlights

- **A new type of formal explanation for tree ensembles.** It covers random
  forests and weighted ensembles (XGBoost, LightGBM, AdaBoost): at most $k$
  features, chosen to minimize a confidence-weighted error over the trees that
  vote for the predicted class.
- **Restricting to the trees that vote for the predicted class keeps the
  explanation consistent and the problem tractable.** Their abductive
  explanations can never justify another class.
- **Two greedy algorithms with guarantees.** GA and GD rely on the supermodularity
  of the weighted error, and their bounds depend on its curvature.
- **Orders of magnitude faster than classical abductive methods.** As $k$ grows,
  the explanations become genuine abductive explanations: up to 94% on placement
  at $k = 15$, and 63% on spambase at $k = 25$.

## Method in brief

Let $F$ be an ensemble and $\mathbf{x}$ the instance to explain. Only the trees
that agree with the prediction are kept:

```math
F_c = \lbrace T_i \in F \;:\; T_i(\mathbf{x}) = F(\mathbf{x}) \rbrace .
```

Each $T_i \in F_c$ has a confidence $P_i$, the majority proportion in the leaf
reached by $\mathbf{x}$. Let $\mu_{T_i,\mathbf{x}}(S)$ be the number of instances
agreeing with $\mathbf{x}$ on $S$ that $T_i$ classifies differently. The objective
weights the trees by $1 - P_i$:

```math
\epsilon_{F_c,\mathbf{x}}(S) \;=\; \frac{1}{|F_c|} \sum_{T_i \in F_c} (1 - P_i)\,\frac{\mu_{T_i,\mathbf{x}}(S)}{2^{\,n-|S|}} .
```

Given a candidate set $I$ built from the paths of the trees of $F_c$, a
**k-pseudo-probable explanation** is a set $S \subseteq I$ with
$\lvert S \rvert \le k$ that minimizes $\epsilon_{F_c,\mathbf{x}}(S)$. This error
is computed in time linear in the size of $F_c$, but the minimization is NP-hard.

The unnormalized error $\mu_{F_c,\mathbf{x}} = \sum_{T_i \in F_c} (1-P_i)\,\mu_{T_i,\mathbf{x}}$
is supermodular and non-increasing. Its curvature $c_f$ is smaller than $1$ as
soon as $I$ is a sufficient reason for one tree of $F_c$. This yields:

```math
\underbrace{\epsilon_{F_c,\mathbf{x}}(S_{GA}) \le \frac{\epsilon_{F_c,\mathbf{x}}(S^\ast)}{1-\gamma},\ \ \gamma = \max\!\left(\tfrac{1}{e}, c_f\right)}_{\text{GA: grows } S \text{ from } \emptyset}
\qquad
\underbrace{\epsilon_{F_c,\mathbf{x}}(S_{GD}) \le \frac{e^{p}-1}{p}\,\epsilon_{F_c,\mathbf{x}}(S^\ast),\ \ p = \tfrac{c_f}{1-c_f}}_{\text{GD: shrinks } S \text{ from } I}
```

When $\epsilon_{F_c,\mathbf{x}}(S) = 0$, $S$ is an abductive explanation. In
general, $S$ is a $\delta$-probable reason with
$\delta = 1 - \epsilon_{F_c,\mathbf{x}}(S)$.

## Experiments

The results of the three experiments of the paper are in `results/`. Each
experiment has a table of raw results (`.csv`, mean ± standard deviation over the
test instances) and a booklet of figures (`.pdf`, one bookmark per dataset).

**Approximation quality** (`exp1_approximation`). GA and GD are compared with an
exhaustive search over the subsets of $I$ of size at most $k$. The files give the
error, the size and the time as a function of $k$, on vote, biomed, placement,
horse, compas, cars, autos, tic-tac-toe, monk, heart, student-por and haberman.
See Fig. 2 of the paper, and Table 1 and Figs. 1–5 of the appendix.

**Against classical abductive explanations** (`exp2_classical_methods`). The
files give the explanation sizes at $k = 7$ and the running times of minimum
majoritary reasons, minimal sufficient reasons, preferred majoritary reasons and
majoritary reasons. They also give the share of GA and GD outputs that are
abductive, on compas, placement, horse and haberman. See Fig. 3 of the paper and
Fig. 6 of the appendix.

**Large-scale datasets** (`exp3_large_scale`). Classical methods do not scale on
spambase, bank, adult, mnist38 and gina. The files give the weighted error, the
share of trees of $F_c$ satisfied by $S$, and the share of abductive explanations
as a function of $k$. See Fig. 4 of the paper and Fig. 7 of the appendix.

The forests were trained with scikit-learn (CART), with default hyperparameters
except the number of trees, and with a time limit of 36,000 s per instance. The
experiments ran on an Intel Core i9-9900 at 3.1 GHz with 64 GB of RAM.

## Code

```
code/
├── Algo_ga_gd_exact.py   greedy_ascent, greedy_descent, exact_search, mu_forest, epsilon_forest, curvature
├── my_tree.py            scikit-learn trees to Boolean trees, DNF encodings, decision paths
├── my_forest.py          ensembles: direct reason, majority implicants, classical explanations
├── encoding_CNF.py       CNF encodings for the classical methods
└── timeout.py            time limit for the exact and classical methods
```

Requirements: `pip install -r requirements.txt` installs NumPy, pandas,
scikit-learn, tqdm, Matplotlib and [PySAT](https://pysathq.github.io/).

The algorithms take the DNFs of the trees of $F_c$, their weights $1 - P_i$, the
candidate set $I$ and the number $n$ of Boolean features. Run from `code/`:

```python
import my_tree as mt, my_forest as mf
from Algo_ga_gd_exact import greedy_ascent, greedy_descent, epsilon_forest

trees = []
for est in rf.estimators_:                           # rf: fitted RandomForestClassifier
    t = mt.decision_tree(); t.from_DecisionTreeClassifier(est); trees.append(t)
forest = mf.decision_forest(trees); forest.labels = rf.classes_
n = len(forest.bina)

x = X_test.values[0]
c = forest.take_decision(x)[0]
I = list(forest.find_direct_reason(x))               # features on the paths of F_c

Phi, w = [], []
for est, t in zip(rf.estimators_, trees):
    if t.take_decision(x) == c:                      # tree of F_c
        Phi.append(t.to_DNF(target=1 - c, hash_bin=forest.bina, threshold_clauses=False)[0].clauses)
        w.append(1 - P_i)                            # P_i: confidence of this tree on x

S = greedy_ascent(7, I, Phi, w, n)                   # or greedy_descent(7, I, Phi, w, n)
print(sorted(S, key=abs), epsilon_forest(S, Phi, w, n))
```

## Citation

```bibtex
@inproceedings{bounia2026kpseudo,
  title     = {k-Pseudo-Probable Path Explanations for Weighted Tree-Based Ensembles:
               Bridging Interpretability and Computational Efficiency},
  author    = {Bounia, Louenas},
  booktitle = {European Conference on Machine Learning and Principles and Practice
               of Knowledge Discovery in Databases (ECML PKDD)},
  year      = {2026}
}
```

Funded by the ANR JCJC project FORM-XAI (ANR-25-CE23-0146).
