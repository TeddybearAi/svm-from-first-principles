# A2 Weekly Implementation Journal — Support Vector Machines from First Principles

## Contents

- [Implementation Log — Week 1](#implementation-log--week-1)
- [Implementation Log — Week 2](#implementation-log--week-2)
- [Implementation Log — Week 3](#implementation-log--week-3)
- [Implementation Log — Week 4](#implementation-log--week-4)
- [Implementation Log — Week 5](#implementation-log--week-5)
- [Implementation Log — Week 6](#implementation-log--week-6)
- [Implementation Log — Week 7](#implementation-log--week-7)
- [Implementation Log — Week 8](#implementation-log--week-8)


> **Note:** These entries document the actual development sequence of the project. They are organised as Weeks 1–8, without invented calendar dates.

---

# Implementation Log — Week 1

## Research question and motivation

This week I settled on **Support Vector Machines (SVMs)** as my Option 1 topic. I wanted to study a fundamental machine-learning model rather than build an application around an off-the-shelf library. My initial research question was:

> **Does increasing the margin, through a smaller regularisation parameter \(C\), actually reduce the gap between training and test performance in practice, and where does the simple "wider margin generalises better" intuition break down?**

I chose this question because it lets me connect the geometric interpretation of SVMs directly to the learning-theory material from the subject. Instead of treating SVM as a classifier that I can simply call from `sklearn`, I want to understand the hypothesis family, the loss function, the optimisation problem, the role of regularisation, and the relationship between the primal and dual formulations.

I also decided that I should not study only a perfectly separable dataset. If the question is about margin, regularisation, and generalisation, then I need controlled regimes with different amounts of overlap/noise so I can see when the simple intuition works and when it becomes less clear.

## Theory studied this week

Before writing the main implementation, I worked through the main SVM components by hand:

- geometric margin and functional margin;
- hard-margin SVM;
- soft-margin SVM;
- slack variables \(\xi_i\);
- hinge loss;
- the role of the regularisation parameter \(C\);
- the Lagrangian dual;
- the equality constraint \(\sum_i \alpha_i y_i=0\);
- support vectors;
- the basic KKT interpretation;
- the idea of kernelisation.

My goal was to be able to explain what each quantity means before relying on code.

I also looked at the historical development of SVMs so that the project would not become only an implementation exercise. The main conceptual progression I wanted to understand was:

1. maximum-margin classification;
2. soft-margin regularisation for noisy/non-separable data;
3. dual optimisation;
4. kernel functions;
5. practical optimisation using SMO.

## Plan for the implementation

My initial technical plan was:

1. build a synthetic binary-classification dataset with controllable overlap;
2. standardise features using training data only;
3. implement a **linear soft-margin SVM from scratch** using subgradient descent;
4. implement the **dual formulation from scratch** with a simplified SMO solver;
5. add linear and RBF kernels to the dual implementation;
6. compare the scratch implementations with `sklearn.svm.SVC` only as an independent reference;
7. run controlled experiments across different \(C\) values;
8. study margin width, training/test accuracy, hinge loss, support-vector count, and generalisation gap.

I also planned to include one real dataset later, mainly to check that the conclusions were not only an artefact of 2D synthetic geometry.

## AI tool use

I used **Claude** to help structure the first weekly plan and to sanity-check my understanding of how the primal and dual formulations relate. I did not want to treat AI-generated explanation as automatically correct, so I used it mainly to help identify what I needed to derive and verify myself.

At this stage, the main questions I asked AI were conceptual:

- What exactly changes when \(C\) increases?
- How does hinge loss connect to slack variables?
- Why is the dual useful?
- Why does SMO update two multipliers at a time?
- What should I measure if my research question is about margin and generalisation?

I used these discussions to guide my own derivations and the design of the experiments.

## Open questions for next week

I still need to clarify:

- whether the research question is narrow enough to be answered convincingly;
- what validation should be used for if \(C\) itself is the independent variable of the study;
- whether the dual implementation should be a generic QP solver or a simplified SMO algorithm;
- how I should verify the scratch implementation beyond checking final classification accuracy;
- whether support-vector count is actually a valid proxy for model complexity.

My next step is to implement the first working primal and dual solvers and then test whether the experimental protocol makes sense in practice.

---

# Implementation Log — Week 2

## Progress this week

This week I built the first complete version of the project pipeline:

- data generation and preprocessing;
- train/validation/test splitting;
- a from-scratch primal soft-margin SVM;
- a from-scratch simplified SMO dual solver;
- linear and RBF kernels;
- initial \(C\)-sweep experiments;
- comparisons with `sklearn.svm.SVC`;
- the first version of the report.

The primal implementation optimises:

\[
J(w,b)
=
\frac{1}{2}\|w\|^2
+
C\sum_i \max(0,1-y_i(w^\top x_i+b))
\]

using full-batch subgradient descent.

The dual implementation optimises the standard SVM dual under:

\[
0\le \alpha_i\le C,
\qquad
\sum_i\alpha_i y_i=0
\]

using a simplified SMO pair update.

At this point the code produced sensible decision boundaries and the scratch models often matched `sklearn.svm.SVC` on test accuracy, so initially I thought the implementation was already fairly reliable.

## Main problem discovered: the validation split existed but was not actually being used properly

The first major issue I found was methodological rather than mathematical.

I had created train, validation, and test splits, and the report claimed that hyperparameters were selected using validation performance. However, when the experiment code was checked carefully, the validation arrays were being created but the main \(C\)-sweep was effectively evaluating configurations directly on the test set.

This was a real mismatch between the report and the implementation.

I revised my interpretation of validation:

- \(C\) should **not** be "selected away" by validation, because \(C\) is the independent variable I am deliberately studying.
- Validation should instead be used for implementation and model-design choices that are not themselves the experimental variable.
- The test set should remain untouched until those choices are frozen.

This gave me a clearer experimental protocol:

- **training set:** fit the model;
- **validation set:** choose implementation settings such as learning-rate behaviour, SMO stopping/pass budget, and later RBF \(\gamma\);
- **test set:** final evaluation after those decisions are fixed.

This was an important improvement because it made the experiment match the stated research question.

## Main problem discovered: repeated-seed variability was larger than I first realised

My early report wording suggested that the standard deviations across seeds were small relative to the effect of \(C\).

When the actual per-seed results were checked, this was not true for the high-overlap generalisation gap.

The mean trend at large \(C\) was small compared with run-to-run variability. This meant my original H3 conclusion was too confident.

I changed the interpretation from:

> large \(C\) clearly increases the generalisation gap

to:

> the mean gap tends to increase at larger \(C\), but the effect is modest relative to seed-to-seed variability and should be treated as suggestive rather than conclusive.

This was one of the first points where the project became more useful to me as a learning exercise: the correct conclusion was not the neatest conclusion.

## Verification problem: matching accuracy is not enough

I initially relied heavily on the fact that my scratch SVM and sklearn often had identical test accuracy.

I realised that this is weak verification. Two different decision functions can classify the same small test set identically.

I therefore planned stronger checks:

### For the primal model

- compare predictions;
- compare raw decision scores;
- compare \(w\) direction using cosine similarity;
- compare \(\|w\|\);
- compare the bias \(b\).

### For the dual model

- check box constraints;
- check \(\sum_i\alpha_i y_i\approx0\);
- inspect KKT residuals;
- compare prediction and decision scores;
- inspect support-vector counts.

This shifted the verification from "same accuracy" to "same model behaviour and correct constraints."

## Support-vector discrepancy discovered

On one matched low-overlap configuration, my scratch SMO solver produced **19 support vectors**, while sklearn produced **17**.

My first tolerance-sensitivity experiment was flawed because I accidentally regenerated a different data/split configuration instead of reproducing the exact 19-vs-17 fit.

I corrected the plan: any investigation of this discrepancy must use the exact same:

- data seed;
- split seed;
- \(C\);
- SMO settings;
- solver seed;
- `max_passes`;

and vary only the support-vector threshold.

## AI tool use and critical review

I used **Claude** to support the early implementation and report-drafting process, including iterative revisions. I then used **ChatGPT as an independent reviewer** to compare the report against the A2 specification and challenge the code/report consistency.

The most useful outcome this week was that AI-assisted review exposed weaknesses in earlier drafts. I did not simply accept whichever suggestion was newest. I treated disagreements as prompts to inspect the implementation and numerical outputs.

## What I learned this week

The biggest lesson was that a project can look correct at the top level while still having a flawed experimental protocol.

I also learned that:

- validation must have a clearly defined role;
- repeated trials are only useful if their variability is actually reported;
- matching accuracy alone does not verify an implementation;
- every debugging experiment must reproduce the exact configuration of the problem it claims to investigate.

## Plan for next week

Next week I will:

- make validation operational;
- add stronger geometry/constraint verification;
- fix the support-vector tolerance experiment;
- investigate whether the 19-vs-17 difference is caused by thresholding or incomplete convergence;
- review the KKT explanation carefully;
- add uncertainty directly to the figures.

---

# Implementation Log — Week 3

## Progress this week

This week the main focus shifted from building the pipeline to **checking whether the theory and diagnostics were actually correct**.

The project already ran end-to-end, but several statements in the report were too strong or mathematically imprecise. I worked through those one by one.

## Soft-margin KKT interpretation corrected

One important mistake was the statement:

\[
\alpha_i>0
\Rightarrow
y_i f(x_i)=1.
\]

This is not generally correct for a soft-margin SVM.

The correct complementary-slackness condition is:

\[
\alpha_i
\left[
y_i f(x_i)-1+\xi_i
\right]
=0.
\]

Therefore:

\[
\alpha_i>0
\Rightarrow
y_i f(x_i)=1-\xi_i.
\]

Only in the interior case:

\[
0<\alpha_i<C
\]

does the second KKT condition force:

\[
\xi_i=0,
\]

which gives:

\[
y_i f(x_i)=1.
\]

This distinction matters because a bound support vector with \(\alpha_i=C\) may lie inside the margin or even be misclassified.

I updated both the explanation and the solver comments so that the code and report used the same interpretation.

## Hinge loss was being computed but not actually reported

The experiment code already computed training and test hinge loss, but the first report versions mostly discussed accuracy.

This was a missed opportunity because the assignment explicitly asks for the relationship between the training loss and the real task objective.

Once I added hinge loss to the results, an important pattern became obvious.

In the low-overlap regime, training hinge loss fell approximately:

\[
0.280
\rightarrow
0.118
\rightarrow
0.072
\rightarrow
0.060
\rightarrow
0.057
\]

as \(C\) increased, while classification accuracy barely changed.

This gave a much clearer empirical explanation of the loss/objective mismatch:

- accuracy only depends on whether the prediction has the correct sign;
- hinge loss continues to penalise a correctly classified sample until its functional margin reaches 1.

Therefore hinge loss can improve substantially while 0-1 accuracy is already saturated.

This became one of the strongest findings in the project.

## Convexity and convergence wording corrected

Another theoretical overstatement was the claim that the primal objective had a **unique global optimum**.

Convexity means there are no spurious local minima, but uniqueness of the complete \((w,b)\) solution is not guaranteed because \(b\) is unregularised.

I changed the interpretation to:

> the objective is convex, so zero initialisation is valid and there are no spurious local minima.

I also corrected the language around the subgradient optimiser.

The `c_scaled` schedule uses a fixed effective step size for a given \(C\). Constant-step subgradient descent on a non-smooth convex objective does not generally guarantee exact convergence to the minimiser.

Therefore I stopped describing the scratch primal optimiser as exactly "converged" and instead used:

> stabilised sufficiently for the experimental analysis.

## Why the large-\(C\) optimiser bug happened

A major implementation issue became clearer this week.

With a fixed learning rate, large-\(C\) runs produced results that initially looked like possible overfitting or instability. However, the loss history showed that the objective was not stabilising at all.

The hinge-loss subgradient contains a factor of \(C\):

\[
-C\sum_{i:m_i<1}y_ix_i.
\]

So when \(C\) becomes large, the update magnitude can become too large if the learning rate is unchanged.

The fix was to scale the effective learning rate as:

\[
\eta_{\text{eff}}
=
\frac{0.01}{\max(1,C)}.
\]

After this change, the loss trajectories became much more stable across the \(C\)-sweep.

This was an important debugging example because the incorrect model produced plausible-looking numbers. Without checking the optimisation history, I could have interpreted an optimiser failure as a genuine machine-learning result.

## Failure-case analysis made reproducible

An earlier report statement claimed that the misclassified points were all on the wrong side of the true synthetic separator, but this was not backed by an executable analysis.

I added an explicit failure-analysis function that:

- selects the exact configuration;
- finds the misclassified test examples;
- compares the scratch model and sklearn error sets;
- checks the points against the true generating separator.

The result showed that the scratch model and sklearn misclassified the **same 15/60 test points**, and 12 of those 15 lay on the wrong side of the true generating separator.

This is much stronger evidence that the errors are driven by data overlap than simply observing that the scratch loss curve looks stable.

## Statistical reporting improved

I changed repeated-run standard deviations to sample SD:

```python
np.std(values, ddof=1)
```

and made the report explicit that:

- \(\pm1\) SD bars are descriptive;
- they are not confidence intervals;
- non-overlapping SD bars are not being treated as a formal significance test.

## AI tool use and verification

This week AI use became more explicitly adversarial.

Claude helped revise the implementation and report. ChatGPT then challenged several of the revised statements, especially:

- soft-margin KKT logic;
- uniqueness claims;
- exact convergence language;
- whether constraint preservation alone proves the SMO algorithm is correct;
- whether reported claims had executable evidence.

I checked those criticisms against the equations and code before accepting them.

## What I learned this week

The main lesson was that **technical precision matters most in the places that look superficially obvious**.

I learned to distinguish:

- \(\alpha_i>0\) from \(0<\alpha_i<C\);
- convexity from uniqueness;
- numerical stability from exact convergence;
- feasibility checks from full optimiser correctness;
- a narrative claim from an executable result.

## Plan for next week

Next week I will:

- extend one \(C\)-sweep to the real Breast Cancer dataset;
- improve the interpretation of H2, H3, and H4;
- explicitly validate the RBF \(\gamma\);
- use paired comparisons where the same splits are reused;
- strengthen the all-regime KKT diagnostics.

---

# Implementation Log — Week 4

## Progress this week

This week I focused on **making the evaluation match the research question more closely**.

The implementation was already functioning, but I wanted the experiments to show not only whether the classifier worked, but what the SVM components were actually doing.

## H2 refined rather than simply accepted or rejected

My original H2 expected very small \(C\) to cause obvious underfitting in both training accuracy and margin compliance.

The results did not support that simple version.

Even at \(C=0.01\):

- low-overlap training accuracy was already high;
- high-overlap training accuracy changed very little across the entire \(C\) range.

However, hinge loss changed dramatically.

This meant that H2 was not simply "wrong." Instead, it contained two different predictions:

1. **0-1 accuracy component:** not supported;
2. **margin-compliance / hinge-loss component:** supported.

I therefore changed the conclusion to:

> **H2 is partially supported and refined.**

The stronger interpretation is that small \(C\) changes margin compliance substantially before it noticeably changes classification accuracy.

This is a more useful conclusion because it directly demonstrates why the training objective and evaluation objective are not identical.

## H4 reframed as an open empirical question

I originally treated support-vector count too much like a direct complexity measure.

The results contradicted that simple interpretation.

Within a single regime:

- support-vector ratio fell strongly as \(C\) increased;
- the generalisation gap did not fall correspondingly and in some cases increased slightly.

Therefore:

> more support vectors does not straightforwardly mean more overfitting.

The more reliable pattern was across data regimes:

- the high-overlap regime had many more support vectors and bound support vectors than the low-overlap regime at matched \(C\);
- this appeared to reflect **data difficulty** more clearly than a simple notion of model complexity.

I therefore changed H4 from a prediction into an investigated question:

> How do support-vector proportion and bound-support-vector proportion vary with \(C\) and overlap, and what do they actually track?

This made the result more informative than trying to force the original interpretation.

## Real-dataset \(C\)-sweep added

I extended the primal \(C\)-sweep to the **Breast Cancer Wisconsin Diagnostic dataset**.

The goal was not to turn the project into an application study. The real dataset was used as a compact check that the regularisation/margin behavimy observed in the synthetic experiments also appears in a genuine 30-dimensional problem.

The real-data results showed:

- margin decreased monotonically as \(C\) increased;
- training accuracy increased slightly;
- test accuracy peaked before the largest \(C\);
- the train-test gap generally increased at larger \(C\).

This strengthened H1 and gave additional support for the direction of H3.

## Paired H3 comparison

Because the same five split seeds were reused at each \(C\), the real-data comparison could be treated as paired.

For \(C=0.01\) versus \(C=10\), all five split-wise changes in generalisation gap were positive.

The paired increases were approximately:

\[
0.0498,\;
0.0030,\;
0.0205,\;
0.0205,\;
0.0293.
\]

The mean paired increase was approximately:

\[
0.0246
\]

with sample SD:

\[
0.0170.
\]

I did not turn this into a formal hypothesis test. I used it as descriptive evidence because the project was not designed with a formal inferential/statistical power analysis.

## Validation of RBF \(\gamma\)

I added a validation grid for the RBF kernel width:

\[
\gamma
\in
\{0.1,0.5,1,2,5\}
\]

at a fixed reference \(C=1\).

Validation accuracy increased from approximately:

- 0.800 at \(\gamma=0.1\);
- 0.933 at \(\gamma=0.5\);
- 0.967 at \(\gamma=1\);

and then plateaued.

I selected \(\gamma=1\) and froze it before the H5 test sweep.

This made the role of the validation set much clearer:

- \(C\) is deliberately varied because it is the experimental variable;
- \(\gamma\) is selected using validation because it is not the variable under study.

## AI tool use and critical review

I used Claude during the validation and experiment revision process, and I used ChatGPT to review whether the resulting interpretations were statistically and mathematically defensible.

This week the main AI-assisted corrections were not code-generation tasks. They were interpretation tasks:

- not overstating the real-data H3 result;
- using the paired design correctly;
- not treating \(\pm1\) SD as formal statistical significance;
- not treating bound-SV count as a direct overfitting measure.

## What I learned this week

This week reinforced that a hypothesis does not need to be fully supported to make the project stronger.

In particular:

- H2 became more informative after being split into accuracy and margin-compliance components;
- H4 became more interesting after the naive support-vector interpretation failed;
- H3 became more credible after the conclusion was weakened to match the uncertainty.

## Plan for next week

Next week I will focus on:

- KKT diagnostics for all three \(\alpha\) regimes;
- resolving the support-vector-count discrepancy as precisely as possible;
- final theory-to-code consistency;
- removing any remaining unsupported or overconfident claims.

---

# Implementation Log — Week 5

## Progress this week

This week I concentrated on the **dual solver and KKT diagnostics**, because that is the most technically vulnerable part of the project.

The main unresolved issue remained the support-vector discrepancy between the scratch SMO solver and sklearn.

## Tolerance hypothesis tested correctly

The original discrepancy was:

- scratch solver: 19 support vectors;
- sklearn: 17 support vectors.

I reran the support-vector threshold experiment on the **exact same fit** that produced this discrepancy.

Changing the support-vector threshold across:

\[
10^{-3},\;
10^{-5},\;
10^{-7}
\]

still produced exactly 19 scratch support vectors.

There were no \(\alpha_i\) values sitting in the intermediate numerical region that could explain the discrepancy.

Therefore:

> the threshold used to define a support vector is not the cause.

## KKT diagnostics extended to all three multiplier regimes

I expanded the diagnostic logic so that it checks all three \(\alpha_i\) regimes.

### Non-support vectors

For:

\[
\alpha_i=0
\]

the KKT conditions imply:

\[
\xi_i=0
\]

and therefore:

\[
y_i f(x_i)\ge1.
\]

### Free support vectors

For:

\[
0<\alpha_i<C
\]

the KKT conditions imply:

\[
\xi_i=0
\]

and:

\[
y_i f(x_i)=1.
\]

### Bound support vectors

For:

\[
\alpha_i=C
\]

the point must satisfy:

\[
y_i f(x_i)=1-\xi_i\le1.
\]

The implemented KKT report now checks:

- box-constraint violation;
- equality-constraint residual;
- non-SV margin violation;
- free-SV margin residual;
- bound-SV margin violation.

## What the KKT diagnostics revealed

On the problematic 19-vs-17 fit:

- box constraints were satisfied;
- \(\sum_i\alpha_i y_i\approx0\);
- non-support-vector margin violations were zero;
- bound-support-vector margin violations were zero;
- the free-support-vector margin residual was approximately **0.044** instead of approximately zero.

This localised the issue.

The scratch solver is not generally violating all constraints. Instead, the imperfection appears specifically in the free-SV boundary condition.

This gave a much more defensible explanation:

> the simplified SMO routine has not reached the exact optimum under the current stopping/working-set strategy.

That is stronger than my earlier speculative explanation that sklearn and the scratch solver had simply found different equally valid optima.

## KKT logic for \(\alpha_i=0\) corrected fully

A final subtle mistake remained in one draft.

I had treated:

\[
\alpha_i=0
\]

as only "typically" implying:

\[
y_i f(x_i)\ge1.
\]

The correct derivation is stronger.

Since:

\[
(C-\alpha_i)\xi_i=0
\]

and \(\alpha_i=0\), I get:

\[
C\xi_i=0
\Rightarrow
\xi_i=0.
\]

Then primal feasibility gives:

\[
y_i f(x_i)\ge1.
\]

So this is a forced KKT consequence, not just a common case.

This correction was applied in the report, notebook markdown, and code comments.

## Dual objective wording corrected

The scratch solver reports a dual objective value, but I realised that a standalone objective number does not prove correctness unless it is compared with an independent reference.

I therefore stopped using the dual objective as standalone verification.

The verification story now combines:

- feasibility constraints;
- KKT residuals;
- prediction agreement;
- decision-score agreement;
- support-vector comparison;
- objective monitoring.

## Duplicate/conflicting-label claim softened

Another small theory correction involved duplicate observations with conflicting labels.

The earlier draft made the universal claim that both would become bound support vectors with:

\[
\alpha_i=C.
\]

This is too strong because the exact optimum can depend on degeneracy and the rest of the training set.

The corrected interpretation is:

> conflicting labels cannot both satisfy a hard margin; the soft-margin formulation handles them through positive slack, and they are likely to become active or bound support vectors depending on the optimum.

## AI tool use and critical review

This week was a good example of why I documented AI use explicitly.

Claude revised the KKT discussion several times. ChatGPT independently checked the derivations and caught the remaining \(\alpha_i=0\) logic issue.

I did not resolve the disagreement by choosing one AI tool over the other. I worked through:

\[
(C-\alpha_i)\xi_i=0
\]

myself and confirmed the implication.

## What I learned this week

The main lesson was that KKT conditions are not only theoretical decoration. They can be used as **numerical diagnostics** to identify where an optimiser is imperfect.

I also learned to separate:

- preserving feasibility;
- satisfying stationarity/KKT;
- reaching the exact optimum;
- matching predictive behaviour.

These are related but not identical ideas.

## Plan for next week

Next week I will perform the final technical consistency pass:

- check validation wording;
- check every theory-to-code mapping;
- remove reviewer/draft comments from the public notebook;
- make sure all figures/tables use the same statistics;
- freeze the ML implementation if no new correctness issue appears.

---

# Implementation Log — Week 6

## Progress this week

This week was the **final technical verification and freeze** of the machine-learning part of the project.

The main goal was no longer to add features. It was to make sure that the report, code, figures, and experimental claims all said the same thing.

## Final validation protocol confirmed

The final validation scheme is:

### Primal learning-rate behaviour

The original fixed learning-rate schedule produced misleadingly good validation accuracy at large \(C\) even though the loss history was clearly unstable.

Therefore I did not select the schedule based only on validation accuracy.

I selected the `c_scaled` schedule because its optimisation trajectory was numerically stable enough for the experiment.

### SMO pass budget

Validation accuracy was unchanged from 5 passes onward.

Support-vector count stabilised by about 10 passes.

I kept:

```text
max_passes = 15
```

as a conservative buffer.

### RBF \(\gamma\)

The validation grid was:

\[
\gamma
\in
\{0.1,0.5,1,2,5\}.
\]

The best validation accuracy was reached from \(\gamma=1\) onward, so \(\gamma=1\) was selected and frozen before the H5 test sweep.

This gave me a clear answer for the viva:

> \(C\) was not tuned away because \(C\) is the variable being studied. Validation was used to freeze implementation settings and the RBF kernel width before test evaluation.

## Final hypothesis conclusions

### H1 — supported

Increasing \(C\) reduced the margin width strongly and consistently.

This was the lowest-variance finding in the project.

### H2 — partially supported / refined

Very small \(C\) did not greatly reduce classification accuracy, but it produced much larger hinge loss.

This demonstrates that margin compliance changes before 0-1 accuracy changes.

### H3 — suggestive on synthetic data; stronger support on real data

The synthetic high-overlap result showed only a small mean increase in generalisation gap relative to seed variability.

The real-data paired result showed a more consistent increase.

The final wording remains cautious because one dataset and five resplits are not enough to claim a universal effect.

### H4 — naive support-vector complexity story rejected

Support-vector count does not monotonically track overfitting within a \(C\)-sweep.

It is more useful as an indicator of which points actively constrain the solution and, across regimes, of data difficulty.

### H5 — supported

The RBF kernel substantially outperformed the linear kernel on two-moons because the curved boundary cannot be represented by a linear hypothesis class.

## Final implementation verification

The final project uses multiple independent checks.

### Primal SVM

- prediction agreement with sklearn;
- decision-score comparison;
- bias comparison;
- weight-vector cosine similarity;
- weight-norm comparison;
- loss-history stability.

At high \(C\), the scratch and sklearn weight vectors were still almost parallel, with cosine similarity close to 1.

### Dual SMO

- box constraints;
- equality constraint;
- KKT diagnostics for all three \(\alpha\) regimes;
- prediction agreement;
- decision-score comparison;
- support-vector comparison;
- dual objective monitoring.

The remaining free-support-vector residual is documented instead of hidden.

## Final theory-to-code consistency pass

I checked that the report and code now agree on:

- `sign(0)` handling;
- zero initialisation;
- full-batch subgradient descent;
- the non-uniqueness caveat for \((w,b)\);
- KKT interpretations;
- the role of \(\gamma\);
- support-vector definitions;
- validation usage;
- sample SD (`ddof=1`);
- the role of the real dataset;
- the difference between numerical stabilisation and exact convergence.

I also removed reviewer-number comments and other drafting fingerprints from the public-facing notebook/code so that the final notebook reads as a completed implementation rather than a revision transcript.

## AI tool use and critical review

The final project used two AI systems in different roles:

- **Claude** was used for implementation support, technical discussion, and drafting assistance.
- **ChatGPT** was used as an independent reviewer across successive versions.

The important point was that AI suggestions were treated as claims to verify rather than as authoritative outputs.

Examples of issues identified or corrected through AI-assisted review include:

- validation being claimed but not actually used;
- standard deviation being understated;
- overconfident H3 interpretation;
- incorrect KKT implications;
- an unsupported "different valid optimum" explanation;
- incorrect uniqueness/convergence wording;
- support-vector count being treated too simply as model complexity;
- a failure-case claim that originally had no executable evidence.

These corrections are part of the implementation history, not just editing.

## Knowledge gaps identified before final submission

There are still areas I do not claim to have mastered fully.

### Full SMO convergence and working-set theory

I understand the simplified pair update and KKT logic used in the project, but I have not independently derived the full convergence theory and working-set heuristics of Platt's complete SMO or libsvm.

### RKHS/generalisation theory

I understand the practical role of margin, \(C\), and kernels, but I have not independently derived the strongest radius-margin or RKHS generalisation bounds.

### Formal statistical inference

I understand repeated seeds, sample SD, and paired descriptive comparisons, but this project was not designed as a formal hypothesis-testing study with power analysis or confidence intervals.

These gaps are documented because they define the boundary between what I can explain confidently and what I would need to study further.

## What I learned this week

The main lesson from the final verification is that the strongest part of this project is not a single accuracy number.

The most important outcome is the connection between:

- the SVM objective;
- the geometry of the margin;
- the role of \(C\);
- the KKT conditions;
- the behavimy of support vectors;
- the difference between surrogate loss and task accuracy;
- and the numerical behavimy of the actual implementation.

The project is now technically frozen unless a new correctness problem is discovered.

## Plan for Week 7

Week 7 will be completed only after the final submission workflow is finished.

It will document:

- public Colab creation;
- clean-runtime execution;
- final PDF export;
- final filename/link checks;
- final reflection on AI-assisted development;
- A3 presentation preparation;
- viva questions and what I learned from preparing to defend the work.

---

# Implementation Log — Week 7

## Extending the project beyond the original H1–H5 design

This week I focused on the gap between a technically correct implementation and a stronger final project contribution.

By the end of Week 6, the core SVM implementation had already been technically frozen. However, the H4 results raised a new question that the original design did not answer well.

Raw support-vector ratio fell as \(C\) increased, even though weaker regularisation did not correspond to a simple improvement in the generalisation gap. This made it clear that support-vector count alone was difficult to interpret as a direct complexity measure.

After discussing the project standard and possible extensions with AI tools and reviewing the existing evidence, I decided not to redesign the project. Instead, I extended the existing H4 analysis with a new hypothesis focused on **support-vector composition**.

## H6 — support-vector composition

I introduced two diagnostics derived directly from the fitted dual variables:

\[
BVR
=
\frac{n(\alpha_i=C)}
     {n(\alpha_i>0)}
\]

and

\[
NDP
=
\frac{1}{n}
\sum_i
\frac{\alpha_i}{C}.
\]

I use BVR as the fraction of support vectors saturated at the upper bound \(C\), while NDP measures the average normalised dual weight across the whole training set.

The motivation was not to claim a new general SVM complexity theory. The aim was narrower:

> if raw support-vector count loses information, does the composition of the support-vector set reveal something more useful about how the fitted solution is using its active constraints?

## Main H6 result on synthetic data

The extension produced a clear difference between the low- and high-overlap regimes.

At \(C=100\):

- low-overlap BVR was approximately 0.427;
- high-overlap BVR was approximately 0.909.

This means that the two models differed not only in how many support vectors remained, but in what kind of support-vector set they retained.

In the high-overlap regime, most support vectors remained pinned at the upper bound, indicating persistent active margin/slack pressure. In the low-overlap regime, a much larger fraction of the remaining support vectors were free rather than saturated.

I also found that BVR and NDP are not redundant.

NDP/SV-ratio represents the average \(\alpha_i/C\) among support vectors, while BVR counts only the fraction exactly at \(C\). In the high-overlap regime these values remain close because most support vectors are bound. In the low-overlap regime they diverge as \(C\) increases, showing that the free support vectors still carry partial weight that is invisible to a simple bound/free count.

## Sweep F — real-data support-vector composition

I extended the composition analysis to the Breast Cancer Wisconsin dataset using the dual solver.

The real-data results across:

\[
C\in\{0.01,0.1,1\}
\]

were:

- SV ratio: \(0.242 \rightarrow 0.118 \rightarrow 0.078\);
- BVR: \(0.901 \rightarrow 0.738 \rightarrow 0.363\);
- NDP: \(0.231 \rightarrow 0.103 \rightarrow 0.045\).

This showed that the decline in saturation was not confined to the constructed 2D example.

I did not claim that the real dataset has the same geometry as the low-overlap synthetic regime. I treated the result only as evidence that the composition trend can appear outside the synthetic setup.

## New solver-scaling limitation discovered

Sweep F exposed a new implementation limitation.

The simplified random-pair SMO solver scaled increasingly poorly as \(C\) increased on the 30-feature Breast Cancer dataset.

Measured outer-iteration counts were approximately:

- \(C=0.01\): 193;
- \(C=0.1\): 925;
- \(C=1\): 6,461;
- \(C=10\): more than 20,000 without meeting the stopping criterion.

Because of this, I restricted Sweep F to:

\[
C\in\{0.01,0.1,1\}.
\]

I documented this rather than forcing a very slow or loosely converged \(C=10\) result into the report.

This became a useful limitation because it showed a practical consequence of using a simplified random working-set strategy instead of a production SMO implementation with more efficient heuristics.

## Publication-style report refinement

This week I also restructured the report so that it reads as a technical project report rather than as a chronological diary.

The main changes included:

- clearer separation between theory, experimental design, results, and discussion;
- a theory-to-code mapping table;
- explicit hypothesis-by-hypothesis conclusions;
- stronger distinction between supported, partially supported, and suggestive findings;
- clearer limitations and future work;
- consistent reporting of sample SD;
- consistent figure and table captions;
- Australian English spelling throughout the report;
- APA-style references;
- a concise Implementation Log instead of embedding the full weekly journal.

I kept the weekly journal separate because its purpose is to document the development sequence and reflection, while the final report is designed to present the finished technical argument cleanly.

## AI-tool use and critical review

This week I used Claude and ChatGPT mainly as review partners rather than as sources of authoritative answers.

I used AI discussions to:

- test whether H6 was a defensible extension;
- check whether BVR and NDP were being overclaimed;
- challenge the interpretation of raw support-vector count;
- review the wording of limitations;
- identify places where the report still sounded like a development transcript rather than a final technical document.

I retained responsibility for deciding which suggestions were accepted.

## What I learned this week

The main lesson was that extending a project does not require adding an unrelated new model.

The strongest extension came from following an unresolved question already present in my own results.

H4 showed that raw support-vector count did not behave as simply as expected. H6 then used the same fitted dual variables to investigate the internal composition of that support-vector set.

This made the final project more coherent because the extension grew directly from the evidence rather than being added only for scope.

## Plan for Week 8

The final week will focus on reproducibility and submission preparation:

- public GitHub repository;
- public Colab notebook;
- clean-runtime execution;
- provenance checks for numerical claims;
- final code cleanup;
- final report consistency checks;
- final PDF preparation;
- preparation for the A3 defence.


---

# Implementation Log — Week 8

## Final reproducibility and submission preparation

This week I focused on closing the gap between the technically complete project and the final submission-ready version.

The objective was no longer to develop new SVM functionality. Instead, I checked whether every important claim in the report could be traced back to executable code, whether the public implementation could be reproduced from a clean environment, and whether the report, notebook, source files, figures, and experimental outputs were mutually consistent.

## Public GitHub repository and Colab notebook

I created a public GitHub repository containing the complete modular implementation:

- `01_data.py`
- `02_primal_svm.py`
- `03_dual_smo.py`
- `04_experiments.py`
- `05_make_figures.py`
- `SVM_Project_Notebook.ipynb`
- `sweep_results.json`
- `requirements.txt`

I then connected the notebook to Google Colab so that the implementation could be accessed through the public notebook URL required by the assignment specification.

I performed a clean Colab runtime test using **Run all**.

The complete workflow executed successfully:

- implementation-setting validation;
- Sweep A — primal \(C\)-sweep;
- Sweep B — dual SMO / support-vector sweep;
- Sweep C — linear vs. RBF kernel comparison;
- Sweep D — support-vector tolerance / KKT diagnostics;
- Sweep E — Breast Cancer primal sweep;
- Sweep F — Breast Cancer support-vector composition;
- failure-case analysis;
- verification checks;
- generation of Figures 1–5.

All five figures were reproduced successfully in the clean runtime.

This gave me stronger confidence that the submitted implementation was not dependent on my local machine or on undocumented intermediate files.

## Closing a verification provenance gap

During final review, one report claim was identified as numerically correct but insufficiently traceable in the public implementation.

The report contained additional \(C=100\) primal-vs-sklearn geometry checks.

I therefore reproduced the comparisons explicitly and added an executable verification function.

For low-overlap, \(C=100\), seed 0:

- scratch \(\|w\|\approx5.729\);
- sklearn \(\|w\|\approx5.804\);
- cosine similarity \(\approx0.99999936\);
- scratch/sklearn test accuracy \(=0.933/0.933\).

For high-overlap, \(C=100\), seed 0:

- scratch \(\|w\|\approx1.427\);
- sklearn \(\|w\|\approx1.428\);
- cosine similarity \(\approx0.99999900\);
- scratch/sklearn test accuracy \(=0.750/0.750\).

I added this check to both the experiment source and the notebook so that the report claim could be reproduced directly rather than existing only as a manually recorded value.

I also corrected the broader \(C=1\) verification wording from an overly narrow cosine-similarity range to:

> greater than 0.9998 across all three checked configurations.

## Final code cleanup

Before publishing the repository, I reviewed the source files for comments and docstrings that reflected the drafting/review process rather than the final implementation.

I shortened several overly defensive or review-oriented comments, but deliberately did not change:

- model algorithms;
- hyperparameters;
- experiment grids;
- outputs;
- plotting logic;
- stored sweep results.

I also corrected several stale descriptions:

- the data module originally described only two regimes even though the final project uses four;
- the experiment module described only three sweeps even though the final implementation contains Sweeps A–F;
- the Figure 2 comment referred to the wrong second-panel quantity and was corrected to bound-SV count.

The purpose of this cleanup was to make the public code read like a completed implementation without changing its computational behaviour.

## Final report consistency checks

I performed a final report-to-code consistency pass.

The main checks included:

- the public Colab URL;
- the GitHub repository URL;
- correct reporting of the \(C=100\) geometry checks;
- consistent use of the narrower Sweep F \(C\)-grid;
- corrected wording that Sweep F uses the shared portion of Sweep E's primal grid rather than the complete Sweep E grid;
- consistent table numbering and captions;
- consistent figure numbering and captions;
- correct table-of-contents page numbers;
- internal TOC links to major report sections;
- consistent terminology across the report and notebook.

The final report remained publication-style. I kept the weekly journal separate and retained only the concise Implementation Log required in the report itself.

## AI-tool reflection

The final phase again demonstrated why AI-assisted suggestions needed verification.

During an AI-assisted final review, a provenance issue was identified in one of the \(C=100\) verification claims. The numerical claim turned out to be correct, but the public implementation initially did not contain an executable path that reproduced it.

I reproduced the experiment, used ChatGPT to review the implementation path, and then added the verification code to the project after confirming the result.

This distinction became important to me:

> a number can be correct and still be inadequately documented.

I therefore treated reproducibility as part of correctness rather than as a separate presentation issue.

## Final reflection

The biggest change from the original Week 1 plan is that the project became less about proving a simple statement such as "larger margin generalises better" and more about understanding why that statement is incomplete.

The final experiments showed that:

- \(C\) changes SVM geometry very reliably;
- hinge loss can change substantially while classification accuracy barely changes;
- generalisation effects are more data-dependent than margin effects;
- raw support-vector count alone is difficult to interpret;
- support-vector composition provides additional information about the fitted dual solution;
- optimisation diagnostics are essential because plausible-looking outputs can still come from an unstable or incompletely converged solver.

Several of my original assumptions were revised rather than protected, and I consider that one of the strongest parts of the project.

## Status at the end of Week 8

At the end of this week:

- the machine-learning implementation is complete;
- the GitHub repository is public;
- the Colab notebook executes successfully from a clean runtime;
- all six experimental sweeps run successfully;
- Figures 1–5 reproduce correctly;
- the final report is consistent with the implementation;
- the final PDF submission is ready;
- presentation materials are being prepared separately for the A3 defence.

The remaining work is no longer model development. My next priority is to prepare a concise 5-minute presentation and make sure I can explain and defend the implementation, experimental decisions, limitations, and results during the A3 question-and-answer session.
