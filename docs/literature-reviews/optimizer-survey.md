# Decision Memo: Optimizer / update-scheme choice for PyTopo3D 0.3.0

**Date:** 2026-06-16
**Scope:** What optimizer should PyTopo3D adopt for the 0.3.0 roadmap items *Mass Minimization* and *Improved Convergence Methods*, and to enable the later (beyond-roadmap) constrained problems (stress, manufacturability, multiphysics)?
**Purpose:** Decision support. Is "just use MMA" right, or is there a more elegant / better-fitting choice?
**Method:** OpenAlex systematic discovery (primary literature) + primary-source verification of license terms (GitHub `LICENSE` files, PyPI) and commercial-tool optimizers (vendor docs). Every load-bearing claim is tagged with its source; unverifiable vendor internals are flagged.

---

## Executive summary

**Most-used ≠ most-flashy, and here they reconcile: MMA *is* the right call — with two corrections to the naive "just `pip install` an MMA package" version.**

1. **Adopt MMA, but via NLopt's MIT reimplementation — not the Svanberg/Deetman code.** The entire Svanberg lineage (his reference MATLAB, the popular `GCMMA-MMA-Python` port, the `mmapy` wheel) is **GPL-3.0** and cannot be bundled into a permissive public package. NLopt ships an **independent MIT-licensed MMA/CCSA** written from Svanberg's 2002 paper — this dissolves the license objection that would otherwise count against MMA. *(This corrects an earlier assumption that "MMA = unavoidable GPL baggage.")*

2. **"Improved Convergence" is a separate, cheaper sub-decision that does *not* need MMA.** The convergence aids (penalty continuation, gray-scale / Heaviside-robust projection) are documented in PyTopo3D's own source paper (Liu & Tovar 2014) as drop-in extensions the base code never included. They are low-risk and license-free, but are brought in **conditionally — coupled to mass minimization**, only where an observed need justifies them (see `0.3.0-implementation-plan.md`, which supersedes the "ship convergence first" framing below). MMA is required by *mass minimization* (a nonlinear constraint), not by convergence.

3. **The "more elegant modern" alternative (neural / differentiable TO) does not win on merit.** A Sigmund-group review reading the primary NN-TO literature (Woldseth et al. 2022) concludes neural reparameterization "involves no actual learning" — it is a conventional optimizer in disguise, with speed-ups that are really an implicit coarser filter. Autodiff's genuine value is *computing sensitivities*, which PyTopo3D already has analytically. So the modern-looking path adds cost without a demonstrated quality gain here.

**Recommendation:** Treat 0.3.0 as two decisions. **(A)** Improve convergence by porting continuation + projection (no new optimizer, no license cost). **(B)** Add mass minimization (and lay the substrate for stress/multiphysics) with **MMA via the `nlopt` MIT dependency**, consuming PyTopo3D's existing analytic sensitivities. Keep **IPOPT (cyipopt, EPL-2.0)** as the documented fallback if a full general NLP solver is later wanted. Do **not** vendor any GPL MMA, and do not pursue NN/autodiff as *the optimizer*.

---

## 1. The question, split correctly

The roadmap lists "Mass Minimization" and "Improved Convergence Methods" as two bullets. They impose very different optimizer requirements:

| Roadmap item | Does it need a new optimizer? | Why |
|---|---|---|
| **Improved Convergence** | **No.** | Achievable with formulation-side aids (continuation on penalty, gray-scale / robust Heaviside projection) on top of the existing OC. These are in PyTopo3D's own source paper as extensions. |
| **Mass Minimization** | **Yes.** | Puts a *nonlinear* quantity (compliance/stress) into the **constraint**. The classic OC update is a single-volume-constraint heuristic (one Lagrange multiplier found by bisection) and does not generalize. A general constrained optimizer is required. |

This split is the crux: "just use MMA" answers (B) but over-serves (A).

---

## 2. Candidate landscape

Density-based (SIMP) TO optimizers cluster into four families. Citations are OpenAlex `cited_by_count` as of 2026-06.

### 2.1 Optimality Criteria (OC) and generalizations
The classic update used by the educational MATLAB lineage — Sigmund's 99-line, Andreassen et al.'s 88-line (`W2141377530`, 1558 cites), and **Liu & Tovar's `top3d` (`W2048785994`, 622 cites) which PyTopo3D ports**. OC is derived from the KKT conditions of *single-volume-constrained compliance minimization*; it is tiny, dependency-free, and cheap (the bisection re-evaluates only volume, never the FE solve). It does **not** extend to a nonlinear constraint. "Generalized OC" variants exist (`W3140833829`, 2021, 36 cites) but remain niche.

### 2.2 Method of Moving Asymptotes (MMA) and GCMMA — the field workhorse
- **MMA** (Svanberg 1987, `W2062523101`, **5401 cites**, full-text abstract verified): each iteration builds and solves "a strictly convex approximating subproblem … controlled by so-called 'moving asymptotes', which may both stabilize and speed up the convergence." Handles arbitrary nonlinear objective + multiple nonlinear constraints.
- **GCMMA** (Svanberg 2002, `W2153245297`, 1342 cites, abstract verified): globally convergent variant. Abstract states it is **proven to converge to KKT points**, each iterate feasible with lower objective, and — decisive for TO — "can be applied to problems with a very large number of variables (say 10⁴–10⁵) even if the Hessian matrices … are dense."
- **Scale evidence:** Aage et al. (2017, *Nature*, `W2761536263`, 742 cites) used MMA with PETSc for *giga-voxel* (~10⁹-element) structural design — proof MMA is the large-scale production tool, not just a textbook method.

### 2.3 General-purpose NLP solvers fed by sensitivities
IPOPT (interior-point), SNOPT/SLSQP (SQP), NLopt. "Don't write an optimizer — hand the problem (objective + constraints + analytic gradients) to a battle-tested NLP solver." Common in research stacks (e.g. dolfin-adjoint/pyadjoint + IPOPT). Elegant and general; the trade-off is dependency/license weight (see §4) and, for dense SQP like SLSQP, poor scaling to 10⁴–10⁶ design variables.

### 2.4 Differentiable / neural topology optimization
- **Neural reparameterization** (Hoyer et al. 2019, arXiv:1909.04240): a CNN re-parameterizes the density field, optimized by L-BFGS.
- **Direct NN design** (TOuNN, Chandrasekhar & Suresh 2020, `W3106486741`, 233 cites): a network *is* the design representation.
- **Autodiff frameworks**: JAX-FEM (`W4379196907`, 2023, 97 cites), AuTO (`W3140287683`, 2021).
- **Critical appraisal** (Woldseth, Aage, Bærentzen & Sigmund 2022, `W4298156653`, arXiv:2208.02563, 187 cites — full text read): strongly skeptical. Direct-design NNs give "poor designs, … expensive to obtain and … very restricted." Reparameterization "is not subjected to any actual learning … an example of using an ANN without the learning aspect"; apparent iteration savings may be "a result of the re-parameterisation causing a perceived larger filter radius or coarser mesh." Verdict: NN approaches today do **not** beat classical gradient-based density TO; their promise is *accelerating* the conventional loop (surrogate FEA), not replacing the optimizer.

### 2.5 Why non-gradient (GA/PSO) is out of scope
Sigmund 2011 (`W2056802356`, 453 cites; abstract verified, body paywalled): gradient-based TO solves "thousands and up to millions of design variables using a few hundred … function evaluations (and even less than 50 in some commercial codes)," whereas non-gradient methods "require orders of magnitude more function evaluations for extremely low resolution examples." Genetic-algorithm / particle-swarm optimizers are ruled out for the high-dimensional 3D regime PyTopo3D targets.

---

## 3. What commercial tools actually use (verified from vendor docs)

This is the "commercial alignment" criterion, and it is decisive: **MMA (or an MMA-derived method) is the verified de-facto backbone for general/sensitivity-based topology optimization in 3 of 4 major tools.**

| Tool | Method | Optimizer (verbatim from vendor doc) | Source |
|---|---|---|---|
| **COMSOL** Optimization Module | density-based | Gradient solvers: "SNOPT, IPOPT, MMA, and Levenberg–Marquardt." The MMA impl "is the globally convergent version … referred to as GCMMA." | doc.comsol.com |
| **Altair OptiStruct** | SIMP | Topology **default = DUAL2**, "an enhanced version of the proprietary CONLIN dual optimizer"; MMA, MFD, SQP, DUAL also offered. | help.altair.com |
| **Dassault SIMULIA Tosca / Abaqus** | condition-based + sensitivity-based | Sensitivity-based path "uses an algorithm based on the Method of Moving Asymptotes from Krister Svanberg." Fast path = controller (optimality-criteria). | docs (Abaqus 2024) |
| **ANSYS Mechanical** | SIMP / level-set / mixable | OC "for a simple compliance objective … volume or mass constraint"; default class = **Sequential Convex Programming (SCP), "an extension of the method of moving asymptotes (MMA)."** | ansyshelp.ansys.com (2025 R1) |

**Reading:** the *general constrained* path is MMA-or-MMA-derived almost everywhere (COMSOL GCMMA, Tosca MMA, ANSYS SCP-extends-MMA); the *fast/simple* path is OC (Tosca controller, ANSYS OC). Altair is the lone exception (proprietary CONLIN/DUAL2 — disclosed family, undisclosed internals). This mirrors the split in §1 exactly: OC for the simple volume-constrained case, MMA once constraints get richer.

---

## 4. License audit (verified from primary `LICENSE` files / PyPI)

For a permissive (MIT/BSD) public PyPI package, the optimizer's license is a tier-① constraint.

| Implementation | License (verified) | Class | Verdict for PyTopo3D |
|---|---|---|---|
| Svanberg MMA/GCMMA (smoptit.se) | **GPL-3.0** | strong copyleft | ❌ cannot vendor |
| `GCMMA-MMA-Python` (Deetman) | **GPL-3.0** | strong copyleft | ❌ cannot vendor |
| `mmapy` (PyPI) | **GPL-3.0** | strong copyleft | ❌ cannot vendor |
| **NLopt MMA/CCSA** (`src/algs/mma`) | **MIT** — S. G. Johnson's independent reimpl "described in" Svanberg 2002 | permissive | ✅ **clean MMA** |
| `nlopt` (PyPI wheel) | MIT, or LGPL-2.1 if built with `luksan` | permissive / weak-copyleft | ✅ fine **as a dependency** (no copyleft propagation to your code) |
| IPOPT / `cyipopt` | **EPL-2.0** (+ MUMPS/HSL linear solver) | weak copyleft | ⚠️ ok as *optional external* dep; heavier |
| SciPy (`scipy.optimize` SLSQP) | **BSD-3-Clause** | permissive | ✅ clean (but SQP, not MMA) |

**Key finding:** the GPL problem is real but *avoidable*. NLopt's MMA is an MIT reimplementation independent of Svanberg's GPL code (verified from `src/algs/mma/README`: "It is under the same MIT license as the rest of my code in NLopt"). The "academic-only / by-request" lore about Svanberg's code is a courtesy request, not the license — the actual `smoptit.se` distribution is GPLv3, which is the genuine blocker for *vendoring*. Depending on the `nlopt` wheel does not impose copyleft on PyTopo3D's own code.

---

## 5. Scoring against the four criteria

Scale: ●●● strong · ●●○ moderate · ●○○ weak. Criteria: **(1)** elegance/modernity · **(2)** license/dependency cleanliness · **(3)** verification/commercial alignment · **(4)** future-roadmap generality.

| Candidate | (1) Elegance | (2) License | (3) Verified/commercial | (4) Generality | Net |
|---|:---:|:---:|:---:|:---:|---|
| OC + continuation/projection (port from `top3d`) | ●●○ | ●●● | ●●○ | ●○○ | **Best for convergence; useless for mass-min** |
| **MMA via NLopt (MIT)** | ●●○ | ●●● | ●●● | ●●● | **Best overall for mass-min + future constraints** |
| MMA/GCMMA via Svanberg/Deetman (GPL) | ●●○ | ●○○ | ●●● | ●●● | Ruled out on license |
| IPOPT via cyipopt (EPL) | ●●● | ●●○ | ●●● | ●●● | Strong alternative; heavier deps |
| SciPy SLSQP (BSD) | ●●○ | ●●● | ●●○ | ●●○ | Small problems only (dense SQP scaling) |
| Differentiable/autodiff TO (JAX) | ●●● | ●●● | ●○○ | ●●○ | Modern but unproven here; needs FE-backend rewrite |
| NN direct design (TOuNN-style) | ●●● | ●●● | ●○○ | ●○○ | Not competitive (Woldseth 2022) |

**Most-used vs most-elegant:** the most-used method (MMA) and the most-elegant *available, proven, license-clean* choice for this codebase coincide once you (a) source MMA from NLopt and (b) discount the flashy NN/autodiff path on the evidence. The only serious rival on pure elegance is **IPOPT** ("use a real NLP solver"), which loses on dependency/license weight (EPL + a MUMPS/HSL linear solver) but is the right documented fallback.

---

## 6. PyTopo3D-specific fit

- **Sensitivities already exist analytically** (`dc`, `dv` in `pytopo3d/core/optimizer.py`). OC, MMA (NLopt), IPOPT, and SLSQP all consume them directly — small, surgical integration. Differentiable/autodiff TO would require rewriting the FE core onto a differentiable backend (JAX/Torch) to obtain a value autodiff provides — sensitivities — that the package **already has**. Low return.
- **0/1 crispness** is handled by SIMP penalty + (to be added) projection, independent of optimizer choice.
- **API-freeze timing:** 0.4.0 freezes the API. Adopting a general optimizer interface (objective + list of constraints) in 0.3.0 means the mass-min/stress/multiphysics signature is settled *before* the freeze. MMA-via-NLopt and IPOPT both fit this interface; OC does not.
- **GPU path:** PyTopo3D's optional CuPy path accelerates the FE/CG solve, which dominates cost. The optimizer update (OC or MMA) is cheap by comparison, so an MMA on CPU (NLopt) does not bottleneck a GPU-accelerated solve.

---

## 7. Recommendation

> **Superseded on sequencing & integration (2026-06-16):** `docs/literature-reviews/0.3.0-implementation-plan.md` is the confirmed plan. It keeps this memo's *optimizer* conclusion (adopt MMA via NLopt) but (a) reverses item 1 — convergence aids are **conditional, coupled to mass minimization**, not shipped first as a standalone change — and (b) adopts NLopt **behind an optimizer-agnostic seam** (physics-as-callback), not as a bare runtime dependency, so the kernel can later be swapped for an owned/commercial engine.

1. **Convergence aids — conditional, coupled to mass-min (not a standalone first step).** Penalty-**continuation** and **gray-scale filter** that Liu & Tovar (2014) document as `top3d` extensions, and/or **robust Heaviside projection** (Wang/Guest/Sigmund lineage), are added only when mass-min or a manufacturability need exercises them. Re-measure the golden master afterward (a model change re-opens the baseline).
2. **Add Mass Minimization on MMA, sourced from `nlopt` (MIT).** Feed it the existing analytic sensitivities. This also unlocks the beyond-roadmap constrained problems (stress, manufacturability, multiphysics) that the roadmap parks for later — Woldseth et al. note MMA makes "switching the objective and constraint functions … straightforward."
3. **Document IPOPT (cyipopt, EPL-2.0) as the fallback** for anyone needing a full interior-point NLP solver; keep it an *optional* extra, not a core dependency.
4. **Do not** vendor GPL MMA, and **do not** pursue NN/autodiff as the optimizer (only consider autodiff later if/when a differentiable backend is wanted for a different reason).

**Net:** "just use MMA" was directionally right, but the decision that survives scrutiny is sharper — *MMA via NLopt's MIT implementation, kept separate from a cheaper convergence-only sub-step, with IPOPT as the named fallback.*

---

## Appendix A: Annotated source list (tiered)

**Tier 1 — core, read in full or abstract-verified:**
- Svanberg 1987, *The method of moving asymptotes* — `W2062523101`, DOI 10.1002/nme.1620240207 (abstract verified).
- Svanberg 2002, *GCMMA / conservative convex separable approximations* — `W2153245297`, DOI 10.1137/s1052623499362822 (abstract verified).
- Liu & Tovar 2014, *An efficient 3D topology optimization code (`top3d`)* — `W2048785994`, DOI 10.1007/s00158-014-1107-x (**full text read**; PyTopo3D's source).
- Woldseth, Aage, Bærentzen & Sigmund 2022, *On the use of ANNs in topology optimisation* — `W4298156653`, arXiv:2208.02563 (**full text read**).
- Sigmund 2011, *On the usefulness of non-gradient approaches* — `W2056802356`, DOI 10.1007/s00158-011-0638-7 (**abstract-only**; body paywalled — flagged).
- Aage et al. 2017, *Giga-voxel computational morphogenesis* — `W2761536263`, DOI 10.1038/nature23911 (abstract/metadata).

**Tier 2 — important context:**
- Sigmund & Maute 2013, *Topology optimization approaches* (review) — `W2220999736`, DOI 10.1007/s00158-013-0978-6.
- Andreassen et al. 2011, *88-line* — `W2141377530`. · Ferrari & Sigmund 2020, *new 99-line* — `W3024236378`, DOI 10.1007/s00158-020-02629-w.
- Chandrasekhar & Suresh 2020, *TOuNN* — `W3106486741`. · Hoyer et al. 2019, *Neural reparameterization* — arXiv:1909.04240.
- *Density-based TO with the Null Space Optimizer: a tutorial and comparison* 2024 — `W4390610057`, DOI 10.1007/s00158-023-03710-w.
- JAX-FEM 2023 — `W4379196907`. · AuTO 2021 — `W3140287683`. · Generalized OC 2021 — `W3140833829`.

**Commercial docs (primary):** COMSOL Optimization Module solver list (GCMMA); Altair OptiStruct gradient-method guide (DUAL2/CONLIN); Abaqus/Tosca optimization-task doc (MMA/Svanberg); ANSYS Structural Optimization guide 2025 R1 (SCP-extends-MMA, OC).

**License sources (primary):** smoptit.se; `arjendeetman/GCMMA-MMA-Python` LICENSE; `mmapy` PyPI; `stevengj/nlopt` `COPYING` + `src/algs/mma/README`; `coin-or/Ipopt` & `mechmotum/cyipopt` LICENSE; `scipy` LICENSE.txt.

## Appendix B: Search documentation
- **Discovery:** OpenAlex REST (works/citations/by-DOI), 4 lenses (seminal, recent, methods, critique) + forward-citation chase from Svanberg-MMA (`cites:W2062523101`). OpenAlex polite pool.
- **Gotcha logged:** OpenAlex `search=` combined with `sort=cited_by_count:desc` (or with no filter) silently returned globally-top-cited noise; the reliable patterns were `search=`+`filter=…` and `/works/doi:…`.
- **Verification:** licenses and commercial optimizers were not in OpenAlex; verified by primary-source web fetches (GitHub `LICENSE`, PyPI, vendor docs) via parallel agents, with the load-bearing NLopt-MMA-is-MIT claim re-checked directly against `src/algs/mma/README`.

## Limitations
- Sigmund 2011 body text is paywalled; only its abstract was verified (thesis is unambiguous there). Several Springer/SMO abstracts are absent from OpenAlex.
- Altair's CONLIN/DUAL2 internals are proprietary (family disclosed, formulation not). The `nlopt` wheel's exact license depends on its build (MIT vs LGPL with `luksan`); either is dependency-safe for a permissive package, but vendoring/modifying would need the build checked.
