# Close-seam-B: laminar vs argmax finite-size cutoff exponent (method-matched, reproducible)

Question: does the laminar-phase length share the argmax finite-size exponent z (paper: z=1.45+/-0.06)?

Pipeline (identical for both observables, reproducible):
- argmax: full-scale durations already in repo (reward_huge_v2/_v3_*_h{3,6,10,15,20}, T=2e5).
- laminar: re-simulated (NO posterior trajectory was stored in the Nh sweep) via
  prod_reruns/run_laminar_nhsweep_prod.py  (AgentReward 'bib', tie=skip/defeat=random_other,
  H_NUM in {3,6,10,15,20}, 4 designs, theta=0.4, both agents pooled; MODERATE scale T=4e4, 8 runs).
- cutoff = truncated-power-law 1/Lambda (powerlaw, discrete, Clauset x_min), rs design = the
  paper's own method (SI: argmax rs 1/Lambda = {6879,2751,1373,591,435}).

Results (rs design, 1/Lambda, Nh={3,6,10,15,20}):
- ARGMAX (paper cutoffs): z = 1.49 +/- 0.07 (R^2=0.99)  == reproduces paper z=1.45+/-0.06.  PIPELINE VALIDATED.
- LAMINAR (same method): 1/Lambda = {10269,2871,719,310,132}; z_lam = 2.30 +/- 0.10 (R^2=0.99)
  [excl-boundary {3,6,10,15}: 2.21 +/- 0.12, R^2=0.99].
- laminar alpha per Nh = {1.51,1.33,1.34,1.56,1.73}; core (Nh 6,10) alpha~1.34 == paper headline.
Cross-check (FSS data-collapse, pooled, fss_collapse.py):
- argmax z=1.53+/-0.06 (~paper 1.45), laminar z=2.05+/-0.05.  Same direction.

CONCLUSION: SEAM DOES NOT CLOSE. z_lam ~ 2.0-2.3 >> argmax z~1.45 (>6 sigma, clean R^2=0.99).
The two observables SHARE the 3/2-class tail but have OBSERVABLE-DEPENDENT finite-size cutoffs
(first-passage times of the same driftless walk to different absorbing conditions). Conservative
framing applied to main (Discussion) + SI (sweep section). Laminar number is a MODERATE-scale
matched re-analysis; full-scale (T=2e5) would refine the value but the >6-sigma gap is robust.

Reproduce: python prod_reruns/run_laminar_nhsweep_prod.py  (repeat until 20/20) ; then
           python prod_reruns/run_laminar_nhsweep_prod.py --fit ; python prod_reruns/fss_collapse.py
