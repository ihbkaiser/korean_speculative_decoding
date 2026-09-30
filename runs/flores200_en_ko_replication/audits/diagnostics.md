# Model diagnostics

All models use prompt-clustered covariance. Full diagnostic records are in `model_diagnostics.json`.

## Pair and pooled model status

### H3

- P1: converged=True, rank=29/29, clusters=1010, max |β|=1.7430860253882716, rank-deficient=False, extreme=False, warnings=[], error=None.
- P2: converged=True, rank=29/29, clusters=1012, max |β|=3.012637731383018, rank-deficient=False, extreme=False, warnings=[], error=None.
- P3: converged=True, rank=29/29, clusters=1012, max |β|=2.1343746626919096, rank-deficient=False, extreme=False, warnings=[], error=None.
- POOLED: converged=True, rank=33/33, clusters=1012, max |β|=2.0479636779926587, warnings=[], error=None.
- POOLED_COMMON: converged=True, rank=33/33, clusters=1010, max |β|=2.0483097382927675, warnings=[], error=None.

### H4 L1/L2/L3

- P1 L1: converged=True, rank=21/21, clusters=966, max |β|=2.0257877991623983, rank-deficient=False, extreme=False, warnings=[], error=None.
- P1 L2: converged=True, rank=32/32, clusters=966, max |β|=3.479056430161901, rank-deficient=False, extreme=False, warnings=['Maximum Likelihood optimization failed to converge. Check mle_retvals'], error=None.
- P1 L3: converged=True, rank=44/44, clusters=966, max |β|=5.590404591254424, rank-deficient=False, extreme=False, warnings=['Maximum Likelihood optimization failed to converge. Check mle_retvals', 'Maximum Likelihood optimization failed to converge. Check mle_retvals'], error=None.
- P2 L1: converged=True, rank=21/21, clusters=872, max |β|=3.388421315495594, rank-deficient=False, extreme=False, warnings=[], error=None.
- P2 L2: converged=True, rank=33/33, clusters=872, max |β|=4.982888512784959, rank-deficient=False, extreme=False, warnings=['Maximum Likelihood optimization failed to converge. Check mle_retvals', 'Maximum Likelihood optimization failed to converge. Check mle_retvals'], error=None.
- P2 L3: converged=True, rank=45/45, clusters=872, max |β|=5.31780315177343, rank-deficient=False, extreme=False, warnings=['Maximum Likelihood optimization failed to converge. Check mle_retvals', 'Maximum Likelihood optimization failed to converge. Check mle_retvals'], error=None.
- P3 L1: converged=True, rank=21/21, clusters=872, max |β|=1.8478054177471879, rank-deficient=False, extreme=False, warnings=[], error=None.
- P3 L2: converged=True, rank=33/33, clusters=872, max |β|=2.0248460594413844, rank-deficient=False, extreme=False, warnings=['Maximum Likelihood optimization failed to converge. Check mle_retvals'], error=None.
- P3 L3: converged=True, rank=45/45, clusters=872, max |β|=1.5215183264904824, rank-deficient=False, extreme=False, warnings=['Maximum Likelihood optimization failed to converge. Check mle_retvals'], error=None.
