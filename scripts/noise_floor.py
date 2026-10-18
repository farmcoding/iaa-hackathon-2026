"""Run the current experiment with several seeds to measure the noise floor of the
primary metric. The keep rule in the loop must beat the best run by more than this."""

import numpy as np

from iaa import experiment, harness

res = []
for seed in (0, 1, 2):
    experiment.PARAMS["seed"] = seed
    r = harness.validate(f"noise floor seed {seed}", log=False)
    res.append((r["primary"], r["min_fold"]))
p = np.array(res)
print(f"primary mean={p[:, 0].mean():.5f} sd={p[:, 0].std(ddof=1):.5f} range={np.ptp(p[:, 0]):.5f}")
print(f"min_fold sd={p[:, 1].std(ddof=1):.5f}")
