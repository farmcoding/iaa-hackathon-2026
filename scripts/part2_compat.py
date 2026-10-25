"""Make a developed-market frame look like the emerging schema (owner item O2).

The emerging market has no credit band, annual mileage or telematics, and gives ages in
bands. A developed model trained with those fields applies its "missing" relativity to
every emerging row, and its age curve was learned on exact ages. compat() blanks the
fields emerging lacks and replaces exact ages by the band mean used for emerging rows
(features.AGE_BAND_MEAN), so the base is fitted on the information both markets share.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from iaa import features

MISSING_IN_EMERGING = ["credit_band", "annual_mileage", "telematics_opt_in", "telematics_score"]
AGES = ["driver_age", "policyholder_age", "customer_age"]
_EDGES = [-np.inf, 24.999, 34.999, 49.999, 64.999, np.inf]


def compat(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in MISSING_IN_EMERGING:
        if c in df.columns:
            df[c] = pd.Series(
                pd.NA if isinstance(df[c].dtype, pd.CategoricalDtype) else np.nan, index=df.index
            ).astype(df[c].dtype)
    means = np.array(list(features.AGE_BAND_MEAN.values()))
    for c in AGES:
        if c in df.columns:
            band = pd.cut(df[c].astype(float), _EDGES, labels=False)
            df[c] = np.where(band.notna(), means[band.fillna(0).astype(int)], np.nan)
    return df
