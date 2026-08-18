import numpy as np
import pandas as pd
from scipy.stats import norm

def windowed_hr(rs, windows=((0, 36), (36, 60), (60, 120), (120, 10 ** 6))):
    from lifelines import CoxPHFitter
    rows = []
    for lo, hi in windows:
        sub = rs[(rs.time_months > lo)].copy()
        sub["time_w"] = sub["time_months"].clip(upper=hi) - lo
        sub["event_w"] = ((sub["time_months"] <= hi) & (sub["event"] == 1)).astype(int)
        sub = sub[sub.time_w > 0]
        if sub.event_w.sum() < 5:
            rows.append(dict(window_lo=lo, window_hi=hi, hr=np.nan, hr_lo=np.nan, hr_hi=np.nan, p=np.nan, n=len(sub), events=int(sub.event_w.sum())))
            continue
        try:
            cph = CoxPHFitter()
            cph.fit(sub[["time_w", "event_w", "risk_score_z", "cohort"]].assign(
                **{c: (sub["cohort"] == c).astype(float) for c in sub["cohort"].unique() if c != sub["cohort"].iloc[0]}
            ).drop(columns=["cohort"]), duration_col="time_w", event_col="event_w")
            hr = float(np.exp(cph.params_["risk_score_z"]))
            se = float(cph.standard_errors_["risk_score_z"])
            hr_lo = float(np.exp(cph.params_["risk_score_z"] - 1.96 * se))
            hr_hi = float(np.exp(cph.params_["risk_score_z"] + 1.96 * se))
            p = float(2 * norm.sf(abs(cph.params_["risk_score_z"] / se)))
        except Exception:
            hr = hr_lo = hr_hi = p = np.nan
        rows.append(dict(window_lo=lo, window_hi=hi, hr=hr, hr_lo=hr_lo, hr_hi=hr_hi, p=p, n=len(sub), events=int(sub.event_w.sum())))
    return pd.DataFrame(rows)

rs = pd.read_csv('loco_risk_scores.csv')

tv = windowed_hr(rs)
tv.to_csv("timevarying_hr_windows.csv", index=False)
print(tv.to_string(index=False))