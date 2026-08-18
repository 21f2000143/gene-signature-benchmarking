import pandas as pd

t1 = pd.read_csv("table1_cohorts.csv")
events = t1.set_index('cohort')['events'].to_dict()
OS6 = ["TCGA","METABRIC","SCANB_GSE96058","SCANB_GSE202203","GSE20711","GSE58812"]

m = pd.read_csv("metrics_harrell_uno.csv")
rows = []
for gs, sub in m.groupby('gene_set'):
    sub = sub.set_index('held_out_cohort')
    w = pd.Series({c: events[c] for c in OS6})
    harrell_ev = (sub.loc[OS6,'harrell_c'] * w).sum() / w.sum()
    uno_ev = (sub.loc[OS6,'uno_c'] * w).sum() / w.sum()
    rows.append({'gene_set': gs, 'harrell_ev': round(harrell_ev,3), 'uno_ev': round(uno_ev,3)})
out = pd.DataFrame(rows).sort_values('uno_ev', ascending=False).reset_index(drop=True)
out.to_csv("uno_event_weighted.csv", index=False)