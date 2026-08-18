# This is hard-coded

import pandas as pd
import numpy as np
from io import StringIO
import os

rows = """cohort,platform,endpoint,n,events,event_rate,median_fu_months,max_fu_months,n_genes,time_unit,join,overlap
TCGA,RNA-seq (RSEM),OS,1086,154,0.1418,25.2,282.7,16829,months,expr[identity]xclin[identity],1100
METABRIC,Microarray (Illumina),OS,1979,1143,0.5776,157.9,355.2,17933,months,expr[identity]xclin[identity],1980
SCANB_GSE96058,RNA-seq (HiSeq/NextSeq),OS,3069,322,0.1049,54.9,81.3,30865,days,expr[identity]xclin[identity],3069
SCANB_GSE202203,RNA-seq (TPM),OS,2912,426,0.1463,78.2,104.9,19580,days,expr[identity]xclin[identity],2913
GSE6532,Affymetrix U133A/B,DMFS,380,96,0.2526,104.3,202.1,13039,days(inferred),expr[identity]xclin[identity],414
GSE11121,Affymetrix U133A,DMFS,200,46,0.23,95.5,240.0,13039,months(inferred),expr[identity]xclin[identity],200
GSE21653,Affymetrix U133 Plus 2.0,DFS,248,79,0.3185,66.2,222.3,21355,months,expr[identity]xclin[identity],266
GSE20711,Affymetrix U133 Plus 2.0,OS,88,25,0.2841,89.3,169.0,21355,years(suffix),expr[identity]xclin[identity],90
GSE58812,Affymetrix U133 Plus 2.0,OS,107,29,0.271,83.0,169.2,21355,days,expr[identity]xclin[identity],107"""

qc = pd.read_csv(StringIO(rows))
qc["role"] = np.where(qc.endpoint.eq("OS") & qc.n.gt(500), "Primary (OS)",
              np.where(qc.endpoint.eq("OS"), "Primary (OS, small)", "Secondary (DMFS/DFS)"))
os.makedirs("setup", exist_ok=True)
qc.to_csv(os.path.join("setup", "table1_cohorts.csv"), index=False)