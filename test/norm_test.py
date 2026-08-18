import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harmonise import zscore

cbio_df = pd.read_table('/mnt/kedargouri/sachin/projects/oncogenic-signaling-pathways/dataset/tcga_cbio_hiseq/corrected_data_mrna_seq_v2_rsem_zscores_ref_diploid_samples.tsv', index_col=0)
gdc_df = pd.read_table('/mnt/kedargouri/sachin/projects/oncogenic-signaling-pathways/dataset/tcga_gdc_hiseq/tumor_rna_seq/corrected_protein_coding_TPM_matrix.tsv', index_col=0)

print("Before zscore")
print(cbio_df.head())
print(gdc_df.head())

after_zscore_cbio = zscore(gdc_df)

after_zscore_gdc = zscore(cbio_df)
print("After zscore: cbio -> gdc")
print(after_zscore_cbio.head())
print(after_zscore_gdc.head())
