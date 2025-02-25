import subprocess
import os.path

# for each of our samples, go through the reads and count how many Ts come at the beginning of read 1 in each case -- note that this gives us the poly-A tail length since our read is antisense to the original RNA

ALDIR="aligned"
OUTDIR="quant"
SRCDIR="src"

sample_names = ["polyA_HM_delta_Nplus_r1_S11", "polyA_HM_delta_Nplus_r2_S15", "polyA_HM_delta_Nplus_r3_S19", "polyA_HM_delta_Nplus_r4_S23", "polyA_ppk1_delta_Nminus24_r1_S13", "polyA_ppk1_delta_Nminus24_r2_S17", "polyA_ppk1_delta_Nminus24_r3_S21", "polyA_ppk1_delta_Nminus24_r4_S25", "polyA_WT_Nminus24_r1_S12", "polyA_WT_Nminus24_r2_S16", "polyA_WT_Nminus24_r3_S20", "polyA_WT_Nminus24_r4_S24", "polyA_wt_Nplus_r1_S10", "polyA_wt_Nplus_r2_S14", "polyA_wt_Nplus_r3_S18", "polyA_wt_Nplus_r4_S22"]

sample_names_no_S = [x[:-3] for x in sample_names]


for s in sample_names_no_S:
    r1_readname = os.path.join(ALDIR, s + "_fwd_trimmed_paired.fq.gz")
    al_bamname = os.path.join(ALDIR, s + "_aligned_bowtie2_sorted.bam")
    out_fname = os.path.join(OUTDIR, s + "_A_lengths.bed")
    cmdline=f"python {os.path.join(SRCDIR,'get_A_counts.py')} {r1_readname} {al_bamname} {out_fname}"
    subprocess.call(cmdline, shell=True)
