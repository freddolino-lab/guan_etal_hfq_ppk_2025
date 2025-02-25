# this script will read the bed files that I have generated with all of the A lengths for each gene,
# and make a long form table containing all of this information
import sys
import subprocess
import gzip

# names of input conditions
INPUT_NAMES=["polyA_HM_delta_Nplus","polyA_ppk1_delta_Nminus24","polyA_WT_Nminus24", "polyA_wt_Nplus"]
REP_NAMES=["r1","r2","r3","r4"]
SUFFIX_IN = "gene_A_lengths.bed"

# names of output conditions
GENOTYPE_NAMES=["dppk","dppk","wt","wt"]
COND_NAMES=["Nplus","Nminus","Nminus","Nplus"]

# helper function that writes the results for a single rep to my output file
def add_f_to_table( inputfile,genotype,condition,repid, outstr):

    with open(inputfile) as instr:
        for line in instr:
            linearr=line.rstrip().split("\t")
            length_distr = linearr[6].split(",")
            gene_name=linearr[3]

            for l in length_distr:
                if l == ".":
                    continue

                outstr.write(f"{genotype},{condition},{repid},{gene_name},{str(l)}\n")



with gzip.open("combined_gene_level_quants_longtab.csv.gz",mode="wt") as ostr:

    ostr.write("genotype,condition,repid,gene,Alength\n")


    for i,g,c in zip(INPUT_NAMES,GENOTYPE_NAMES,COND_NAMES):
        for r in REP_NAMES:
            in_fname=f"{i}_{r}_{SUFFIX_IN}"

            add_f_to_table( in_fname, g,c,r, ostr )


