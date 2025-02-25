import statsmodels.discrete.count_model as st
import pandas as pd
import numpy
import re

INFILE="combined_gene_level_quants_longtab.csv.gz"
#INFILE="test_inps.csv"
OUTPREFIX="polyA_zinb_analysis"

# read the input data
fulltab=pd.read_csv(INFILE)

# generate the design matrix
ALLGT=["wt","dppk"]
ALLCOND=["Nplus","Nminus"]
BASELINE_GT = "wt"
BASELINE_COND="Nplus"
ALLGENES=list(set(fulltab.gene))
ALLGENES.sort()
n_genes=len(ALLGENES)

## make it faster to get an index corresponding to each gene
#GENE_I_DICT = { gn:i for i,gn in enumerate(ALLGENES)}
#I_GENE_DICT = {i:gn for i,gn in enumerate(ALLGENES)}


# fit the zinb model for all of our data
# we have to do this one gene at a time because otherwise the size of the fit is nuts

def fit_zinb_for_gene( genedata ):

    # set up the design matrix
    # we will have an intercept plus three parameters, corresponding to the baseline, genotype effect, condition effect, and interaction

    n_data = len(genedata)
    n_design = 4
    design_mat = numpy.zeros( ( n_data, n_design ) )
    print(design_mat.shape)

    i_row = 0
    for c,gt in zip(genedata.condition,genedata.genotype):
        # intercept term
        design_mat[i_row,0] = 1 

        # genotype term
        if gt != BASELINE_GT:
            design_mat[i_row,1] = 1

        # condition term
        if c != BASELINE_COND:
            design_mat[i_row,2] = 1

        # interaction term
        if (gt != BASELINE_GT) and (c != BASELINE_COND):
            design_mat[i_row,3] = 1

        i_row += 1


    myfit = st.ZeroInflatedNegativeBinomialP( endog = genedata.Alength, exog=design_mat)
    fitted_mod = myfit.fit(maxiter=500)

    if not(extract_converged(fitted_mod.summary())):
        raise(ValueError("Model did not converge"))

    return fitted_mod

# stupid hack to get convergence messages
def extract_converged(parameter_string):
    converged_pattern = r'converged:\s+(\w+)'
    match = re.search(converged_pattern, str(parameter_string))
    if match:
        return match.group(1) == 'True'
    else:
        raise ValueError("Converged parameter not found.")

def print_output_for_gene( gn,fitobj,ostr ):
    # write a line of my output tale with the fitted effects and p values

    fits=fitobj.params
    pvals=fitobj.pvalues


    ostr.write(f"{gn},{fits['const']},{pvals['const']},{fits['x1']},{pvals['x1']},{fits['x2']},{pvals['x2']},{fits['x3']},{pvals['x3']}\n")


with open("gene_level_fits.csv","w") as gene_outs:
    gene_outs.write("gene_name,fit_gene_eff,pval_gene_eff,fit_dppk_eff,pval_dppk_eff,fit_Nminus_eff,pval_Nminus_eff,fit_dppk:Nminus_ff,pval_dppk:Nminus_eff\n")
    for gene_name in ALLGENES:
        print(gene_name)
        try:
            gene_dat = fulltab[fulltab.gene == gene_name]
            this_fit = fit_zinb_for_gene(gene_dat)
            print_output_for_gene( gene_name,this_fit, gene_outs )

        except:
            print(f"Skipping gene {gene_name}")


