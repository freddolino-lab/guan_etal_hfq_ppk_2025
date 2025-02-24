# these commands should be used in the conda environment defined by decay_conda.txt
import pandas
import numpy
import seaborn
import matplotlib.pyplot as plt
import scipy.optimize

# data matrix of the count/abundance information
datmat=pandas.read_csv("full_datamat.csv")



# information on the cfu counts
cfudat=pandas.read_csv("hfq_rif_startcfu_counts.csv")
cfudat.set_index("sample", inplace=True)
cfudict = cfudat.to_dict()


# first get a filtered set of ercc transcripts that are easily detectable
ercc_dat = datmat.loc[((x[0:4]=="ERCC") for x in datmat["Unnamed: 0"])]
ercc_means = ercc_dat.mean(numeric_only=True, axis=1)
ercc_flags=ercc_means>0.0005
ercc_subset = ercc_dat[ercc_flags]

# now get the relative amount of each measureable ercc transcript vs the mean, and average across the transcripts
ercc_mat = ercc_subset.drop("Unnamed: 0",axis=1).to_numpy()
ercc_norm_mat = ercc_mat / numpy.reshape(numpy.mean(ercc_mat,axis=1),(-1,1))
#print(ercc_norm_mat)
#print(ercc_norm_mat.shape)
#print("====")
ercc_normfracs = numpy.median(ercc_norm_mat, axis=0)
#print(ercc_normfracs)
#print(ercc_normfracs.shape)

#for x,y in zip(datmat.columns[1:], ercc_normfracs):
#    print(f"{x} : {y}")

#print("----")



# additionally, we normalize based on the cfu data that we have available
full_mat = datmat.drop("Unnamed: 0", axis=1).to_numpy()
cfu_norm_mat = numpy.ones_like(full_mat)

for i,name in enumerate(datmat.columns[1:]):
    cfu_norm_mat[:,i] *= cfudict['cfu'][name]

cfu_norm_mat /= numpy.mean(cfu_norm_mat)

# and now apply the normalizations to the entire matrix
full_mat_normed = full_mat / ercc_normfracs
full_mat_normed /= cfu_norm_mat

# apply a special correction to one column that did not get the same amount of spikein as the other samples
bad_col = list(datmat.columns[1:]).index("ppk_delta_t60")
full_mat_normed[:,bad_col] /= (5.0/1.6)

# and convert it back to a pandas array so we can continue processing
sample_names = datmat.columns[1:]
gene_names = datmat["Unnamed: 0"]

normed_table = pandas.DataFrame( full_mat_normed )
normed_table.rename( columns = { x : sample_names[x] for x in range(len(sample_names)) }, inplace=True)
normed_table.set_index(gene_names.rename("bnum"), inplace=True)

# save this as a data matrix
normed_table.to_csv("fully_normed_mat_with_startcfu.csv")

# also make a long form table that is more useful for plotting
normed_table_melted = normed_table.melt( var_name="sample", value_name = "normed_expression", ignore_index=False)
normed_table_melted.reset_index(inplace=True)
normed_table_melted["genotype"] = [x.split("_t")[0] for x in normed_table_melted["sample"]]
normed_table_melted["condition"] = [x.split(".")[0] for x in normed_table_melted["sample"]]
normed_table_melted["riff"] = [len(x.split("_riff")) > 1 for x in normed_table_melted["sample"]]
tmpnames = [x.split("_riff")[0] for x in normed_table_melted["condition"]]
normed_table_melted["timepoint"] = [x.split("_")[-1] for x in tmpnames] 
gr = [str(x) + str(y) for x,y in zip( normed_table_melted["genotype"], normed_table_melted["riff"])]
unique_times = pandas.Categorical( gr, categories=set(gr))
normed_table_melted["timevals"] = [float(x[1:]) for x in normed_table_melted["timepoint"]]
normed_table_melted["plot_times"] = normed_table_melted["timevals"] + 2*( unique_times.codes - len(set(gr))/2.0)
normed_table_melted["runtype"] = gr


def plot_gene( datmat, genename, outfile ):
    # plot the time course of a single gene to a file
    # datmat should be a long form table as generated above
    plt.figure()
    this_gene_data = datmat[datmat.bnum == genename]
    seaborn.relplot(data=this_gene_data, x="plot_times", y="normed_expression", style="riff", hue="genotype")
    plt.xticks(rotation=45)
    plt.savefig(outfile)
    plt.close()

# plot some genes of interest -- the same paradigm can be applied to plot other genes if you wish
plot_gene(normed_table_melted, "b1286", "rnaseE_with_startcfu.pdf")
plot_gene(normed_table_melted, "b4172", "hfq_with_startcfu.pdf")
plot_gene(normed_table_melted, "b2501", "ppk_with_startcfu.pdf")
plot_gene(normed_table_melted, "ERCC-00130", "spikein_with_startcfu.pdf")
plot_gene(normed_table_melted, "b1716", "rplT_with_startcfu.pdf")
plot_gene(normed_table_melted, "b2779", "eno_with_startcfu.pdf")

# now calculate decay rates with vs without rif

## here are some helper functions that we use for the decay rate fits

def exp_fit(x,a,b):
    y=a*numpy.exp(-1*x/b) 
    return y

def fit_for_gene( main_df, target_conds, gene_name ):
    # fit an exponential decay curve for the specified gene and return thefits
    # the tuple returned has the baseline abundance and half life
    target_decay_data = pandas.DataFrame(main_df[ [x in target_conds for x in main_df["condition"]] ])
    gene_dat = target_decay_data[ target_decay_data["bnum"] == gene_name ]
    meandat=gene_dat.groupby('timevals').agg({'normed_expression' : 'mean'})
    meandat.reset_index(inplace=True)
    try:
        fit = scipy.optimize.curve_fit(exp_fit, gene_dat["timevals"], gene_dat["normed_expression"], p0=[0.001, 20], bounds=([0.0,0.0],[numpy.inf, numpy.inf]), method='dogbox')
    except:
        fit = [ [numpy.nan, numpy.nan] ]
    return fit

# here is how we could plt these fits if we wanted
#plt.figure()
#plt.plot( meandat["timevals"], meandat["normed_expression"], 'bo')
#plt.plot( gene_dat["timevals"], gene_dat["normed_expression"], 'bo', alpha=0.4, markersize=5 )
#plt.plot( numpy.arange(60), exp_fit(numpy.arange(60), fit[0][0], fit[0][1], fit[0][2]), 'r--')
#plt.savefig("look.png")

# now we march through the conditions and collect all of the fits
allgenes=list(set(normed_table_melted["bnum"]))
genes=[]
conditions=[]
abunds = []
halflives=[]

## here is how we fit all of the genes for the wt +rif case
target_conds_wt_rif = set( ["wt_t00", "wt_t10_riff", "wt_t30_riff", "wt_t60_riff"] )

for gene in allgenes:
    print(f"wtrif : {gene}")
    this_fit = fit_for_gene( normed_table_melted, target_conds_wt_rif, gene )
    genes.append(gene)
    conditions.append("wt_plusrif")
    abunds.append(this_fit[0][0])
    halflives.append(this_fit[0][1])

## and we do the same thing for the other conditions
target_conds_wt_norif = set( ["wt_t00", "wt_t10", "wt_t30", "wt_t60"] )

for gene in allgenes:
    print(f"wtnorif : {gene}")
    this_fit = fit_for_gene( normed_table_melted, target_conds_wt_norif, gene )
    genes.append(gene)
    conditions.append("wt_minusrif")
    abunds.append(this_fit[0][0])
    halflives.append(this_fit[0][1])

target_conds_dppk_norif = set( ["ppk_delta_t00", "ppk_delta_t10", "ppk_delta_t30", "ppk_delta_t60"] )

for gene in allgenes:
    print(f"ppknorif : {gene}")
    this_fit = fit_for_gene( normed_table_melted, target_conds_dppk_norif, gene )
    genes.append(gene)
    conditions.append("dppk_minusrif")
    abunds.append(this_fit[0][0])
    halflives.append(this_fit[0][1])

target_conds_dppk_rif = set( ["ppk_delta_t00", "ppk_delta_t10_riff", "ppk_delta_t30_riff", "ppk_delta_t60_riff"] )

for gene in allgenes:
    print(f"ppkrif : {gene}")
    this_fit = fit_for_gene( normed_table_melted, target_conds_dppk_rif, gene )
    genes.append(gene)
    conditions.append("dppk_plusrif")
    abunds.append(this_fit[0][0])
    halflives.append(this_fit[0][1])

full_halflife_tab = pandas.DataFrame({'bnum' : genes, 'condition' : conditions, 'base_abundance' : abunds, 'halflife_minutes' : halflives})

# also add gene names alongside b numbers
bnum_to_gn_dict = {}
with open("bnum_gn_conv.txt") as instr:
    for line in instr:
        linearr = line.rstrip().split() 
        print(linearr)
        bnum_to_gn_dict[linearr[1]] = linearr[0]

# small helper function for dictionary lookups with a default
def lookup_in_dict_default(dictionary,thiskey):
    try:
        return dictionary[thiskey]
    except KeyError:
        return "unknown"

gns = [lookup_in_dict_default(bnum_to_gn_dict,x) for x in full_halflife_tab["bnum"]]
full_halflife_tab["gene_name"] = gns

full_halflife_tab.to_csv("halflives_full_with_startcfu_bounded.csv")


