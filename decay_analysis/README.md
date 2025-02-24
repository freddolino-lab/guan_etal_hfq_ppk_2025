This directory contains python code necessary for replicating the RNA half life analysis (from RIF chase experiments) described in the accompanying paper.
Key files include:

`analyze_decay_ercc_normed.py` -- this python program actually implements the half-life analysis, with final values written to the file `halflives_full_with_startcfu_bounded.csv`
`input` -- this directory contains a set of files that are presumed to exist in the working directory from which the main code is run. Note that there is also a conda environment definition file contained here that can be used to install prerequisites for running the analysis
`example_output` -- this directory contains outputs obtained when running the main driver script noted above in our environment
