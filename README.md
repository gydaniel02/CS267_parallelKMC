# MultiTimescaleKMC
Performs KMC simulations to obtain DELTA structures from a DRX starting point using smol. The distinct timescales of Mn and Li migration are worked around by implementing canonical montecarlo between Mn hops.

# Environment

`conda env create -f environment.yml -n KMCenv`

`conda activate KMCenv`

In addition, to run the CMC, you need to install smol from the Github repository

`git clone https://github.com/CederGroupHub/smol.git`

`cd smol`

`pip install .`
