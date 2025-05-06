# Weighted Random Dot Product Graphs

![Embeddings for the footbal matches dataset](figures/football_communities.png)

Accompanying code to the paper "Weighted Random Dot Product Graphs" (B. Marenco, P. Bermolen, M. Fiori F. Larroca, and G. Mateos, soon to be on arXiv)

## Installation

Clone repository and install dependencies using requirements file:

`pip install -r requirements.txt`

## Usage

To generate Figure 7 from the paper, run:

`python wrdpg_discrete_distribution_generation.py`

To generate Figure 8, run:

`python wrdpg_continuous_distribution_generation.py`

To generate Figures 9, 10 and 11, first modify the `football_data_dir` parameter in football_discrete_distribution.py to a valid path on your system, then run it with:

`python football_discrete_distribution.py`

To generate Figure 12, first clone the [repository for the PyMaxEnt package](https://github.com/saadgroup/PyMaxEnt). Then, run

`python maximum_entropy_example.py`
