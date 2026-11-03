# sim/

Synthetic-data generators and simulation scripts (ground-truth recovery checks,
toy-world experiments). Each script is a plain `.py` file run as
`python sim/<name>.py --seed <int>`; it must write its CSV/JSON and figure to
`results/` via `structure_of_truth.io`. No notebooks.
