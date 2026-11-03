import json

import matplotlib.pyplot as plt

from structure_of_truth.io import save_figure, write_csv, write_json


def test_writers_record_seed(tmp_path):
    j = write_json("demo", {"x": 1}, seed=3, root=tmp_path)
    c = write_csv("demo", [{"a": 1}], seed=3, root=tmp_path)
    fig, ax = plt.subplots()
    ax.plot([0, 1])
    f = save_figure("demo", fig, root=tmp_path)
    assert json.loads(j.read_text())["seed"] == 3
    assert "seed" in c.read_text().splitlines()[0]
    assert f.exists()
