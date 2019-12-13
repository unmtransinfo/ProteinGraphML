# ProteinGraphML : Visualization

The initial visualization code was developed by Danny Byrd, to visualize the
evidence subgraph representing XGBoost feature importance results.

### Visualize

`MakeVis.py` generates HTML/JS files, with feature importance, for visualization
via web browser.  `ProteinGraphML/Analysis/` contains code for graph and feature
visualization. Visualize has code for creating HTML/JS graphs, and featureLabel has
code for taking a dictionary of feature importance, and giving it human readable labels.

Command-line parameters:

* `--disease` : Disease name.
* `--featurefile` : full path to the pickled features file produced by `TrainModelML.py`, e.g. results/104300/featImportance_XGBCrossVal.pkl.
* `--num` : number of top important features selected.
* `--kgfile` : Pickled KG file, produced by `BuildKG_OlegDb.py` (default: ProteinDisease_GRAPH.pkl).

Example command:

```
MakeVis.py --disease 104300 --featurefile results/104300/featImportance_XGBCrossVal.pkl --num 2
```

## <a name="Notes"/>Notes
