rf
==

This is a Go implementation of a random forest classifier, as described in Louppe, G. (2014) ["Understanding Random Forests: From Theory to Practice"](http://arxiv.org/abs/1407.7502) (PhD thesis). Both the random forest and the decision tree are usable as standalone Go packages. The commandline app can fit a model from a csv file and make predictions from a previously fitted model and a csv file with new examples. The csv parser is rather limited, only numeric values are accepted.

[![GoDoc](https://godoc.org/github.com/wlattner/rf?status.svg)](http://godoc.org/github.com/wlattner/rf)

Install
-------
```bash
go get github.com/wlattner/rf
```

Usage
-----
### Fit
A model can be fitted from a csv file, the label should be the first column and the remaining columns should be numeric features. The file should have no header. For example, the iris data would appear as:

	"setosa",5.1,3.5,1.4,0.2
	"setosa",4.9,3,1.4,0.2
	"setosa",4.7,3.2,1.3,0.2
	"setosa",4.6,3.1,1.5,0.2
	"setosa",5,3.6,1.4,0.2
	"setosa",5.4,3.9,1.7,0.4
	"setosa",4.6,3.4,1.4,0.3
	"setosa",5,3.4,1.5,0.2
	"setosa",4.4,2.9,1.4,0.2
	...

Assuming these data are in a file named `iris.csv`, a model would be fitted with the following command:

```bash
rf -d iris.csv -f iris.model
```

**Args**

`-d, --data arg` example data

`-f --final_model arg (=rf.model)` file to output fitted model

`--trees arg (=10)` number of trees to include in forest

`--min_split arg (=2)` minimum number of samples required to split an internal node

`--min_leaf arg (=1)` minimum number of samples in newly created leaves

`--max_features arg (=-1)`  number of features to consider when looking for the best split, -1 will default to √(# features)

`--impurity arg (=gini)` the measure to use for evaluating candidate splits, must be `gini` or `entropy`

`--workers arg (=1)` number of workers for fitting trees


### Predict
Predictions can be made from a previously fitted model. The data for making predictions should be in a csv file with a format similar to the data used to fit the model, however, the first column will be ignored.

	"",5.1,3.5,1.4,0.2
	"",4.9,3,1.4,0.2
	"",4.7,3.2,1.3,0.2
	"",4.6,3.1,1.5,0.2
	"",5,3.6,1.4,0.2
	"",5.4,3.9,1.7,0.4
	"",4.6,3.4,1.4,0.3
	"",5,3.4,1.5,0.2
	"",4.4,2.9,1.4,0.2
	...

```bash
rf -d iris.csv -p iris_predictions.csv -f iris.model
```

**Args**

`-d, --data arg` example data

`-p, --predictions arg` file to output predictions

`-f, --final_model arg (=rf.model)` file with previously fitted model

Docs
----
Documentation for the two packages, forest and tree can be found on godoc. `tree` implements classification trees while `forest` implements random forests using `tree`. See `rf.go` in this repository for an example of using the `forest` package.

**forest:** http://godoc.org/github.com/wlattner/rf/forest

**tree:** http://godoc.org/github.com/wlattner/rf/tree