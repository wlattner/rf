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
rf --data iris.csv --model iris.model
```

**Args**

`--impurity` the measure to use for evaluating candidate splits, should be `gini` (default) or `entropy`

`--max-depth` the maximum depth to grow the trees, use `-1` (default) for full depth trees

`--max-features` the number of features to consider at each split, `-1` (default) will use sqrt(# features)

`--min-leaf` the minimum number of examples for a leaf node

`--min-split` the minimum number of examples for a node to be split

`--ntree` the number of trees to include in the forest

`--workers` the number of workers to use for fitting trees, should be less than number of cpus/cores

`--data` csv file containing training data

`--model` file to save the fitted model

### Predict
Predictions can be made from a previously fitted model. The data for making predictions should be in a csv file with a format similar to the data used to fit the model, however, the first column should not have labels.

	5.1,3.5,1.4,0.2
	4.9,3,1.4,0.2
	4.7,3.2,1.3,0.2
	4.6,3.1,1.5,0.2
	5,3.6,1.4,0.2
	5.4,3.9,1.7,0.4
	4.6,3.4,1.4,0.3
	5,3.4,1.5,0.2
	4.4,2.9,1.4,0.2

```bash
rf --data iris_new.csv --predictions iris_predictions.csv --model iris.model
```

**Args**

`--data` csv file containing data for making predictions

`--predictions` file for writing predictions

`--model` location of previously fitted model file

