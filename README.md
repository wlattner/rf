rf
==

This is a Go implementation of the random forest algorithm for classification and regression, as described in Louppe, G. (2014) ["Understanding Random Forests: From Theory to Practice"](http://arxiv.org/abs/1407.7502) (PhD thesis). Both the random forest and the decision tree are usable as standalone Go packages. The cli can fit a model from a csv file and make predictions from a previously fitted model. The csv parser is rather limited, only numeric values are accepted.

[![GoDoc](https://godoc.org/github.com/wlattner/rf?status.svg)](http://godoc.org/github.com/wlattner/rf)

Install
-------
```bash
go get github.com/wlattner/rf
```

Usage
-----
### Fit
A model can be fitted from a csv file, the label or target value should be the first column and the remaining columns should be numeric features. The file may contain a header row. If a header row is present, the column names will be used for variable names in the variable importance report (see below). For example, the iris data would appear as:
	
	"Species","Sepal.Length","Sepal.Width","Petal.Length","Petal.Width"
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

`--var_importance arg` file to output variable importance estimates

`--trees arg (=10)` number of trees to include in forest

`--stop_early` stop fitting trees if OOB error estimate converges

`--min_split arg (=2)` minimum number of samples required to split an internal node

`--min_leaf arg (=1)` minimum number of samples in newly created leaves

`--max_features arg (=-1)`  number of features to consider when looking for the best split, -1 will default to √(# features)

`--impurity arg (=gini)` the measure to use for evaluating candidate splits, must be `gini` or `entropy`

`--workers arg (=1)` number of workers for fitting trees

`-c, --classification` force parser to use integer/numeric labels for classification

Regression is also supported, the csv parser will detect if the first column is numeric or categorical. If the class labels look like numbers:

	"Species",Sepal.Length","Sepal.Width","Petal.Length","Petal.Width"
	"1",5.1,3.5,1.4,0.2
	"1",4.9,3,1.4,0.2
	...
	"2",5.9,3,5.1,1.8

The parser will get confused and fit a regression model, if this happens, try running with the `--classification` flag.


**Output**

After the input data is parsed and the forest fitted, rf will write a diagnostic report to stderr.
```bash
Fit 10 trees using 150 examples in 0.00 seconds

Variable Importance
-------------------
Petal.Length   : 0.55
Petal.Width    : 0.40
Sepal.Width    : 0.03
Sepal.Length   : 0.02

Confusion Matrix
----------------
               setosa         versicolor     virginica
setosa         50             1              0
versicolor     0              46             5
virginica      0              3              45

Overall Accuracy: 94.00%
```
The confusion matrix and overall accuracy are estimated from out of bag samples for each tree in the forest. The report will show up to 20 variables in the variable importance section in decreasing order of importance. If your data have more predictors, the importance estimates for all variables can be written to a csv file using the `--var_importance` flag.

For a regression model:
```bash
Fit 10 trees using 506 examples in 0.01 seconds

Variable Importance
-------------------
rm             : 0.38
lstat          : 0.30
nox            : 0.12
crim           : 0.05
dis            : 0.03
ptratio        : 0.03
tax            : 0.02
age            : 0.02
black          : 0.02
rad            : 0.01
indus          : 0.01
zn             : 0.01
chas           : 0.00


Mean Squared Error: 15.677
R-Squared: 81.487%
```
The mean squared error is computed from out of bag samples for each tree in the forest. The variable importance is reported in the same manner as classification.

### Predict
Predictions can be made from a previously fitted model. The data for making predictions should be in a csv file with a format similar to the data used to fit the model, however, the first column will be ignored.
	
	"Species","Sepal.Length","Sepal.Width","Petal.Length","Petal.Width"
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