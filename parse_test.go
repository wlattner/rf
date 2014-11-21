package main

import (
	"strings"
	"testing"
)

func TestDetectBostonRegression(t *testing.T) {
	r := strings.NewReader(bostonCSV)

	p, err := parseCSV(r)
	if err != nil {
		t.Error("unexpected error parsing boston data:", err)
		return
	}

	if !p.isRegression {
		t.Error("expected parser to detect regression from boston data")
	}

	if len(p.YClf) > 0 {
		t.Error("expected class labels to be cleared for regression, have non-zero length:", len(p.YClf))
	}

	if p.VarNames[0] != "crim" {
		t.Error("expected first variable name to be crim, got:", p.VarNames[0])
	}

	// check number of rows
	if len(p.X) != 9 {
		t.Error("expected dataset to have 9 rows, got:", len(p.X))
	}

	// num cols
	if len(p.X[0]) != 13 {
		t.Error("expected dataset to have 13 columns, got:", len(p.X[0]))
	}

	// spot check some y vals
	if p.YReg[3] != 33.4 {
		t.Error("expected 4th row to have target value of 33.4, got:", p.YReg[3])
	}
}

func TestDetectIrisClassification(t *testing.T) {
	r := strings.NewReader(irisCSV)

	p, err := parseCSV(r)
	if err != nil {
		t.Error("unexpected error parsing iris data:", err)
		return
	}

	if p.isRegression {
		t.Error("expected parser to detect classification from iris data")
	}

	if len(p.YReg) > 0 {
		t.Error("expected target values to be cleared for classification, have non-zero length:", len(p.YReg))
	}

	if p.VarNames[0] != "Sepal.Length" {
		t.Error("expected first variable name to be Sepal.Length, got:", p.VarNames[0])
	}

	//check num rows
	if len(p.X) != 9 {
		t.Error("expected dataset to have 9 rows, got:", len(p.X))
	}

	// num cols
	if len(p.X[0]) != 4 {
		t.Error("expected dataset to have 4 columns, got:", len(p.X[0]))
	}

	// spot check y val
	if p.YClf[4] != "virginica" {
		t.Error("expected 5th row to have target label of virginica, got:", p.YClf[4])
	}
}

var bostonCSV = `"medv","crim","zn","indus","chas","nox","rm","age","dis","rad","tax","ptratio","black","lstat"
24,0.00632,18,2.31,0,0.538,6.575,65.2,4.09,1,296,15.3,396.9,4.98
21.6,0.02731,0,7.07,0,0.469,6.421,78.9,4.9671,2,242,17.8,396.9,9.14
34.7,0.02729,0,7.07,0,0.469,7.185,61.1,4.9671,2,242,17.8,392.83,4.03
33.4,0.03237,0,2.18,0,0.458,6.998,45.8,6.0622,3,222,18.7,394.63,2.94
36.2,0.06905,0,2.18,0,0.458,7.147,54.2,6.0622,3,222,18.7,396.9,5.33
28.7,0.02985,0,2.18,0,0.458,6.43,58.7,6.0622,3,222,18.7,394.12,5.21
22.9,0.08829,12.5,7.87,0,0.524,6.012,66.6,5.5605,5,311,15.2,395.6,12.43
27.1,0.14455,12.5,7.87,0,0.524,6.172,96.1,5.9505,5,311,15.2,396.9,19.15
16.5,0.21124,12.5,7.87,0,0.524,5.631,100,6.0821,5,311,15.2,386.63,29.93
`

var irisCSV = `"Species","Sepal.Length","Sepal.Width","Petal.Length","Petal.Width"
"setosa",5.1,3.5,1.4,0.2
"setosa",4.9,3,1.4,0.2
"setosa",4.7,3.2,1.3,0.2
"setosa",4.6,3.1,1.5,0.2
"virginica",5,3.6,1.4,0.2
"setosa",5.4,3.9,1.7,0.4
"setosa",4.6,3.4,1.4,0.3
"setosa",5,3.4,1.5,0.2
"setosa",4.4,2.9,1.4,0.2
`
