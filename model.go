package main

import (
	"encoding/csv"
	"encoding/gob"
	"fmt"
	"io"
	"sort"
	"strconv"
	"time"

	"github.com/wlattner/rf/forest"
)

//TODO: consider moving this to rf/forest

type Model struct {
	IsRegression bool
	Clf          *forest.Classifier
	Reg          *forest.Regressor
	VarNames     []string
	fitTime      time.Duration
	opt          modelOptions
	nSample      int
}

func (m *Model) Fit(d *parsedInput, opt modelOptions) {
	start := time.Now()
	if d.isRegression {
		reg := forest.NewRegressor(forest.MaxTrees(opt.nTree), forest.MinSplit(opt.minSplit),
			forest.MinLeaf(opt.minLeaf), forest.MaxFeatures(opt.maxFeatures),
			forest.NumWorkers(opt.nWorkers), forest.ComputeOOB)

		reg.Fit(d.X, d.YReg)
		m.Reg = reg
		m.IsRegression = true
		opt.nTree = m.Reg.NTrees
	} else {
		clf := forest.NewClassifier(forest.MaxTrees(opt.nTree), forest.MinSplit(opt.minSplit),
			forest.MinLeaf(opt.minLeaf), forest.MaxFeatures(opt.maxFeatures), forest.Impurity(opt.impurity),
			forest.NumWorkers(opt.nWorkers), forest.ComputeOOB)

		clf.Fit(d.X, d.YClf)
		m.Clf = clf
		opt.nTree = m.Clf.NTrees
	}
	m.fitTime = time.Since(start)
	m.VarNames = d.VarNames
	m.nSample = len(d.X)
	m.opt = opt
}

func (m *Model) Predict(d *parsedInput) ([]string, error) {
	var pStr []string

	// make sure model and data match
	// No, assume the user knows what they are doing...
	// if d.isRegression != m.IsRegression {
	// 	return pStr, errors.New("model type and datatype don't match")
	// }

	pStr = make([]string, len(d.X))

	if m.IsRegression {
		pNum := m.Reg.Predict(d.X)

		for i, v := range pNum {
			pStr[i] = strconv.FormatFloat(v, 'f', -1, 64)
		}
	} else {
		pID := m.Clf.Predict(d.X)

		for i, id := range pID {
			pStr[i] = m.Clf.Classes[id]
		}
	}

	return pStr, nil
}

func (m *Model) Report(w io.Writer) {
	// generic stuff
	fmt.Fprintf(w, "Fit %d trees using %d examples in %.2f seconds\n",
		m.opt.nTree, m.nSample, m.fitTime.Seconds())
	fmt.Fprintf(w, "\n")

	m.ReportVarImp(w, 20)

	if m.IsRegression {
		m.reportReg(w)
	} else {
		m.reportClf(w)
	}
}

func (m *Model) reportClf(w io.Writer) {
	fmt.Fprintf(w, "Confusion Matrix\n")
	fmt.Fprintf(w, "----------------\n")
	// print confusion matrix
	// headers
	fmt.Fprintf(w, "%-14s ", "")
	for _, class := range m.Clf.Classes {
		fmt.Fprintf(w, "%-14s ", class)
	}
	fmt.Fprintf(w, "\n")

	// rows
	for predictedID, class := range m.Clf.Classes {
		fmt.Fprintf(w, "%-14s ", class)

		for actualID := range m.Clf.Classes {
			fmt.Fprintf(w, "%-14d ", m.Clf.ConfusionMatrix[actualID][predictedID])
		}

		fmt.Fprintf(w, "\n")
	}

	fmt.Fprintf(w, "\n")
	fmt.Fprintf(w, "Overall Accuracy: %.2f%%\n", 100.0*m.Clf.Accuracy)
}

func (m *Model) reportReg(w io.Writer) {
	fmt.Fprintf(w, "\n")
	fmt.Fprintf(w, "Mean Squared Error: %.3f\n", m.Reg.MSE)
	fmt.Fprintf(w, "R-Squared: %.3f%%\n", 100*m.Reg.RSquared)
}

func (m *Model) VarImp() []float64 {
	if m.IsRegression {
		return m.Reg.VarImp()
	} else {
		return m.Clf.VarImp()
	}
}

func (m *Model) SaveVarImp(w io.Writer) error {
	writer := csv.NewWriter(w)

	for i, score := range m.VarImp() {
		err := writer.Write([]string{m.VarNames[i], strconv.FormatFloat(score, 'f', -1, 64)})
		if err != nil {
			return err
		}
	}

	writer.Flush()
	return nil
}

func (m *Model) ReportVarImp(w io.Writer, maxVars int) {
	fmt.Fprintf(w, "Variable Importance\n")
	fmt.Fprintf(w, "-------------------\n")

	varImp := m.VarImp()
	varNames := make([]string, len(m.VarNames))
	copy(varNames, m.VarNames) // don't sort the orig.
	sortByImportance(varImp, varNames)

	// only show top n
	if maxVars > len(varImp) {
		maxVars = len(varImp)
	}

	for i, imp := range varImp[:maxVars] {
		fmt.Fprintf(w, "%-15s: %-10.2f\n", varNames[i], imp)
	}

	fmt.Fprintf(w, "\n")
}

func (m *Model) Load(r io.Reader) error {
	d := gob.NewDecoder(r)
	return d.Decode(m)
}

func (m *Model) Save(w io.Writer) error {
	e := gob.NewEncoder(w)
	return e.Encode(m)
}

type varImpSort struct {
	varName []string
	imp     []float64
}

func (v varImpSort) Len() int {
	return len(v.imp)
}

func (v varImpSort) Less(i, j int) bool {
	return v.imp[i] < v.imp[j]
}

func (v varImpSort) Swap(i, j int) {
	v.imp[i], v.imp[j] = v.imp[j], v.imp[i]
	v.varName[i], v.varName[j] = v.varName[j], v.varName[i]
}

func sortByImportance(imp []float64, names []string) {
	sort.Sort(sort.Reverse(varImpSort{imp: imp, varName: names}))
}
