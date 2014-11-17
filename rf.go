package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"runtime"
	"strconv"
	"time"

	"github.com/wlattner/rf/forest"
	"github.com/wlattner/rf/tree"

	"github.com/davecheney/profile"
	flag "github.com/docker/docker/pkg/mflag"
)

var (
	// model/prediction files
	dataFile    = flag.String([]string{"d", "-data"}, "", "example data")
	predictFile = flag.String([]string{"p", "-predictions"}, "", "file to output predictions")
	modelFile   = flag.String([]string{"f", "-final_model"}, "rf.model", "file to output fitted model")
	// model params
	nTree       = flag.Int([]string{"-trees"}, 10, "number of trees")
	minSplit    = flag.Int([]string{"-min_split"}, 2, "minimum number of samples required to split an internal node")
	minLeaf     = flag.Int([]string{"-min_leaf"}, 1, " minimum number of samples in newly created leaves")
	maxFeatures = flag.Int([]string{"-max_features"}, -1, "number of features to consider when looking for the best split, -1 will default to √(# features)")
	impurity    = flag.String([]string{"-impurity"}, "gini", "impurity measure for evaluating splits")
	// runtime params
	nWorkers   = flag.Int([]string{"-workers"}, 1, "number of workers for fitting trees")
	runProfile = flag.Bool([]string{"-profile"}, false, "cpu profile")
)

// lookup table for impurity measure
var impurityCode = map[string]tree.ImpurityMeasure{
	"gini":    tree.Gini,
	"entropy": tree.Entropy,
}

func main() {
	flag.Parse()

	if *dataFile == "" && *predictFile == "" {
		// nothing to do
		fmt.Fprintf(os.Stderr, "Usage of rf:\n\n")
		flag.PrintDefaults()
		os.Exit(1)
	}

	if *nWorkers > 1 {
		runtime.GOMAXPROCS(runtime.NumCPU())
	}

	if *predictFile == "" {
		// is fit
		f, err := os.Open(*dataFile)
		if err != nil {
			fmt.Fprintln(os.Stderr, "error opening", *dataFile, err.Error())
			os.Exit(1)
		}
		defer f.Close()

		X, Y, err := parseCSV(f)
		if err != nil {
			fmt.Fprintln(os.Stderr, "error parsing", err.Error())
			os.Exit(1)
		}

		imp, ok := impurityCode[*impurity]
		if !ok {
			fmt.Fprintln(os.Stderr, "invalid impurity measure", *impurity)
			os.Exit(1)
		}

		clf := forest.NewClassifier(forest.NumTrees(*nTree), forest.MinSplit(*minSplit),
			forest.MinLeaf(*minLeaf), forest.MaxFeatures(*maxFeatures), forest.Impurity(imp),
			forest.NumWorkers(*nWorkers))

		if *runProfile {
			defer profile.Start(profile.CPUProfile).Stop()
		}
		start := time.Now()
		clf.Fit(X, Y)
		d := time.Since(start)
		fmt.Fprintln(os.Stderr, "fitting took %.2fs\n", d.Seconds())

		out, err := os.Create(*modelFile)
		if err != nil {
			fmt.Fprintln(os.Stderr, "error creating", *modelFile, err.Error())
			os.Exit(1)
		}

		err = clf.Save(out)
		if err != nil {
			fmt.Fprintln(os.Stderr, "error writing model to", *modelFile, err.Error())
			os.Exit(1)
		}

		err = out.Close()
		if err != nil {
			fmt.Fprintln(os.Stderr, "error writing model to", *modelFile, err.Error())
			os.Exit(1)
		}

	} else {
		// is predict
		m, err := os.Open(*modelFile)
		if err != nil {
			fmt.Fprintln(os.Stderr, "error opening model", *modelFile, err.Error())
			os.Exit(1)
		}
		defer m.Close()

		clf := forest.NewClassifier()
		clf.Load(m)

		f, err := os.Open(*dataFile)
		if err != nil {
			fmt.Fprintln(os.Stderr, "error opening", *dataFile, err.Error())
			os.Exit(1)
		}
		defer f.Close()

		X, _, err := parseCSV(f)
		if err != nil {
			fmt.Fprintln(os.Stderr, "error parsing", *dataFile, err.Error())
			os.Exit(1)
		}

		pred := clf.Predict(X)

		out, err := os.Create(*predictFile)
		if err != nil {
			fmt.Fprintln(os.Stderr, "error creating", *predictFile, err.Error())
			os.Exit(1)
		}
		defer out.Close()

		err = writePred(out, pred)
		if err != nil {
			fmt.Fprintln(os.Stderr, "error writing", *predictFile, err.Error())
			os.Exit(1)
		}
	}
}

func writePred(w io.Writer, prediction []string) error {
	wtr := bufio.NewWriter(w)

	for _, pred := range prediction {
		_, err := wtr.WriteString(pred)
		if err != nil {
			return err
		}

		err = wtr.WriteByte('\n')
		if err != nil {
			return err
		}
	}

	return wtr.Flush()
}

func parseCSV(r io.Reader) ([][]float64, []string, error) {
	reader := csv.NewReader(r)

	var (
		X [][]float64
		Y []string
	)

	for {
		row, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return X, Y, err
		}

		Y = append(Y, row[0])

		var rowVal []float64
		for _, val := range row[1:] { // data starts in 2nd column
			fv, err := strconv.ParseFloat(val, 64)
			if err != nil {
				return X, Y, err
			}
			rowVal = append(rowVal, fv)
		}
		X = append(X, rowVal)
	}

	return X, Y, nil

}
