package main

import (
	"bufio"
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"os"
	"runtime"
	"strconv"
	"time"

	"github.com/wlattner/rf/forest"
	"github.com/wlattner/rf/tree"

	"github.com/davecheney/profile"
)

var (
	nTree       = flag.Int("ntree", 10, "number of trees")
	minSplit    = flag.Int("min-split", 2, "min examples for node to be split")
	minLeaf     = flag.Int("min-leaf", 1, "min examples for leaf node")
	maxDepth    = flag.Int("max-depth", -1, "max depth to grow trees")
	maxFeatures = flag.Int("max-features", -1, "max number of features to consider at each split, -1 will default to sqrt(# features)")
	impurity    = flag.String("impurity-measure", "gini", "impurity measure for evaluating splits")
	dataFile    = flag.String("data", "", "csv file with training data")
	predictFile = flag.String("predictions", "", "output file for predictions")
	modelFile   = flag.String("model", "rf.model", "file to write/read model")
	nWorkers    = flag.Int("workers", 1, "number of workers for fitting trees")
	runProfile  = flag.Bool("profile", false, "cpu profile")
)

var impurityCode = map[string]tree.ImpurityMeasure{"gini": tree.Gini, "entropy": tree.Entropy}

func main() {
	flag.Parse()

	if *nWorkers > 1 {
		runtime.GOMAXPROCS(runtime.NumCPU())
	}

	if *predictFile == "" {
		// is fit
		f, err := os.Open(*dataFile)
		if err != nil {
			fmt.Println("error opening", *dataFile, err.Error())
			os.Exit(1)
		}
		defer f.Close()

		X, Y, err := parseCSV(f, true)
		if err != nil {
			fmt.Println("error parsing", err.Error())
			os.Exit(1)
		}

		imp, ok := impurityCode[*impurity]
		if !ok {
			fmt.Println("invalid impurity measure", *impurity)
			os.Exit(1)
		}

		clf := forest.NewClassifier(forest.NumTrees(*nTree), forest.MinSplit(*minSplit),
			forest.MinLeaf(*minLeaf), forest.MaxDepth(*maxDepth),
			forest.MaxFeatures(*maxFeatures), forest.Impurity(imp),
			forest.NumWorkers(*nWorkers))

		if *runProfile {
			defer profile.Start(profile.CPUProfile).Stop()
		}
		start := time.Now()
		clf.Fit(X, Y)
		d := time.Since(start)
		fmt.Printf("fitting took %.2fs\n", d.Seconds())

		out, err := os.Create(*modelFile)
		if err != nil {
			fmt.Println("error creating", *modelFile, err.Error())
			os.Exit(1)
		}

		err = clf.Save(out)
		if err != nil {
			fmt.Println("error writing model to", *modelFile, err.Error())
			os.Exit(1)
		}

		err = out.Close()
		if err != nil {
			fmt.Println("error writing model to", *modelFile, err.Error())
			os.Exit(1)
		}

	} else {
		// is predict
		m, err := os.Open(*modelFile)
		if err != nil {
			fmt.Println("error opening model", *modelFile, err.Error())
			os.Exit(1)
		}
		defer m.Close()

		clf := forest.NewClassifier()
		clf.Load(m)

		f, err := os.Open(*dataFile)
		if err != nil {
			fmt.Println("error opening", *dataFile, err.Error())
			os.Exit(1)
		}
		defer f.Close()

		X, _, err := parseCSV(f, false)
		if err != nil {
			fmt.Println("error parsing", *dataFile, err.Error())
			os.Exit(1)
		}

		pred := clf.Predict(X)

		out, err := os.Create(*predictFile)
		if err != nil {
			fmt.Println("error creating", *predictFile, err.Error())
			os.Exit(1)
		}
		defer out.Close()

		err = writePred(out, pred)
		if err != nil {
			fmt.Println("error writing", *predictFile, err.Error())
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

func parseCSV(r io.Reader, hasLabels bool) ([][]float64, []string, error) {
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

		var (
			col    int
			rowVal []float64
		)

		if hasLabels {
			Y = append(Y, row[0])
			col++
		}

		for _, val := range row[col:] {
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
