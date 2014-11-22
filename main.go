package main

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"runtime"

	"github.com/davecheney/profile"

	flag "github.com/docker/docker/pkg/mflag"
)

var (
	// model/prediction files
	dataFile    = flag.String([]string{"d", "-data"}, "", "example data")
	predictFile = flag.String([]string{"p", "-predictions"}, "", "file to output predictions")
	modelFile   = flag.String([]string{"f", "-final_model"}, "rf.model", "file to output fitted model")
	impFile     = flag.String([]string{"-var_importance"}, "", "file to output variable importance estimates")
	// model params
	nTree       = flag.Int([]string{"-trees"}, 10, "number of trees")
	minSplit    = flag.Int([]string{"-min_split"}, 2, "minimum number of samples required to split an internal node")
	minLeaf     = flag.Int([]string{"-min_leaf"}, 1, "minimum number of samples in newly created leaves")
	maxFeatures = flag.Int([]string{"-max_features"}, -1, "number of features to consider when looking for the best split, -1 will default to √(# features) or # features / 3 for regression")
	// force classification
	forceClf = flag.Bool([]string{"c", "-classification"}, false, "force parser to use integer targets/labels for classification")
	// runtime params
	nWorkers   = flag.Int([]string{"-workers"}, 1, "number of workers for fitting trees")
	runProfile = flag.Bool([]string{"-profile"}, false, "cpu profile")
)

type modelOptions struct {
	nTree       int
	minSplit    int
	minLeaf     int
	maxFeatures int
	nWorkers    int
}

func parseModelOpts() (modelOptions, error) {
	o := modelOptions{
		nTree:       *nTree,
		minSplit:    *minSplit,
		minLeaf:     *minLeaf,
		maxFeatures: *maxFeatures,
		nWorkers:    *nWorkers,
	}

	return o, nil
}

func main() {
	flag.Parse()

	if *nWorkers > 1 {
		runtime.GOMAXPROCS(runtime.NumCPU())
	}

	if *runProfile {
		defer profile.Start(profile.CPUProfile).Stop()
	}

	// make sure user specified csv file w/ data
	if *dataFile == "" {
		fmt.Fprintf(os.Stderr, "Usage of rf:\n\n")
		flag.PrintDefaults()
		os.Exit(1)
	}

	f, err := os.Open(*dataFile)
	if err != nil {
		fatal("error opening data file", err.Error())
	}
	defer f.Close()

	d, err := parseCSV(f, *forceClf)
	if err != nil {
		fatal("error parsing input data", err.Error())
	}

	// consider non-blank *predictFile as prediction, fit otherwise
	if *predictFile != "" {
		m, err := loadModel(*modelFile)
		if err != nil {
			fatal("error opening model file", err.Error())
		}

		pred, err := m.Predict(d)
		if err != nil {
			fatal(err.Error())
		}

		// write the predictions to file
		o, err := os.Create(*predictFile)
		if err != nil {
			fatal("error creating", *predictFile, err.Error())
		}
		defer o.Close()

		err = writePred(o, pred)
		if err != nil {
			fatal("error writing predictions", err.Error())
		}
		os.Exit(0)

	} else {
		// must be model fitting
		opt, err := parseModelOpts()
		if err != nil {
			fatal("invalid model option", err.Error())
		}

		// fit model
		m := new(Model)
		m.Fit(d, opt)

		// save model to disk
		o, err := os.Create(*modelFile)
		if err != nil {
			fatal("error saving model", err.Error())
		}
		defer o.Close()

		err = m.Save(o)
		if err != nil {
			fatal("error saving model", err.Error())
		}

		// write var importance to file
		if *impFile != "" {
			f, err := os.Create(*impFile)
			if err != nil {
				fatal("error saving variable importance", err.Error())
			}
			defer f.Close()
			err = m.SaveVarImp(f)
			if err != nil {
				fatal("error saving variable importance", err.Error())
			}
		}

		m.Report(os.Stderr)
	}
}

func loadModel(fName string) (*Model, error) {
	f, err := os.Open(*modelFile)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	m := new(Model)
	err = m.Load(f)
	return m, err
}

func fatal(a ...interface{}) {
	fmt.Fprintln(os.Stderr, a...)
	os.Exit(1)
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
