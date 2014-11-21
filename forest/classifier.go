package forest

import (
	"math"
	"time"

	"github.com/wlattner/rf/tree"
)

type Classifier struct {
	NTrees          int
	MinSplit        int
	MinLeaf         int
	MaxDepth        int
	MaxFeatures     int
	Classes         []string
	Trees           []*tree.Classifier
	impurity        tree.ImpurityMeasure
	nWorkers        int
	computeOOB      bool
	ConfusionMatrix [][]int
	Accuracy        float64
	NSample         int
	nFeatures       int
}

// methods for the forestConfiger interface
func (c *Classifier) setMinSplit(n int)                  { c.MinSplit = n }
func (c *Classifier) setMinLeaf(n int)                   { c.MinLeaf = n }
func (c *Classifier) setMaxDepth(n int)                  { c.MaxDepth = n }
func (c *Classifier) setImpurity(f tree.ImpurityMeasure) { c.impurity = f }
func (c *Classifier) setMaxFeatures(n int)               { c.MaxFeatures = n }
func (c *Classifier) setNumTrees(n int)                  { c.NTrees = n }
func (c *Classifier) setNumWorkers(n int)                { c.nWorkers = n }
func (c *Classifier) setComputeOOB()                     { c.computeOOB = true }

// NewClassifier returns a configured/initialized random forest classifier.
// If no options are passed, the returned Classifier will be equivalent to
// the following call:
//
//	clf := NewClassifier(NumTrees(10), MaxFeatures(-1), MinSplit(2), MinLeaf(1),
//		MaxDepth(-1), Impurity(Gini), NumWorkers(1))
func NewClassifier(options ...func(forestConfiger)) *Classifier {
	f := &Classifier{
		NTrees:      10,
		MaxFeatures: -1,
		MinSplit:    2,
		MinLeaf:     1,
		MaxDepth:    -1,
		impurity:    Gini,
	}

	for _, opt := range options {
		opt(f)
	}

	return f
}

// Fit constructs a forest from fitting n trees from the provided features X, and
// labels Y.
func (f *Classifier) Fit(X [][]float64, Y []string) {
	// labels as integer ids, ensure all trees know about all classes
	var yIDs []int
	uniq := make(map[string]int)
	var classes []string
	for _, val := range Y {
		id, ok := uniq[val]
		if !ok {
			id = len(uniq)
			uniq[val] = id
			classes = append(classes, val)
		}
		yIDs = append(yIDs, id)
	}
	f.Classes = classes
	f.NSample = len(yIDs)

	f.nFeatures = len(X[0])

	f.Trees = make([]*tree.Classifier, f.NTrees)

	if f.MaxFeatures < 0 {
		f.MaxFeatures = int(math.Sqrt(float64(f.nFeatures)))
	}

	var oobClassCtr *oobCtr
	if f.computeOOB {
		oobClassCtr = newOOBCtr(len(Y), len(f.Classes))
	}

	in := make(chan *fitTree)
	out := make(chan *fitTree)

	nWorkers := f.nWorkers
	if nWorkers < 1 {
		nWorkers = 1
	}

	// start workers
	for i := 0; i < nWorkers; i++ {
		go func(id int) {
			for w := range in {
				clf := tree.NewClassifier(tree.MinSplit(f.MinSplit), tree.MinLeaf(f.MinLeaf),
					tree.MaxDepth(f.MaxDepth), tree.Impurity(f.impurity), tree.MaxFeatures(f.MaxFeatures),
					tree.RandState(int64(id)*time.Now().UnixNano()))
				clf.FitInx(X, yIDs, w.inx, classes)

				w.t = clf

				if f.computeOOB {
					oobClassCtr.update(X, w.inBag, w.t)
				}

				out <- w
			}
		}(i)
	}

	// fill the queue
	go func() {
		for _ = range f.Trees {
			inx, inBag := bootstrapInx(len(X))
			in <- &fitTree{inx: inx, inBag: inBag}
		}
		close(in)
	}()

	for i := range f.Trees {
		w := <-out
		f.Trees[i] = w.t
	}

	if f.computeOOB {
		f.ConfusionMatrix, f.Accuracy = oobClassCtr.compute(yIDs)
	}
}

// Predict returns the most probable class id for each example. The id
// corresponds to the index of the class label in Classifier.Classes.
func (f *Classifier) Predict(X [][]float64) []int {
	classVotes := make([][]int, len(X))
	for i := range classVotes {
		classVotes[i] = make([]int, len(f.Classes))
	}

	for _, t := range f.Trees {
		for i, class := range t.Predict(X) {
			classVotes[i][class]++
		}
	}

	// find max class for each example
	maxClass := make([]int, len(X))

	for i := range maxClass {
		maxCt := 0
		maxC := 0
		for class, count := range classVotes[i] {
			if count > maxCt {
				maxCt = count
				maxC = class
			}
		}
		maxClass[i] = maxC
	}

	return maxClass
}

// PredictProb returns the class probability for each example. The indices of the
// return value correspond to Classifier.Classes.
func (f *Classifier) PredictProb(X [][]float64) [][]float64 {
	//TODO: weighted voting...
	probs := make([][]float64, len(X))
	// initialize the other dim
	for row := range probs {
		probs[row] = make([]float64, len(f.Classes))
	}

	for _, t := range f.Trees {
		tProbs := t.PredictProb(X)
		for row := range tProbs {
			for class := range tProbs[row] {
				probs[row][class] += tProbs[row][class] / float64(f.NTrees)
			}
		}
	}

	return probs
}

// VarImp returns importance scores for the model.
func (f *Classifier) VarImp() []float64 {
	imp := make([]float64, f.nFeatures)

	for _, t := range f.Trees {
		for inx, importance := range t.VarImp() {
			imp[inx] += importance / float64(f.NTrees)
		}
	}

	return imp
}

type fitTree struct {
	t     *tree.Classifier
	inx   []int
	inBag []bool
}

type oobCtr struct {
	classVotes [][]int // array of nExample x nClasses
}

func newOOBCtr(nExample, nClasses int) *oobCtr {
	classVotes := make([][]int, nExample)
	for i := range classVotes {
		classVotes[i] = make([]int, nClasses)
	}
	m := oobCtr{classVotes: classVotes}
	return &m
}

// accumulate oob predictions for a tree
func (o *oobCtr) update(X [][]float64, inBag []bool, t *tree.Classifier) {
	var inx []int
	for i, in := range inBag {
		if !in {
			inx = append(inx, i)
		}
	}

	pred := t.PredictID(X, inx)

	for i, sampleInx := range inx {
		o.classVotes[sampleInx][pred[i]]++
	}
}

// compute confusion matrix and overall accuracy from oob predictions
func (o *oobCtr) compute(Y []int) ([][]int, float64) {
	confMat := make([][]int, len(o.classVotes[0]))
	for i := range confMat {
		confMat[i] = make([]int, len(o.classVotes[0]))
	}

	for i, actual := range Y {
		// find max vote from forest
		maxClass := 0
		maxVotes := 0
		for class, nVotes := range o.classVotes[i] {
			if nVotes > maxVotes {
				maxVotes = nVotes
				maxClass = class
			}
		}

		confMat[actual][maxClass]++
	}

	correctCt := 0
	for i := range confMat {
		correctCt += confMat[i][i]
	}
	accuracy := float64(correctCt) / float64(len(Y))

	return confMat, accuracy
}
