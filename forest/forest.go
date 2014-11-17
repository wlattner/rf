// forest implements random forests as described in
// Louppe, G. (2014) "Understanding Random Forests: From Theory to Practice" (PhD thesis)
// http://arxiv.org/abs/1407.7502
//
// Most of the algorithms implemented in this package come from chapter 4 of the
// thesis.
package forest

import (
	"encoding/gob"
	"io"
	"math"
	"math/rand"

	"github.com/wlattner/rf/tree"
)

type ForestClassifier struct {
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
}

// methods for the forestConfiger interface
func (c *ForestClassifier) setMinSplit(n int)                  { c.MinSplit = n }
func (c *ForestClassifier) setMinLeaf(n int)                   { c.MinLeaf = n }
func (c *ForestClassifier) setMaxDepth(n int)                  { c.MaxDepth = n }
func (c *ForestClassifier) setImpurity(f tree.ImpurityMeasure) { c.impurity = f }
func (c *ForestClassifier) setMaxFeatures(n int)               { c.MaxFeatures = n }
func (c *ForestClassifier) setNumTrees(n int)                  { c.NTrees = n }
func (c *ForestClassifier) setNumWorkers(n int)                { c.nWorkers = n }
func (c *ForestClassifier) setComputeOOB()                     { c.computeOOB = true }

type forestConfiger interface {
	setMinSplit(n int)
	setMinLeaf(n int)
	setMaxDepth(n int)
	setImpurity(f tree.ImpurityMeasure)
	setMaxFeatures(n int)
	setNumTrees(n int)
	setNumWorkers(n int)
	setComputeOOB()
}

var (
	Gini    = tree.Gini
	Entropy = tree.Entropy
)

// MinSplit limits the size for a node to be split vs marked as a leaf
func MinSplit(n int) func(forestConfiger) {
	return func(c forestConfiger) {
		c.setMinSplit(n)
	}
}

// MinLeaf limits the size of a child/leaf node for a split
// threshold to be considered
func MinLeaf(n int) func(forestConfiger) {
	return func(c forestConfiger) {
		c.setMinLeaf(n)
	}
}

// MaxDepth limits the depth of the fitted tree. Specifying -1 for n will
// grow a full tree, subject to MinLeaf and MinSplit constraints.
func MaxDepth(n int) func(forestConfiger) {
	return func(c forestConfiger) {
		c.setMaxDepth(n)
	}
}

// Impurity sets the impurity measure used to evaluate each candidate split.
// Currently Gini and Entropy are the only implemented options.
func Impurity(f tree.ImpurityMeasure) func(forestConfiger) {
	return func(c forestConfiger) {
		c.setImpurity(f)
	}
}

// MaxFeatures limits the number of features considered for splitting at each
// step. If not provided or -1 then all features are considered.
func MaxFeatures(n int) func(forestConfiger) {
	return func(c forestConfiger) {
		c.setMaxFeatures(n)
	}
}

// NumTrees sets the number of trees used in the random forest.
func NumTrees(n int) func(forestConfiger) {
	return func(c forestConfiger) {
		c.setNumTrees(n)
	}
}

// NumWorkers sets the number of workes used to fit trees; ensure
// GOMAXPROCS is also set > 1 to take advantage of multi cpu.
func NumWorkers(n int) func(forestConfiger) {
	return func(c forestConfiger) {
		c.setNumWorkers(n)
	}
}

// ComputeOOB computes the confusion matrix from out of bag samples
// for each tree.
func ComputeOOB() func(forestConfiger) {
	return func(c forestConfiger) {
		c.setComputeOOB()
	}
}

// NewClassifier returns a configured/initialized random forest classifier.
// If no options are passed, the returned Classifier will be equivalent to
// the following call:
//
//	clf := NewClassifier(NumTrees(10), MaxFeatures(-1), MinSplit(2), MinLeaf(1),
//		MaxDepth(-1), Impurity(Gini), NumWorkers(1))
func NewClassifier(options ...func(forestConfiger)) *ForestClassifier {
	f := &ForestClassifier{
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
func (f *ForestClassifier) Fit(X [][]float64, Y []string) {
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

	f.Trees = make([]*tree.Classifier, f.NTrees)

	if f.MaxFeatures < 0 {
		f.MaxFeatures = int(math.Sqrt(float64(len(X[0]))))
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
					tree.MaxDepth(f.MaxDepth), tree.Impurity(f.impurity),
					tree.MaxFeatures(f.MaxFeatures), tree.RandState(int64(id)))
				clf.FitInx(X, yIDs, w.inx, classes)

				w.t = clf

				if f.computeOOB {
					w.confMat = oobConfusionMat(X, yIDs, w.inBag, w.t)
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

	if f.computeOOB {
		for _ = range f.Classes {
			f.ConfusionMatrix = append(f.ConfusionMatrix, make([]int, len(f.Classes)))
		}
	}

	for i := range f.Trees {
		w := <-out
		f.Trees[i] = w.t

		if f.computeOOB {
			for row := range w.confMat {
				for col := range w.confMat {
					f.ConfusionMatrix[row][col] += w.confMat[row][col]
				}
			}
		}
	}
}

// Predict returns the most probable label for each example.
func (f *ForestClassifier) Predict(X [][]float64) []string {
	p := f.PredictProb(X)
	maxC := make([]string, len(X))

	for i := range maxC {
		// find the max vote
		var (
			maxP float64
			maxJ int
		)

		for j := range p[i] {
			if p[i][j] > maxP {
				maxP = p[i][j]
				maxJ = j
			}
		}
		maxC[i] = f.Classes[maxJ]
	}

	return maxC
}

// PredictProb returns the class probability for each example. The indices of the
// return value correspond to Classifier.Classes.
func (f *ForestClassifier) PredictProb(X [][]float64) [][]float64 {
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

func (f *ForestClassifier) Save(w io.Writer) error {
	e := gob.NewEncoder(w)
	return e.Encode(f)
}

func (f *ForestClassifier) Load(r io.Reader) error {
	d := gob.NewDecoder(r)
	return d.Decode(f)
}

type fitTree struct {
	t       *tree.Classifier
	inx     []int
	inBag   []bool
	confMat [][]int
}

func bootstrapInx(n int) ([]int, []bool) {
	inBag := make([]bool, n)
	inx := make([]int, n)
	for i := range inx {
		id := rand.Intn(n)
		inx[i] = id
		inBag[id] = true
	}
	return inx, inBag
}

func oobConfusionMat(X [][]float64, Y []int, inBag []bool, t *tree.Classifier) [][]int {
	// find the indices not inBag
	var inx []int
	for i, in := range inBag {
		if !in {
			inx = append(inx, i)
		}
	}

	confusionMat := make([][]int, len(t.Classes))
	for i := range confusionMat {
		confusionMat[i] = make([]int, len(t.Classes))
	}

	pred := t.PredictID(X, inx)

	for i, id := range inx {
		confusionMat[Y[id]][pred[i]]++
	}

	return confusionMat
}
