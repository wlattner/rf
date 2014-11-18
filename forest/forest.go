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
	"time"

	"github.com/wlattner/rf/tree"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

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

	f.Trees = make([]*tree.Classifier, f.NTrees)

	if f.MaxFeatures < 0 {
		f.MaxFeatures = int(math.Sqrt(float64(len(X[0]))))
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

func (f *Classifier) Save(w io.Writer) error {
	e := gob.NewEncoder(w)
	return e.Encode(f)
}

func (f *Classifier) Load(r io.Reader) error {
	d := gob.NewDecoder(r)
	return d.Decode(f)
}

type fitTree struct {
	t     *tree.Classifier
	inx   []int
	inBag []bool
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
