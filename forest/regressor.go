package forest

import (
	"math"
	"time"

	"github.com/wlattner/rf/tree"
)

type Regressor struct {
	NTrees      int
	MinSplit    int
	MinLeaf     int
	MaxDepth    int
	MaxFeatures int
	Trees       []*tree.Regressor
	nWorkers    int
	computeOOB  bool
	MSE         float64
	RSquared    float64
	NSample     int
	nFeatures   int
	earlyStop   bool
}

// methods for the forestConfiger interface
func (c *Regressor) setMinSplit(n int)                  { c.MinSplit = n }
func (c *Regressor) setMinLeaf(n int)                   { c.MinLeaf = n }
func (c *Regressor) setMaxDepth(n int)                  { c.MaxDepth = n }
func (c *Regressor) setImpurity(f tree.ImpurityMeasure) {}
func (c *Regressor) setMaxFeatures(n int)               { c.MaxFeatures = n }
func (c *Regressor) setNumTrees(n int)                  { c.NTrees = n }
func (c *Regressor) setNumWorkers(n int)                { c.nWorkers = n }
func (c *Regressor) setComputeOOB()                     { c.computeOOB = true }
func (c *Regressor) setEarlyStop()                      { c.earlyStop = true }

// NewRegressor returns a configured/initilized random forest regressor.
// If no options are passed, the returned Regressor will be equivalent to
// the following call:
//
//	reg := NewRegressor(NumTrees(10), MaxFeatures(-1), MinSplit(2), MinLeaf(1),
//			MaxDepth(-1), NumWorkers(1))
func NewRegressor(options ...func(Configer)) *Regressor {
	f := &Regressor{
		NTrees:      10,
		MaxFeatures: -1,
		MinSplit:    2,
		MinLeaf:     1,
		MaxDepth:    -1,
	}

	for _, opt := range options {
		opt(f)
	}

	return f
}

// Fit constructs a forest from fitting n trees to the provided features X, and
// targets Y.
func (f *Regressor) Fit(X [][]float64, Y []float64) {
	f.NSample = len(Y)

	f.nFeatures = len(X[0])

	if f.MaxFeatures < 0 {
		f.MaxFeatures = f.nFeatures / 3
	}

	if f.earlyStop {
		f.computeOOB = true
	}

	var oob *oobRegCtr
	if f.computeOOB {
		oob = newOOBRegCtr(len(Y))
	}

	in := make(chan *fitRegTree)
	out := make(chan *fitRegTree)

	nWorkers := f.nWorkers
	if nWorkers < 1 {
		nWorkers = 1
	}

	// start workers
	for i := 0; i < nWorkers; i++ {
		go func(id int) {
			for w := range in {
				reg := tree.NewRegressor(tree.MinSplit(f.MinSplit), tree.MinLeaf(f.MinLeaf),
					tree.MaxDepth(f.MaxDepth), tree.MaxFeatures(f.MaxFeatures),
					tree.RandState(int64(id)*time.Now().UnixNano()))
				reg.FitInx(X, Y, w.inx)

				w.t = reg

				if f.computeOOB {
					oob.update(X, w.inBag, w.t)
				}

				out <- w
			}
		}(i)
	}

	go func() {
		for i := 0; i < f.NTrees; i++ {
			inx, inBag := bootstrapInx(len(X))
			in <- &fitRegTree{inx: inx, inBag: inBag}
		}
		close(in)
	}()

	// fill the queue
	var mse, prevMSE float64
	for i := 0; i < f.NTrees; i++ {
		w := <-out

		if f.earlyStop {
			mse, _ = oob.compute(Y)
			if i > 4 && math.Abs(mse-prevMSE) < 1e-6 { // oob error converged
				break
			}
			prevMSE = mse
		}

		f.Trees = append(f.Trees, w.t)
	}

	f.NTrees = len(f.Trees)

	if f.computeOOB {
		f.MSE, f.RSquared = oob.compute(Y)
	}
}

// Predict returns the expected value for each example.
func (f *Regressor) Predict(X [][]float64) []float64 {
	sum := make([]float64, len(X))

	for _, t := range f.Trees {
		for i, val := range t.Predict(X) {
			sum[i] += val
		}
	}

	for i := range sum {
		sum[i] /= float64(f.NTrees)
	}

	return sum
}

// VarImp returns importance scores for the model.
func (f *Regressor) VarImp() []float64 {
	imp := make([]float64, f.nFeatures)

	for _, t := range f.Trees {
		for inx, importance := range t.VarImp() {
			imp[inx] += importance / float64(f.NTrees)
		}
	}

	return imp
}

type fitRegTree struct {
	t     *tree.Regressor
	inx   []int
	inBag []bool
}

type oobRegCtr struct {
	sum []float64
	ct  []int
}

func newOOBRegCtr(nExample int) *oobRegCtr {
	sum := make([]float64, nExample)
	ct := make([]int, nExample)
	return &oobRegCtr{sum, ct}
}

func (o *oobRegCtr) update(X [][]float64, inBag []bool, t *tree.Regressor) {
	var inx []int
	for i, in := range inBag {
		if !in {
			inx = append(inx, i)
		}
	}

	pred := t.PredictInx(X, inx)

	for i, sampleInx := range inx {
		o.sum[sampleInx] += pred[i]
		o.ct[sampleInx]++
	}
}

// compute returns mean squared error and rsquared
func (o *oobRegCtr) compute(Y []float64) (float64, float64) {
	rss := 0.0 // residual sum square

	// tss of Y
	n := 0
	mean := 0.0
	tss := 0.0

	for i := range Y {
		// skip examples that were in all trees
		if o.ct[i] < 1 {
			continue
		}
		predVal := o.sum[i] / float64(o.ct[i])
		d := Y[i] - predVal
		rss += d * d

		// update var
		n++
		d = Y[i] - mean
		mean += d / float64(n)
		tss += d * (Y[i] - mean)
	}

	if n < 1 {
		tss = 0.0
	}

	rSquared := 1.0 - rss/tss
	mse := rss / float64(n)

	return mse, rSquared
}
