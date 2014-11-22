package tree

import (
	"math/rand"
	"time"
)

type Regressor struct {
	Tree
}

// NewRegressor returns a configured/initialized regression tree.
// If no options are passed, the returned Regressor will be equivalent to
// the following call:
//
//	reg := NewRegressor(MinSplit(2), MinLeaf(1), MaxDepth(-1))
func NewRegressor(options ...func(*Tree)) *Regressor {
	t := Tree{
		MinSplit:    2,
		MinLeaf:     1,
		MaxDepth:    -1,
		MaxFeatures: -1,
		randState:   rand.New(rand.NewSource(time.Now().UnixNano())),
	}

	for _, opt := range options {
		opt(&t)
	}

	reg := Regressor{t}

	return &reg
}

// Fit constructs a tree from the provided features X, and targets Y.
func (r *Regressor) Fit(X [][]float64, Y []float64) {
	inx := make([]int, len(Y))
	for i := 0; i < len(Y); i++ {
		inx[i] = i
	}

	r.FitInx(X, Y, inx)
}

// FitInx constructs a tree as in Fit, but uses only the examples
// referenced in inx.
func (r *Regressor) FitInx(X [][]float64, Y []float64, inx []int) {
	r.nFeatures = len(X[0])
	r.v = newVarValuer(Y)
	r.build(X, inx)
}

// Predict returns the expected value for each example X.
func (r *Regressor) Predict(X [][]float64) []float64 {
	p := make([]float64, len(X))

	for i := range p {
		n := r.Root
		for !n.Leaf {
			if X[i][n.SplitVar] > n.SplitVal {
				n = n.Right
			} else {
				n = n.Left
			}
		}
		p[i] = n.Value
	}
	return p
}

// Predict returns the expected value for each example X referenced in
// inx.
func (r *Regressor) PredictInx(X [][]float64, inx []int) []float64 {
	p := make([]float64, len(inx))

	for i, id := range inx {
		n := r.Root
		for !n.Leaf {
			if X[id][n.SplitVar] > n.SplitVal {
				n = n.Right
			} else {
				n = n.Left
			}
		}
		p[i] = n.Value
	}
	return p
}

// VarImp returns an estimate of the importance of the variables used to fit
// the tree.
func (r *Regressor) VarImp() []float64 {
	imp := make([]float64, r.nFeatures)

	s := new(nodeStack)
	s.Push(r.Root)

	for !s.Empty() {
		n := s.Pop()

		if !n.Leaf {
			imp[n.SplitVar] += (float64(n.Samples)*n.Impurity -
				float64(n.Right.Samples)*n.Right.Impurity -
				float64(n.Left.Samples)*n.Left.Impurity)

			s.Push(n.Left)
			s.Push(n.Right)
		}
	}
	nSamples := float64(r.Root.Samples)
	total := 0.0
	for i := range imp {
		imp[i] /= nSamples
		total += imp[i]
	}

	// normalize
	for i := range imp {
		imp[i] /= total
	}

	return imp
}
