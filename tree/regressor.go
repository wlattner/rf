package tree

import (
	"math/rand"
	"time"
)

type Regressor struct {
	Root        *RegNode
	MinSplit    int
	MinLeaf     int
	MaxDepth    int
	MaxFeatures int
	randState   *rand.Rand
	nFeatures   int
}

// methods for treeConfiger interface
func (c *Regressor) setMinSplit(n int)             { c.MinSplit = n }
func (c *Regressor) setMinLeaf(n int)              { c.MinLeaf = n }
func (c *Regressor) setMaxDepth(n int)             { c.MaxDepth = n }
func (c *Regressor) setImpurity(f ImpurityMeasure) {}
func (c *Regressor) setMaxFeatures(n int)          { c.MaxFeatures = n }
func (c *Regressor) setRandState(n int64)          { c.randState = rand.New(rand.NewSource(n)) }

// NewRegressor returns a configured/initialized regression tree.
// If no options are passed, the returned Regressor will be equivalent to
// the following call:
//
//	reg := NewRegressor(MinSplit(2), MinLeaf(1), MaxDepth(-1))
func NewRegressor(options ...func(treeConfiger)) *Regressor {
	r := &Regressor{
		MinSplit:    2,
		MinLeaf:     1,
		MaxDepth:    -1,
		MaxFeatures: -1,
		randState:   rand.New(rand.NewSource(time.Now().UnixNano())),
	}

	for _, opt := range options {
		opt(r)
	}

	return r
}

// Fit constructs a tree from the provided features X, and targets Y.
func (t *Regressor) Fit(X [][]float64, Y []float64) {
	inx := make([]int, len(Y))
	for i := 0; i < len(Y); i++ {
		inx[i] = i
	}

	t.FitInx(X, Y, inx)
}

// FitInx constructs a tree as in Fit, but uses only the indices
// of X and Y specified in inx.
func (t *Regressor) FitInx(X [][]float64, Y []float64, inx []int) {
	t.Root = &RegNode{Samples: len(inx)}

	t.nFeatures = len(X[0])

	maxFeatures := t.MaxFeatures
	if maxFeatures < 0 {
		maxFeatures = t.nFeatures
	}

	minSplit := t.MinSplit
	if minSplit < 0 {
		minSplit = 2
	}

	minLeaf := t.MinLeaf
	if minLeaf < 0 {
		minLeaf = 1
	}

	features := make([]int, len(X[0]))
	for i := range features {
		features[i] = i
	}

	// working copies of features and labels
	xBuf := make([]float64, len(inx))

	var s regStack
	s.Push(&regStackNode{node: t.Root, inx: inx})

	for !s.Empty() {
		w := s.Pop()
		n := w.node

		n.Impurity, n.Value = meanVar(Y, w.inx)

		// TODO: this condition is getting complex
		if len(w.inx) < minSplit ||
			len(w.inx) < 2*minLeaf ||
			(t.MaxDepth > 0 && w.depth == t.MaxDepth) ||
			n.Impurity <= 1e-7 {
			// mark as leaf node, too small to split
			n.Leaf = true
		} else {
			var (
				dBest float64 // best impurity improvement
				vBest float64 // best threshold
				xBest int     // best split var
				iBest = -1    // left = w.inx[:iBest], right = w.inx[iBest:]
			)

			// sample maxFeatures from features using Fisher-Yates,
			// Algorithm P, Knuth, The Art of Computer Programming Vol. 2, p. 145
			j := t.nFeatures - 1
			visited := 0
			nDrawnConstant := 0
			// need to visit at least one non-constant feature
			for j > 0 && (visited < maxFeatures || visited <= nDrawnConstant) {

				k := rand.Intn(j + 1)
				currentFeature := features[k]
				features[k], features[j] = features[j], features[k]

				j--
				visited++

				// do work on feature[j]
				if len(w.constantFeatures) > 0 && w.constantFeatures[currentFeature] {
					nDrawnConstant++
					continue
				}

				// copy feature values to buffer
				for i, inx := range w.inx {
					xBuf[i] = X[inx][currentFeature]
				}
				xt := xBuf[:len(w.inx)]

				// sort labels and indices by the value of the ith feature
				bSort(xt, w.inx)

				//TODO: find a better way to share the constant feature list with
				// child nodes
				if xt[len(xt)-1] <= xt[0]+1e-7 {
					nDrawnConstant++
					c := make([]bool, t.nFeatures)
					copy(c, w.constantFeatures)
					c[currentFeature] = true
					w.constantFeatures = c
					continue // constant feature, skip
				}

				v, d, pos := t.bestSplit(xt, Y, w.inx, n.Impurity)

				if d > dBest {
					dBest = d
					vBest = v
					xBest = currentFeature
					iBest = pos
				}
			}

			if iBest > 0 {
				// partition w.inx into left/right
				i := 0
				j := len(w.inx)

				for i < j {
					if X[w.inx[i]][xBest] < vBest {
						i++
					} else {
						j--
						w.inx[j], w.inx[i] = w.inx[i], w.inx[j]
					}
				}

				l, r := w.inx[:iBest], w.inx[iBest:]

				n.Left = &RegNode{Samples: len(l)}
				n.Right = &RegNode{Samples: len(r)}
				n.SplitVar = xBest
				n.SplitVal = vBest

				s.Push(&regStackNode{node: n.Left, depth: w.depth + 1, inx: l, constantFeatures: w.constantFeatures})
				s.Push(&regStackNode{node: n.Right, depth: w.depth + 1, inx: r, constantFeatures: w.constantFeatures})
			} else {
				// we couldn't split the node, mark as leaf node
				n.Leaf = true
			}
		}
	}
}

func (t *Regressor) bestSplit(xi []float64, Y []float64, inx []int, dInit float64) (float64, float64, int) {
	var (
		dBest, vBest, v, d float64
		pos                = -1
	)

	n := len(xi)
	nLeft := 0
	nRight := n

	var lastCtr int // last time the counters were incremented

	// l/r counters
	var (
		sR, ssR float64 // sum Y right, sum y^2 right
		sL, ssL float64 // sum Y left, sum y^2 right
	)

	// all examples on right to start
	for _, i := range inx {
		sR += Y[i]
		ssR += Y[i] * Y[i]
	}

	for i := 1; i < n; i++ {
		if xi[i] <= xi[i-1]+1e-7 {
			continue // can't split when x_i == x_i+1
		}

		for j := lastCtr; j < i; j++ {
			yVal := Y[inx[j]]

			// increment class count and n for examples moving to left
			nLeft++
			sL += yVal
			ssL += yVal * yVal
			// decrement class count and n for examples moving from right
			nRight--
			sR -= yVal
			ssR -= yVal * yVal
		}
		lastCtr = i

		// make sure the left and right splits are large enough
		// we could roll this condition into the for loop,
		// start at i = minLeaf and end at n - minLeaf
		if t.MinLeaf > 0 && (nLeft < t.MinLeaf || nRight < t.MinLeaf) {
			continue
		}

		v = (xi[i-1] + xi[i]) / 2.0 // candidate split

		// l/r variance
		rMean := sR / float64(nRight)
		iR := ssR/float64(nRight) - rMean*rMean
		lMean := sL / float64(nLeft)
		iL := ssL/float64(nLeft) - lMean*lMean

		d = dInit - (float64(nLeft)/float64(n))*iL - (float64(nRight)/float64(n))*iR

		if d > dBest {
			dBest = d
			vBest = v
			pos = nLeft
		}

	}
	return vBest, dBest, pos
}

func meanVar(Y []float64, inx []int) (float64, float64) {
	var ss, s float64

	for _, i := range inx {
		ss += Y[i] * Y[i]
		s += Y[i]
	}

	mean := s / float64(len(inx))
	v := ss/float64(len(inx)) - mean*mean

	return v, mean
}

// Predict returns the expected value for each example X.
func (t *Regressor) Predict(X [][]float64) []float64 {
	p := make([]float64, len(X))

	for i := range p {
		n := t.Root
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

// Predict returns the expected value for each example selected by inx.
func (t *Regressor) PredictInx(X [][]float64, inx []int) []float64 {
	p := make([]float64, len(inx))

	for i, id := range inx {
		n := t.Root
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
func (t *Regressor) VarImp() []float64 {
	imp := make([]float64, t.nFeatures)

	var s regStack
	s.Push(&regStackNode{node: t.Root})

	for !s.Empty() {
		n := s.Pop()

		if !n.node.Leaf {
			imp[n.node.SplitVar] += (float64(n.node.Samples)*n.node.Impurity -
				float64(n.node.Right.Samples)*n.node.Right.Impurity -
				float64(n.node.Left.Samples)*n.node.Left.Impurity)

			s.Push(&regStackNode{node: n.node.Left})
			s.Push(&regStackNode{node: n.node.Right})
		}
	}
	nSamples := float64(t.Root.Samples)
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

type RegNode struct {
	Left     *RegNode
	Right    *RegNode
	SplitVar int
	SplitVal float64
	Value    float64
	Impurity float64
	Leaf     bool
	Samples  int
}

type regStackNode struct {
	inx              []int
	constantFeatures []bool
	depth            int
	node             *RegNode
}

type regStack []*regStackNode

func (s regStack) Empty() bool           { return len(s) == 0 }
func (s *regStack) Push(n *regStackNode) { *s = append(*s, n) }
func (s *regStack) Pop() *regStackNode {
	d := (*s)[len(*s)-1]
	*s = (*s)[:len(*s)-1]
	return d
}
