package tree

import (
	"encoding/gob"
	"io"
	"math"
	"math/rand"
	"time"
)

// Classifier implements a decision tree classifier. The classifier
// should be initialized with NewClassifier.
type Classifier struct {
	//TODO: store nodes in a slice so we can serialize/deserialize later
	Root        *Node
	MinSplit    int // min node size for split
	MinLeaf     int // min leaf size for split
	MaxDepth    int // max depth
	MaxFeatures int // number of features to consider for splitting
	Classes     []string
	impurityFn  func(int, []int) float64
	randState   *rand.Rand
	nFeatures   int
}

// methods for the treeConfiger interface
func (c *Classifier) setMinSplit(n int) { c.MinSplit = n }
func (c *Classifier) setMinLeaf(n int)  { c.MinLeaf = n }
func (c *Classifier) setMaxDepth(n int) { c.MaxDepth = n }
func (c *Classifier) setImpurity(f ImpurityMeasure) {
	switch f {
	case Gini:
		c.impurityFn = gini
	case Entropy:
		c.impurityFn = entropy
	default:
		c.impurityFn = gini
	}
}
func (c *Classifier) setMaxFeatures(n int) { c.MaxFeatures = n }
func (c *Classifier) setRandState(n int64) { c.randState = rand.New(rand.NewSource(n)) }

// NewClassifier returns a configured/initialized decision tree classifier.
// If no options are passed, the returned Classifier will be equivalent to the
// following call:
//
//	clf := NewClassifier(MinSplit(2), MinLeaf(1), MaxDepth(-1), Impurity(tree.Gini))
func NewClassifier(options ...func(treeConfiger)) *Classifier {
	c := &Classifier{
		MinSplit:    2,
		MinLeaf:     1,
		MaxDepth:    -1,
		MaxFeatures: -1,
		impurityFn:  gini,
		randState:   rand.New(rand.NewSource(time.Now().UnixNano())),
	}

	for _, opt := range options {
		opt(c)
	}

	return c
}

// Fit constructs a tree from the provided features X, and labels Y.
func (t *Classifier) Fit(X [][]float64, Y []string) {
	// labels as integer ids
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

	inx := make([]int, len(Y))
	for i := 0; i < len(Y); i++ {
		inx[i] = i
	}

	t.fit(X, yIDs, inx, classes)
}

// FitInx constructs a tree as in Fit, but uses the inx slice to mask
// the examples in X and Y. The caller also needs to supply a slice of unique
// classes where the ith class corresponds to the integer id used in Y (a mapping
// of class id to class name). FitInx is intended to be used with a meta algorithm
// that rely on bootstrap sampling, such as RandomForest.
func (t *Classifier) FitInx(X [][]float64, Y []int, inx []int, classes []string) {
	t.fit(X, Y, inx, classes)
}

// classes should be a mapping from integer ids to string class names, len(classes)
// should equal max(Y)
func (t *Classifier) fit(X [][]float64, Y []int, inx []int, classes []string) {
	// all examples are in root node
	t.Root = &Node{Samples: len(inx)}

	t.Classes = classes

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

	classCtL := make([]int, len(classes))
	classCtR := make([]int, len(classes))
	classCtrZero := make([]int, len(classes))

	var s stack
	s.Push(&stackNode{node: t.Root, inx: inx})

	for !s.Empty() {
		w := s.Pop()
		n := w.node

		n.ClassCounts = make([]int, len(classes))
		for _, inx := range w.inx {
			n.ClassCounts[Y[inx]]++
		}

		n.Impurity = t.impurityFn(len(w.inx), n.ClassCounts)

		// TODO: this condition is getting complex
		if len(w.inx) < minSplit ||
			len(w.inx) < 2*minLeaf ||
			(t.MaxDepth > 0 && w.depth == t.MaxDepth) ||
			n.Impurity <= 1e-7 {
			// mark as leaf node, too small to split
			n.Leaf = true
		} else {

			// compute impurity for node

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

				// zero left crt
				copy(classCtL, classCtrZero) // faster than clearing w/ for loop
				// copy current class counts
				copy(classCtR, n.ClassCounts)

				v, d, pos := t.bestSplit(xt, Y, w.inx, n.Impurity, classCtL, classCtR)

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

				n.Left = &Node{Samples: len(l)}
				n.Right = &Node{Samples: len(r)}
				n.SplitVar = xBest
				n.SplitVal = vBest

				s.Push(&stackNode{node: n.Left, depth: w.depth + 1, inx: l, constantFeatures: w.constantFeatures})
				s.Push(&stackNode{node: n.Right, depth: w.depth + 1, inx: r, constantFeatures: w.constantFeatures})
			} else {
				// we couldn't split the node, mark as leaf node
				n.Leaf = true
			}
		}
	}
}

// Predict returns the most probable class id for each example. The id
// corresponds to the index of the class label in Classifier.Classes
func (t *Classifier) Predict(X [][]float64) []int {
	p := make([]int, len(X))

	for i := range p {
		n := t.Root
		for !n.Leaf {
			if X[i][n.SplitVar] > n.SplitVal {
				n = n.Right
			} else {
				n = n.Left
			}
		}

		maxCt := 0
		maxC := 0
		for class, count := range n.ClassCounts {
			if count > maxCt {
				maxCt = count
				maxC = class
			}
		}
		p[i] = maxC
	}

	return p
}

// PredictInx returns the most probable class index for each input example masked by inx.
// This function is intended for OOB error estimating.
func (t *Classifier) PredictID(X [][]float64, inx []int) []int {
	p := make([]int, len(inx))

	for i, id := range inx {
		n := t.Root
		for !n.Leaf {
			if X[id][n.SplitVar] > n.SplitVal {
				n = n.Right
			} else {
				n = n.Left
			}
		}

		maxCt := 0
		maxC := 0
		for class, count := range n.ClassCounts {
			if count > maxCt {
				maxCt = count
				maxC = class
			}
		}
		p[i] = maxC
	}
	return p
}

// PredictProb returns the class probability for each example. The indices
// of the return value correspond to Classifier.Classes.
func (t *Classifier) PredictProb(X [][]float64) [][]float64 {
	p := make([][]float64, len(X))

	for i := range p {
		n := t.Root
		for !n.Leaf {
			if X[i][n.SplitVar] > n.SplitVal {
				n = n.Right
			} else {
				n = n.Left
			}
		}

		row := make([]float64, len(n.ClassCounts))
		for i := range row {
			row[i] = float64(n.ClassCounts[i]) / float64(n.Samples)
		}
		p[i] = row
	}
	return p
}

// VarImp returns an estimate of the importance of the variables used to fit
// the tree.
func (t *Classifier) VarImp() []float64 {
	imp := make([]float64, t.nFeatures)

	var s stack
	s.Push(&stackNode{node: t.Root})

	for !s.Empty() {
		n := s.Pop()

		if !n.node.Leaf {
			imp[n.node.SplitVar] += (float64(n.node.Samples)*n.node.Impurity -
				float64(n.node.Right.Samples)*n.node.Right.Impurity -
				float64(n.node.Left.Samples)*n.node.Left.Impurity)

			s.Push(&stackNode{node: n.node.Left})
			s.Push(&stackNode{node: n.node.Right})
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

// Save serializes the Classifier using encoding/gob to an io.Writer.
func (t *Classifier) Save(w io.Writer) error {
	e := gob.NewEncoder(w)
	return e.Encode(t)
}

// Load deserializes the Classifier using encoding/gob from an io.Reader.
func (t *Classifier) Load(r io.Reader) error {
	d := gob.NewDecoder(r)
	return d.Decode(t)
}

// this function takes a lot of args
// classCtl and classCtR should be initialized by the caller, classCtL should
// be all zeros, classCtR should be the counts for the current node
func (t *Classifier) bestSplit(xi []float64, y []int, inx []int, dInit float64,
	classCtL []int, classCtR []int) (float64, float64, int) {

	var (
		dBest, vBest, v, d float64
		pos                = -1
	)

	n := len(xi)
	nLeft := 0
	nRight := n

	var lastCtr int // last time the counters were incremented

	for i := 1; i < n; i++ {
		if xi[i] <= xi[i-1]+1e-7 {
			continue // can't split when x_i == x_i+1
		}

		for j := lastCtr; j < i; j++ {
			yVal := y[inx[j]]

			// increment class count and n for examples moving to left
			nLeft++
			classCtL[yVal]++
			// decrement class count and n for examples moving from right
			nRight--
			classCtR[yVal]--
		}
		lastCtr = i

		// make sure the left and right splits are large enough
		// we could roll this condition into the for loop,
		// start at i = minLeaf and end at n - minLeaf
		if t.MinLeaf > 0 && (nLeft < t.MinLeaf || nRight < t.MinLeaf) {
			continue
		}

		v = (xi[i-1] + xi[i]) / 2.0 // candidate split

		// compute entropy/gini
		iR := t.impurityFn(nRight, classCtR)
		iL := t.impurityFn(nLeft, classCtL)

		d = dInit - (float64(nLeft)/float64(n))*iL - (float64(nRight)/float64(n))*iR

		if d > dBest {
			dBest = d
			vBest = v
			pos = nLeft
		}

	}
	return vBest, dBest, pos
}

// gini impurity
// i_t = sum over k p(c_k|t) (1 - p(c_k|t))
func gini(n int, ct []int) float64 {
	g := 0.0
	for _, c := range ct {
		if c > 0 {
			p := float64(c) / float64(n)
			g += p * p
		}
	}
	return 1.0 - g
}

// entropy
// e_t = sum over k p(c_k|t) log p(c_k|t)
func entropy(n int, ct []int) float64 {
	e := 0.0
	for _, c := range ct {
		if c > 0 {
			p := float64(c) / float64(n)
			e -= p * math.Log2(p)
		}
	}
	return e
}

type stackNode struct {
	inx              []int
	constantFeatures []bool
	depth            int
	node             *Node
}

type Node struct {
	Left     *Node
	Right    *Node
	SplitVar int
	SplitVal float64
	//TODO: do we need to store class counts at each node?
	ClassCounts []int
	Impurity    float64
	Leaf        bool
	Samples     int
}

// lifo stack for unexpanded nodes
type stack []*stackNode

func (s stack) Empty() bool        { return len(s) == 0 }
func (s *stack) Push(n *stackNode) { *s = append(*s, n) }
func (s *stack) Pop() *stackNode {
	d := (*s)[len(*s)-1]
	*s = (*s)[:len(*s)-1]
	return d
}
