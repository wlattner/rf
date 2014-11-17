// tree implements classification and regression trees as described in
// Louppe, G. (2014) "Understanding Random Forests: From Theory to Practice" (PhD thesis)
// http://arxiv.org/abs/1407.7502
//
// Most of the algorithms implemented in this package come from chapter 3 of the
// thesis. The Fit follows Algorithm 3.2, the bestSplit method follows Algorithm 3.4.
package tree

import (
	"encoding/gob"
	"io"
	"math"
	"math/rand"
	"time"
)

type ImpurityMeasure int

const (
	Gini ImpurityMeasure = iota
	Entropy
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

// interface for configuration so we can use the same args/functions to set
// regression trees
type treeConfiger interface {
	setMinSplit(n int)
	setMinLeaf(n int)
	setMaxDepth(n int)
	setImpurity(f ImpurityMeasure)
	setMaxFeatures(n int)
	setRandState(n int64)
}

// MinSplit limits the size for a node to be split vs marked as a leaf
func MinSplit(n int) func(treeConfiger) {
	return func(c treeConfiger) {
		c.setMinSplit(n)
	}
}

// MinLeaf limits the size of a child/leaf node for a split
// threshold to be considered
func MinLeaf(n int) func(treeConfiger) {
	return func(c treeConfiger) {
		c.setMinLeaf(n)
	}
}

// MaxDepth limits the depth of the fitted tree. Specifying -1 for n will
// grow a full tree, subject to MinLeaf and MinSplit constraints.
func MaxDepth(n int) func(treeConfiger) {
	return func(c treeConfiger) {
		c.setMaxDepth(n)
	}
}

// Impurity sets the impurity measure used to evaluate each candidate split.
// Currently Gini and Entropy are the only implemented options.
func Impurity(f ImpurityMeasure) func(treeConfiger) {
	return func(c treeConfiger) {
		c.setImpurity(f)
	}
}

// MaxFeatures limits the number of features considered for splitting at each
// step. If not provided or -1 then all features are considered.
func MaxFeatures(n int) func(treeConfiger) {
	return func(c treeConfiger) {
		c.setMaxFeatures(n)
	}
}

// RandState sets the seed for the random number generator
func RandState(n int64) func(treeConfiger) {
	return func(c treeConfiger) {
		c.setRandState(n)
	}
}

// NewClassifier returns a configured/initialized decision tree classifier.
// If no options are passed, the returned Classifier will be equivalent the
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
	inx := make([]int, len(Y))
	for i := 0; i < len(Y); i++ {
		inx[i] = i
	}

	t.fit(X, Y, inx)
}

// FitInx constructs a tree as in Fit, but uses the inx slice to mask
// the examples in X and Y. FitInx is intended to be used with meta algorithm
// that rely on bootstrap sampling, such as RandomForest.
func (t *Classifier) FitInx(X [][]float64, Y []string, inx []int) {
	//TODO: []int for Y instead; caller's responsibility to keep track
	t.fit(X, Y, inx)
}

func (t *Classifier) fit(X [][]float64, Y []string, inx []int) {
	// all examples are in root node
	t.Root = &Node{Samples: len(inx)}

	// recode Y with integer ids
	var yIDs []int
	uniq := make(map[string]int)
	for _, val := range Y {
		id, ok := uniq[val]
		if !ok {
			id = len(uniq)
			uniq[val] = id
		}
		yIDs = append(yIDs, id)
	}

	// save class id to name mapping
	t.Classes = make([]string, len(uniq))
	for class, id := range uniq {
		t.Classes[id] = class
	}

	nFeatures := len(X[0])

	maxFeatures := t.MaxFeatures
	if maxFeatures < 0 {
		maxFeatures = nFeatures
	}

	features := make([]int, len(X[0]))
	for i := range features {
		features[i] = i
	}

	// working copies of features and labels
	xBuf := make([]float64, len(yIDs))

	classCtL := make([]int, len(uniq))
	classCtR := make([]int, len(uniq))
	classCtrZero := make([]int, len(uniq))

	var s stack
	s.Push(&stackNode{node: t.Root, inx: inx})

	for !s.Empty() {
		w := s.Pop()
		n := w.node

		n.ClassCounts = make([]int, len(uniq))
		for _, inx := range w.inx {
			n.ClassCounts[yIDs[inx]]++
		}

		// TODO: this condition is getting complex
		if (t.MinSplit > 0 && len(w.inx) < t.MinSplit) || (t.MaxDepth > 0 && w.depth == t.MaxDepth) {
			// mark as leaf node, too small to split
			n.Leaf = true
		} else {

			// compute impurity for node
			n.Impurity = t.impurityFn(len(w.inx), n.ClassCounts)

			var (
				dBest float64
				vBest float64
				xBest int
				iBest int
			)

			// sample from maxFeatures from features using Fisher-Yates,
			// Algorithm P, Knuth, The Art of Computer Programming Vol. 2, p. 145
			j := nFeatures - 1
			visited := 0
			for j > 0 && visited < maxFeatures {
				u := rand.Float64()
				k := int(float64(j) * u)
				features[k], features[j] = features[j], features[k]

				currentFeature := features[j]
				j--
				visited++

				// do work on feature[j]
				if len(w.constantFeatures) > 0 && w.constantFeatures[currentFeature] {
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
					c := make([]bool, nFeatures)
					copy(c, w.constantFeatures)
					c[currentFeature] = true
					w.constantFeatures = c
					continue // constant feature, skip
				}

				// zero left crt
				copy(classCtL, classCtrZero) // faster than clearing w/ for loop
				// copy current class counts
				copy(classCtR, n.ClassCounts)

				v, d, pos := t.bestSplit(xt, yIDs, w.inx, n.Impurity, classCtL, classCtR)

				if d > dBest {
					dBest = d
					vBest = v
					xBest = currentFeature
					iBest = pos
				}
			}

			if dBest > 1e-6 {
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

// Predict returns the most probable label for each example.
func (t *Classifier) Predict(X [][]float64) []string {
	p := make([]string, len(X))

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
		p[i] = t.Classes[maxC]
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
		pos                int
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
