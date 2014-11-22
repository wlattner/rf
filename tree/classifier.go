package tree

import (
	"math/rand"
	"time"
)

type Classifier struct {
	Tree
}

func NewClassifier(options ...func(*Tree)) *Classifier {
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

	clf := Classifier{t}

	return &clf
}

func (c *Classifier) Fit(X [][]float64, Y []string) {
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

	c.FitInx(X, yIDs, inx, classes)
}

func (c *Classifier) FitInx(X [][]float64, Y []int, inx []int, classes []string) {
	c.Classes = classes
	c.nFeatures = len(X[0])
	c.v = newGiniValuer(Y, classes)
	c.build(X, inx)
}

func (c *Classifier) Predict(X [][]float64) []int {
	p := make([]int, len(X))

	for i := range p {
		n := c.Root
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

func (c *Classifier) PredictInx(X [][]float64, inx []int) []int {
	p := make([]int, len(inx))

	for i, id := range inx {
		n := c.Root
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

func (c *Classifier) PredictProb(X [][]float64) [][]float64 {
	p := make([][]float64, len(X))

	for i := range p {
		n := c.Root
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

func (c *Classifier) VarImp() []float64 {
	imp := make([]float64, c.nFeatures)

	s := new(nodeStack)
	s.Push(c.Root)

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

	nSamples := float64(c.Root.Samples)
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
