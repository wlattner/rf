package tree

import "math/rand"

func (t *Tree) build(X [][]float64, inx []int) {

	if t.MaxFeatures < 0 || t.MaxFeatures > t.nFeatures {
		t.MaxFeatures = t.nFeatures
	}

	if t.MinSplit < 0 {
		t.MinSplit = 2
	}

	if t.MinLeaf < 0 {
		t.MinLeaf = 1
	}

	features := make([]int, t.nFeatures)
	for i := range features {
		features[i] = i
	}

	t.Root = &Node{}

	s := new(buildStack)
	s.Push(&stackItem{t.Root, inx, []bool{}, 0})

	sp := newSplitter(X, t.v, t.MaxFeatures, t.MinLeaf, t.randState)

	for !s.Empty() {
		w := s.Pop()
		n := w.node

		t.v.init(w.inx)
		currentImpurity := t.v.initialVal()

		n.Impurity = currentImpurity
		n.Samples = len(w.inx)
		n.setValue(t.v.nodeVal())

		if len(w.inx) < t.MinSplit || len(w.inx) < 2*t.MinLeaf ||
			(t.MaxDepth > 0 && w.depth == t.MaxDepth) || currentImpurity <= 1e-7 {
			n.Leaf = true
			continue
		}

		split := sp.bestSplit(w.inx, w.constantFeatures)

		// partition w.inx into left and right
		if split.inx > 0 {
			i := 0
			j := len(w.inx)

			for i < j {
				if X[w.inx[i]][split.feature] < split.val {
					i++
				} else {
					j--
					w.inx[j], w.inx[i] = w.inx[i], w.inx[j]
				}
			}

			l, r := w.inx[:split.inx], w.inx[split.inx:]

			n.Left = &Node{}
			n.Right = &Node{}
			n.SplitVar = split.feature
			n.SplitVal = split.val

			s.Push(&stackItem{n.Left, l, split.constantFeatures, w.depth + 1})
			s.Push(&stackItem{n.Right, r, split.constantFeatures, w.depth + 1})
		} else {
			// couldn't find a split
			n.Leaf = true
		}
	}
}

type splitter struct {
	xBuf        []float64
	X           [][]float64
	maxFeatures int
	minLeaf     int
	features    []int
	v           valuer
	randState   *rand.Rand
}

type split struct {
	delta            float64
	val              float64
	feature          int
	inx              int
	constantFeatures []bool
}

func newSplitter(X [][]float64, v valuer, maxFeatures, minLeaf int, r *rand.Rand) *splitter {
	s := splitter{
		xBuf:        make([]float64, len(X)),
		X:           X,
		maxFeatures: maxFeatures,
		minLeaf:     minLeaf,
		features:    make([]int, len(X[0])),
		v:           v,
		randState:   r,
	}

	for i := range s.features {
		s.features[i] = i
	}

	return &s
}

func (s *splitter) bestSplit(inx []int, constantFeatures []bool) split {

	var (
		deltaBest float64 // best impurity improvement
		valBest   float64 // best threshold
		varBest   int     // best split var
		inxBest   = -1    // left = w.inx[:inxBest], right = w.inx[inxBest:]
	)

	var (
		j                  = len(s.features) - 1
		visited, nConstant int
	)

	for j > 0 && (visited < s.maxFeatures || visited <= nConstant) {
		k := s.randState.Intn(j + 1)
		currentFeature := s.features[k]
		s.features[k], s.features[j] = s.features[j], s.features[k]
		j--
		visited++

		if len(constantFeatures) > 0 && constantFeatures[currentFeature] {
			nConstant++
			continue
		}

		// copy current feature to buffer
		for i, id := range inx {
			s.xBuf[i] = s.X[id][currentFeature]
		}
		xt := s.xBuf[:len(inx)]

		bSort(xt, inx)

		if xt[len(xt)-1] <= xt[0]+1e-7 {
			nConstant++
			c := make([]bool, len(s.features))
			copy(c, constantFeatures)
			c[currentFeature] = true
			constantFeatures = c
			continue
		}

		// look for split pt of currentFeature
		s.v.reset()
		for i := 1; i < len(xt); i++ {
			if xt[i] <= xt[i-1]+1e-7 {
				continue
			}

			s.v.update(i)

			if i < s.minLeaf || len(xt)-i < s.minLeaf {
				continue
			}

			sp := (xt[i-1] + xt[i]) / 2.0

			if d := s.v.delta(); d > deltaBest {
				deltaBest = d
				varBest = currentFeature
				valBest = sp
				inxBest = i
			}
		}
	}

	return split{deltaBest, valBest, varBest, inxBest, constantFeatures}
}

type buildStack []*stackItem

func (s buildStack) Empty() bool        { return len(s) == 0 }
func (s *buildStack) Push(n *stackItem) { *s = append(*s, n) }
func (s *buildStack) Pop() *stackItem {
	d := (*s)[len(*s)-1]
	*s = (*s)[:len(*s)-1]
	return d
}

type stackItem struct {
	node             *Node
	inx              []int
	constantFeatures []bool
	depth            int
}
