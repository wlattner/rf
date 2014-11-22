// tree implements classification and regression trees as described in
// Louppe, G. (2014) "Understanding Random Forests: From Theory to Practice" (PhD thesis)
// http://arxiv.org/abs/1407.7502
//
// Most of the algorithms implemented in this package come from chapter 3 of the
// thesis. The Fit follows Algorithm 3.2, the bestSplit method follows Algorithm 3.4.
package tree

import "math/rand"

type Tree struct {
	Root        *Node
	MinSplit    int      // min node size for split
	MinLeaf     int      // min leaf size for split
	MaxDepth    int      // max depth
	MaxFeatures int      // number of features to consider for splitting
	Classes     []string // will be blank for regression
	randState   *rand.Rand
	nFeatures   int
	v           valuer
}

type Node struct {
	Left, Right *Node
	SplitVar    int
	SplitVal    float64
	Impurity    float64
	Leaf        bool
	Samples     int
	ClassCounts []int   // will be nil for regression
	Value       float64 // fill be 0 for classification
}

func (n *Node) setValue(v interface{}) {
	switch val := v.(type) {
	case float64:
		n.Value = val
	case []int:
		n.ClassCounts = val
	}
}

// MinSplit limits the size for a node to be split vs marked as a leaf
func MinSplit(n int) func(*Tree) {
	return func(t *Tree) {
		t.MinSplit = n
	}
}

// MinLeaf limits the size of a child/leaf node for a split
// threshold to be considered
func MinLeaf(n int) func(*Tree) {
	return func(t *Tree) {
		t.MinLeaf = n
	}
}

// MaxDepth limits the depth of the fitted tree. Specifying -1 for n will
// grow a full tree, subject to MinLeaf and MinSplit constraints.
func MaxDepth(n int) func(*Tree) {
	return func(t *Tree) {
		t.MaxDepth = n
	}
}

// MaxFeatures limits the number of features considered for splitting at each
// step. If not provided or -1 then all features are considered.
func MaxFeatures(n int) func(*Tree) {
	return func(t *Tree) {
		t.MaxFeatures = n
	}
}

// RandState sets the seed for the random number generator
func RandState(n int64) func(*Tree) {
	return func(t *Tree) {
		t.randState = rand.New(rand.NewSource(n))
	}
}

type nodeStack []*Node

func (s nodeStack) Empty() bool   { return len(s) == 0 }
func (s *nodeStack) Push(n *Node) { *s = append(*s, n) }
func (s *nodeStack) Pop() *Node {
	d := (*s)[len(*s)-1]
	*s = (*s)[:len(*s)-1]
	return d
}
