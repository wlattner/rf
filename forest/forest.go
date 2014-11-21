// forest implements random forests as described in
// Louppe, G. (2014) "Understanding Random Forests: From Theory to Practice" (PhD thesis)
// http://arxiv.org/abs/1407.7502
//
// Most of the algorithms implemented in this package come from chapter 4 of the
// thesis.
package forest

import (
	"math/rand"
	"time"

	"github.com/wlattner/rf/tree"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

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
// Currently Gini and Entropy are the only implemented options. Impurity will
// be ignored for regression.
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

// ComputeOOB will compute mean squared error (Regressor) or overall accuracy
// and confusion matrix from out of bag samples for each tree.
func ComputeOOB(c forestConfiger) {
	c.setComputeOOB()
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
