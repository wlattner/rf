// tree implements classification and regression trees as described in
// Louppe, G. (2014) "Understanding Random Forests: From Theory to Practice" (PhD thesis)
// http://arxiv.org/abs/1407.7502
//
// Most of the algorithms implemented in this package come from chapter 3 of the
// thesis. The Fit follows Algorithm 3.2, the bestSplit method follows Algorithm 3.4.
package tree

type ImpurityMeasure int

const (
	Gini ImpurityMeasure = iota
	Entropy
)

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
// Currently Gini and Entropy are the only implemented options. The impurity
// setting will be ignored for regression.
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
