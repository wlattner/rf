package tree

import (
	"math"
	"testing"
)

func TestBestSplit(t *testing.T) {
	clf := NewClassifier()

	xi := []float64{0.08918780255911574, 0.097704546453666, 0.15739526725378827, 0.1772808696619108, 0.47001967423520297, 0.5621969807319502, 0.6055333992245421, 0.6462220030737842, 0.8020611535912714, 0.9244669313190392}
	y := []int{0, 0, 0, 0, 0, 1, 1, 1, 1, 0}
	classCount := []int{6, 4}
	inx := make([]int, len(y))
	for i := range inx {
		inx[i] = i
	}

	classCtL := make([]int, 2)
	classCtR := make([]int, 2)
	copy(classCtR, classCount)
	sp, gain, _ := clf.bestSplit(xi, y, inx, 0.48, classCtL, classCtR)

	spActual := (xi[4] + xi[5]) / 2.0
	if sp != spActual {
		t.Error("expected split to be:", spActual, " got:", sp)
	}
	gainActual := 0.32
	if math.Abs(gain-gainActual) > 1e-6 {
		t.Error("expected gain to be:", gainActual, "got:", gain)
	}
}

func TestBestSplitConstant(t *testing.T) {
	clf := NewClassifier()

	xi := []float64{1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1}
	y := []int{0, 0, 0, 0, 0, 1, 1, 1, 1, 0}
	classCount := []int{6, 4}
	inx := make([]int, len(y))
	for i := range inx {
		inx[i] = i
	}

	classCtL := make([]int, 2)
	classCtR := make([]int, 2)
	copy(classCtR, classCount)
	sp, gain, pos := clf.bestSplit(xi, y, inx, 0.48, classCtL, classCtR)
	spActual := 0.0 // feature is constant, should be no split
	if sp != spActual {
		t.Error("expected split to be:", spActual, " got:", sp)
	}
	gainActual := 0.0 // no split, no gain
	if gain != gainActual {
		t.Error("expected gain to be:", gainActual, "got:", gain)
	}
	posActual := -1
	if pos != posActual {
		t.Error("expected split pos to be:", posActual, "got:", pos)
	}
}

func TestBestSplitSomeConstant(t *testing.T) {
	clf := NewClassifier()

	xi := []float64{0.08918780255911574, 0.09, 0.09, 0.09, 0.47001967423520297, 0.5621969807319502, 0.6055333992245421, 0.6462220030737842, 0.8020611535912714, 0.9244669313190392}
	y := []int{0, 0, 0, 0, 0, 1, 1, 1, 1, 0}
	classCount := []int{6, 4}
	inx := make([]int, len(y))
	for i := range inx {
		inx[i] = i
	}

	classCtL := make([]int, 2)
	classCtR := make([]int, 2)
	copy(classCtR, classCount)
	sp, gain, _ := clf.bestSplit(xi, y, inx, 0.48, classCtL, classCtR)

	spActual := (xi[4] + xi[5]) / 2.0
	if sp != spActual {
		t.Error("expected split to be:", spActual, " got:", sp)
	}
	gainActual := 0.32
	if math.Abs(gain-gainActual) > 1e-6 {
		t.Error("expected gain to be:", gainActual, "got:", gain)
	}
}
