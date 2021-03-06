package forest

import (
	"math"
	"testing"
)

func TestIrisFitPredict(t *testing.T) {
	clf := NewClassifier(NumTrees(10))

	clf.Fit(X, Y)

	pred := clf.Predict(X)

	correctFrac := 0.0
	contrib := 1.0 / float64(len(pred))
	for i := range Y {
		if Y[i] == clf.Classes[pred[i]] {
			correctFrac += contrib
		}
	}

	if correctFrac < 0.98 {
		t.Errorf("expected accuracy on iris data to be at least 0.98, got: %f", correctFrac)
	}
}

func TestIrisVariableImportance(t *testing.T) {
	clf := NewClassifier(NumTrees(10))
	clf.Fit(X, Y)

	sum := 0.0
	for _, val := range clf.VarImp() {
		sum += val
	}

	if math.Abs(sum-1.0) > 1e-7 {
		t.Error("expected variable importance to sum to 1, got:", sum)
	}
}

func TestIrisOOBError(t *testing.T) {
	clf := NewClassifier(NumTrees(10), ComputeOOB)

	clf.Fit(X, Y)

	// check oob accuracy
	if clf.Accuracy < 0.90 {
		t.Errorf("expected oob accuracy to be at least 0.90, got: %f", clf.Accuracy)
	}

	// check confusion matrix
	for i := range clf.ConfusionMatrix {
		if clf.ConfusionMatrix[i][i] < 40 || clf.ConfusionMatrix[i][i] > 50 {
			t.Errorf("expected confusion matrix entry to be at least 45 and less than 50, got: %d", clf.ConfusionMatrix[i][i])
		}
	}
}

func BenchmarkIrisFit(b *testing.B) {
	for i := 0; i < b.N; i++ {
		clf := NewClassifier(NumTrees(10))

		clf.Fit(X, Y)
	}
}

func BenchmarkIrisPredict(b *testing.B) {
	clf := NewClassifier(NumTrees(10))
	clf.Fit(X, Y)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = clf.Predict(X)
	}
}

var X = [][]float64{
	[]float64{3.5, 1.4, 5.1, 0.2},
	[]float64{3.0, 1.4, 4.9, 0.2},
	[]float64{3.2, 1.3, 4.7, 0.2},
	[]float64{3.1, 1.5, 4.6, 0.2},
	[]float64{3.6, 1.4, 5.0, 0.2},
	[]float64{3.9, 1.7, 5.4, 0.4},
	[]float64{3.4, 1.4, 4.6, 0.3},
	[]float64{3.4, 1.5, 5.0, 0.2},
	[]float64{2.9, 1.4, 4.4, 0.2},
	[]float64{3.1, 1.5, 4.9, 0.1},
	[]float64{3.7, 1.5, 5.4, 0.2},
	[]float64{3.4, 1.6, 4.8, 0.2},
	[]float64{3.0, 1.4, 4.8, 0.1},
	[]float64{3.0, 1.1, 4.3, 0.1},
	[]float64{4.0, 1.2, 5.8, 0.2},
	[]float64{4.4, 1.5, 5.7, 0.4},
	[]float64{3.9, 1.3, 5.4, 0.4},
	[]float64{3.5, 1.4, 5.1, 0.3},
	[]float64{3.8, 1.7, 5.7, 0.3},
	[]float64{3.8, 1.5, 5.1, 0.3},
	[]float64{3.4, 1.7, 5.4, 0.2},
	[]float64{3.7, 1.5, 5.1, 0.4},
	[]float64{3.6, 1.0, 4.6, 0.2},
	[]float64{3.3, 1.7, 5.1, 0.5},
	[]float64{3.4, 1.9, 4.8, 0.2},
	[]float64{3.0, 1.6, 5.0, 0.2},
	[]float64{3.4, 1.6, 5.0, 0.4},
	[]float64{3.5, 1.5, 5.2, 0.2},
	[]float64{3.4, 1.4, 5.2, 0.2},
	[]float64{3.2, 1.6, 4.7, 0.2},
	[]float64{3.1, 1.6, 4.8, 0.2},
	[]float64{3.4, 1.5, 5.4, 0.4},
	[]float64{4.1, 1.5, 5.2, 0.1},
	[]float64{4.2, 1.4, 5.5, 0.2},
	[]float64{3.1, 1.5, 4.9, 0.2},
	[]float64{3.2, 1.2, 5.0, 0.2},
	[]float64{3.5, 1.3, 5.5, 0.2},
	[]float64{3.6, 1.4, 4.9, 0.1},
	[]float64{3.0, 1.3, 4.4, 0.2},
	[]float64{3.4, 1.5, 5.1, 0.2},
	[]float64{3.5, 1.3, 5.0, 0.3},
	[]float64{2.3, 1.3, 4.5, 0.3},
	[]float64{3.2, 1.3, 4.4, 0.2},
	[]float64{3.5, 1.6, 5.0, 0.6},
	[]float64{3.8, 1.9, 5.1, 0.4},
	[]float64{3.0, 1.4, 4.8, 0.3},
	[]float64{3.8, 1.6, 5.1, 0.2},
	[]float64{3.2, 1.4, 4.6, 0.2},
	[]float64{3.7, 1.5, 5.3, 0.2},
	[]float64{3.3, 1.4, 5.0, 0.2},
	[]float64{3.2, 4.7, 7.0, 1.4},
	[]float64{3.2, 4.5, 6.4, 1.5},
	[]float64{3.1, 4.9, 6.9, 1.5},
	[]float64{2.3, 4.0, 5.5, 1.3},
	[]float64{2.8, 4.6, 6.5, 1.5},
	[]float64{2.8, 4.5, 5.7, 1.3},
	[]float64{3.3, 4.7, 6.3, 1.6},
	[]float64{2.4, 3.3, 4.9, 1.0},
	[]float64{2.9, 4.6, 6.6, 1.3},
	[]float64{2.7, 3.9, 5.2, 1.4},
	[]float64{2.0, 3.5, 5.0, 1.0},
	[]float64{3.0, 4.2, 5.9, 1.5},
	[]float64{2.2, 4.0, 6.0, 1.0},
	[]float64{2.9, 4.7, 6.1, 1.4},
	[]float64{2.9, 3.6, 5.6, 1.3},
	[]float64{3.1, 4.4, 6.7, 1.4},
	[]float64{3.0, 4.5, 5.6, 1.5},
	[]float64{2.7, 4.1, 5.8, 1.0},
	[]float64{2.2, 4.5, 6.2, 1.5},
	[]float64{2.5, 3.9, 5.6, 1.1},
	[]float64{3.2, 4.8, 5.9, 1.8},
	[]float64{2.8, 4.0, 6.1, 1.3},
	[]float64{2.5, 4.9, 6.3, 1.5},
	[]float64{2.8, 4.7, 6.1, 1.2},
	[]float64{2.9, 4.3, 6.4, 1.3},
	[]float64{3.0, 4.4, 6.6, 1.4},
	[]float64{2.8, 4.8, 6.8, 1.4},
	[]float64{3.0, 5.0, 6.7, 1.7},
	[]float64{2.9, 4.5, 6.0, 1.5},
	[]float64{2.6, 3.5, 5.7, 1.0},
	[]float64{2.4, 3.8, 5.5, 1.1},
	[]float64{2.4, 3.7, 5.5, 1.0},
	[]float64{2.7, 3.9, 5.8, 1.2},
	[]float64{2.7, 5.1, 6.0, 1.6},
	[]float64{3.0, 4.5, 5.4, 1.5},
	[]float64{3.4, 4.5, 6.0, 1.6},
	[]float64{3.1, 4.7, 6.7, 1.5},
	[]float64{2.3, 4.4, 6.3, 1.3},
	[]float64{3.0, 4.1, 5.6, 1.3},
	[]float64{2.5, 4.0, 5.5, 1.3},
	[]float64{2.6, 4.4, 5.5, 1.2},
	[]float64{3.0, 4.6, 6.1, 1.4},
	[]float64{2.6, 4.0, 5.8, 1.2},
	[]float64{2.3, 3.3, 5.0, 1.0},
	[]float64{2.7, 4.2, 5.6, 1.3},
	[]float64{3.0, 4.2, 5.7, 1.2},
	[]float64{2.9, 4.2, 5.7, 1.3},
	[]float64{2.9, 4.3, 6.2, 1.3},
	[]float64{2.5, 3.0, 5.1, 1.1},
	[]float64{2.8, 4.1, 5.7, 1.3},
	[]float64{3.3, 6.0, 6.3, 2.5},
	[]float64{2.7, 5.1, 5.8, 1.9},
	[]float64{3.0, 5.9, 7.1, 2.1},
	[]float64{2.9, 5.6, 6.3, 1.8},
	[]float64{3.0, 5.8, 6.5, 2.2},
	[]float64{3.0, 6.6, 7.6, 2.1},
	[]float64{2.5, 4.5, 4.9, 1.7},
	[]float64{2.9, 6.3, 7.3, 1.8},
	[]float64{2.5, 5.8, 6.7, 1.8},
	[]float64{3.6, 6.1, 7.2, 2.5},
	[]float64{3.2, 5.1, 6.5, 2.0},
	[]float64{2.7, 5.3, 6.4, 1.9},
	[]float64{3.0, 5.5, 6.8, 2.1},
	[]float64{2.5, 5.0, 5.7, 2.0},
	[]float64{2.8, 5.1, 5.8, 2.4},
	[]float64{3.2, 5.3, 6.4, 2.3},
	[]float64{3.0, 5.5, 6.5, 1.8},
	[]float64{3.8, 6.7, 7.7, 2.2},
	[]float64{2.6, 6.9, 7.7, 2.3},
	[]float64{2.2, 5.0, 6.0, 1.5},
	[]float64{3.2, 5.7, 6.9, 2.3},
	[]float64{2.8, 4.9, 5.6, 2.0},
	[]float64{2.8, 6.7, 7.7, 2.0},
	[]float64{2.7, 4.9, 6.3, 1.8},
	[]float64{3.3, 5.7, 6.7, 2.1},
	[]float64{3.2, 6.0, 7.2, 1.8},
	[]float64{2.8, 4.8, 6.2, 1.8},
	[]float64{3.0, 4.9, 6.1, 1.8},
	[]float64{2.8, 5.6, 6.4, 2.1},
	[]float64{3.0, 5.8, 7.2, 1.6},
	[]float64{2.8, 6.1, 7.4, 1.9},
	[]float64{3.8, 6.4, 7.9, 2.0},
	[]float64{2.8, 5.6, 6.4, 2.2},
	[]float64{2.8, 5.1, 6.3, 1.5},
	[]float64{2.6, 5.6, 6.1, 1.4},
	[]float64{3.0, 6.1, 7.7, 2.3},
	[]float64{3.4, 5.6, 6.3, 2.4},
	[]float64{3.1, 5.5, 6.4, 1.8},
	[]float64{3.0, 4.8, 6.0, 1.8},
	[]float64{3.1, 5.4, 6.9, 2.1},
	[]float64{3.1, 5.6, 6.7, 2.4},
	[]float64{3.1, 5.1, 6.9, 2.3},
	[]float64{2.7, 5.1, 5.8, 1.9},
	[]float64{3.2, 5.9, 6.8, 2.3},
	[]float64{3.3, 5.7, 6.7, 2.5},
	[]float64{3.0, 5.2, 6.7, 2.3},
	[]float64{2.5, 5.0, 6.3, 1.9},
	[]float64{3.0, 5.2, 6.5, 2.0},
	[]float64{3.4, 5.4, 6.2, 2.3},
	[]float64{3.0, 5.1, 5.9, 1.8},
}

var XNames = []string{"Sepal.Width", "Petal.Length", "Sepal.Length", "Petal.Width"}

var Y = []string{
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"setosa",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"versicolor",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
	"virginica",
}
