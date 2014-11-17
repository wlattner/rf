package tree

// sort borrowed from the standard library
// https://code.google.com/p/go/source/browse/src/pkg/sort/sort.go?name=release
//
// The time complexity of the Fit method is bounded by sorting the features at
// each node. Specializing the sort algorithm instead of using the Sort
// interface reduces the running time by approximately 60%.
//
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func swap(x []float64, inx []int, i, j int) {
	x[i], x[j] = x[j], x[i]
	inx[i], inx[j] = inx[j], inx[i]
}

// Insertion sort
func insertionSort(x []float64, inx []int, a, b int) {
	for i := a + 1; i < b; i++ {
		for j := i; j > a && x[j] < x[j-1]; j-- {
			swap(x, inx, j, j-1)
		}
	}
}

// siftDown implements the heap property on data[lo, hi).
// first is an offset into the array where the root of the heap lies.
func siftDown(x []float64, inx []int, lo, hi, first int) {
	root := lo
	for {
		child := 2*root + 1
		if child >= hi {
			break
		}
		if child+1 < hi && x[first+child] < x[first+child+1] {
			child++
		}
		if !(x[first+root] < x[first+child]) {
			return
		}
		swap(x, inx, first+root, first+child)
		root = child
	}
}

func heapSort(x []float64, inx []int, a, b int) {
	first := a
	lo := 0
	hi := b - a

	// Build heap with greatest element at top.
	for i := (hi - 1) / 2; i >= 0; i-- {
		siftDown(x, inx, i, hi, first)
	}

	// Pop elements, largest first, into end of data.
	for i := hi - 1; i >= 0; i-- {
		swap(x, inx, first, first+i)
		siftDown(x, inx, lo, i, first)
	}
}

// Quicksort, following Bentley and McIlroy,
// ``Engineering a Sort Function,'' SP&E November 1993.

// medianOfThree moves the median of the three values data[a], data[b], data[c] into data[a].
func medianOfThree(x []float64, inx []int, a, b, c int) {
	m0 := b
	m1 := a
	m2 := c
	// bubble sort on 3 elements
	if x[m1] < x[m0] {
		swap(x, inx, m1, m0)
	}
	if x[m2] < x[m1] {
		swap(x, inx, m2, m1)
	}
	if x[m1] < x[m0] {
		swap(x, inx, m1, m0)
	}
	// now data[m0] <= data[m1] <= data[m2]
}

func swapRange(x []float64, inx []int, a, b, n int) {
	for i := 0; i < n; i++ {
		swap(x, inx, a+i, b+i)
	}
}

func doPivot(x []float64, inx []int, lo, hi int) (midlo, midhi int) {
	m := lo + (hi-lo)/2 // Written like this to avoid integer overflow.
	if hi-lo > 40 {
		// Tukey's ``Ninther,'' median of three medians of three.
		s := (hi - lo) / 8
		medianOfThree(x, inx, lo, lo+s, lo+2*s)
		medianOfThree(x, inx, m, m-s, m+s)
		medianOfThree(x, inx, hi-1, hi-1-s, hi-1-2*s)
	}
	medianOfThree(x, inx, lo, m, hi-1)

	// Invariants are:
	//	data[lo] = pivot (set up by ChoosePivot)
	//	data[lo <= i < a] = pivot
	//	data[a <= i < b] < pivot
	//	data[b <= i < c] is unexamined
	//	data[c <= i < d] > pivot
	//	data[d <= i < hi] = pivot
	//
	// Once b meets c, can swap the "= pivot" sections
	// into the middle of the slice.
	pivot := lo
	a, b, c, d := lo+1, lo+1, hi, hi
	for {
		for b < c {
			if x[b] < x[pivot] { // data[b] < pivot
				b++
			} else if !(x[pivot] < x[b]) { // data[b] = pivot
				swap(x, inx, a, b)
				a++
				b++
			} else {
				break
			}
		}
		for b < c {
			if x[pivot] < x[c-1] { // data[c-1] > pivot
				c--
			} else if !(x[c-1] < x[pivot]) { // data[c-1] = pivot
				swap(x, inx, c-1, d-1)
				c--
				d--
			} else {
				break
			}
		}
		if b >= c {
			break
		}
		// data[b] > pivot; data[c-1] < pivot
		swap(x, inx, b, c-1)
		b++
		c--
	}

	n := min(b-a, a-lo)
	swapRange(x, inx, lo, b-n, n)

	n = min(hi-d, d-c)
	swapRange(x, inx, c, hi-n, n)

	return lo + b - a, hi - (d - c)
}

func quickSort(x []float64, inx []int, a, b, maxDepth int) {
	for b-a > 7 {
		if maxDepth == 0 {
			heapSort(x, inx, a, b)
			return
		}
		maxDepth--
		mlo, mhi := doPivot(x, inx, a, b)
		// Avoiding recursion on the larger subproblem guarantees
		// a stack depth of at most lg(b-a).
		if mlo-a < b-mhi {
			quickSort(x, inx, a, mlo, maxDepth)
			a = mhi // i.e., quickSort(data, mhi, b)
		} else {
			quickSort(x, inx, mhi, b, maxDepth)
			b = mlo // i.e., quickSort(data, a, mlo)
		}
	}
	if b-a > 1 {
		insertionSort(x, inx, a, b)
	}
}

// Sort sorts data.
// It makes one call to data.Len to determine n, and O(n*log(n)) calls to
// data.Less and data.Swap. The sort is not guaranteed to be stable.
func bSort(x []float64, inx []int) {
	// Switch to heapsort if depth of 2*ceil(lg(n+1)) is reached.
	n := len(inx)
	maxDepth := 0
	for i := n; i > 0; i >>= 1 {
		maxDepth++
	}
	maxDepth *= 2
	quickSort(x, inx, 0, n, maxDepth)
}
