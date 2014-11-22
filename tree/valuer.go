package tree

type valuer interface {
	init(inx []int)
	reset()
	update(pos int)
	initialVal() float64
	delta() float64
	nodeVal() interface{} // this doesn't quite belong here
}

type baseValuer struct {
	iInitial float64
	posLast  int
	nLeft    int
	nRight   int
	inx      []int
}

// gini valuer for classification
type giniValuer struct {
	baseValuer
	classCt, classCtR, classCtL, zeroCtr []int
	Y                                    []int
}

func newGiniValuer(Y []int, classes []string) *giniValuer {
	return &giniValuer{
		zeroCtr:  make([]int, len(classes)),
		classCt:  make([]int, len(classes)),
		classCtR: make([]int, len(classes)),
		classCtL: make([]int, len(classes)),
		Y:        Y,
	}
}

func (c *giniValuer) nodeVal() interface{} {
	classCt := make([]int, len(c.classCt))
	copy(classCt, c.classCt)
	return classCt
}

func (c *giniValuer) init(inx []int) {
	c.inx = inx

	copy(c.classCt, c.zeroCtr)
	for _, i := range c.inx {
		c.classCt[c.Y[i]]++
	}

	c.iInitial = gini(len(c.inx), c.classCt)

	c.reset()
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

func (c *giniValuer) reset() {
	c.nLeft = 0
	c.nRight = len(c.inx)

	copy(c.classCtL, c.zeroCtr)
	copy(c.classCtR, c.classCt)

	c.posLast = 0
}

func (c *giniValuer) update(pos int) {
	for j := c.posLast; j < pos; j++ {
		yVal := c.Y[c.inx[j]]

		c.nLeft++
		c.classCtL[yVal]++

		c.nRight--
		c.classCtR[yVal]--
	}
	c.posLast = pos
}

func (c *giniValuer) initialVal() float64 {
	return c.iInitial
}

func (c *giniValuer) delta() float64 {
	fracLeft := float64(c.nLeft) / float64(len(c.inx))
	fracRight := float64(c.nRight) / float64(len(c.inx))
	iR := gini(c.nRight, c.classCtR)
	iL := gini(c.nLeft, c.classCtL)

	return c.iInitial - fracLeft*iL - fracRight*iR
}

// variance valuer for regression
type varValuer struct {
	baseValuer
	sL, ssL float64
	sR, ssR float64
	Y       []float64
	mean    float64
}

func newVarValuer(Y []float64) *varValuer {
	return &varValuer{Y: Y}
}

func (c *varValuer) nodeVal() interface{} {
	return c.mean
}

func (c *varValuer) init(inx []int) {
	c.inx = inx
	c.iInitial, c.mean = meanVar(c.Y, c.inx)
	c.reset()
}

func meanVar(Y []float64, inx []int) (float64, float64) {
	var ss, s float64

	for _, i := range inx {
		ss += Y[i] * Y[i]
		s += Y[i]
	}

	mean := s / float64(len(inx))
	v := ss/float64(len(inx)) - mean*mean

	return v, mean
}

func (c *varValuer) reset() {
	c.sL = 0
	c.ssL = 0
	c.sR = 0
	c.ssR = 0
	c.nRight = len(c.inx)
	c.nLeft = 0

	for _, i := range c.inx {
		c.sR += c.Y[i]
		c.ssR += c.Y[i] * c.Y[i]
	}

	c.posLast = 0
}

func (c *varValuer) update(pos int) {
	for j := c.posLast; j < pos; j++ {
		yVal := c.Y[c.inx[j]]

		c.nLeft++
		c.sL += yVal
		c.ssL += yVal * yVal

		c.nRight--
		c.sR -= yVal
		c.ssR -= yVal * yVal
	}
	c.posLast = pos
}

func (c *varValuer) initialVal() float64 {
	return c.iInitial
}

func (c *varValuer) delta() float64 {
	fracLeft := float64(c.nLeft) / float64(len(c.inx))
	fracRight := float64(c.nRight) / float64(len(c.inx))
	rMean := c.sR / float64(c.nRight)
	iR := c.ssR/float64(c.nRight) - rMean*rMean
	lMean := c.sL / float64(c.nLeft)
	iL := c.ssL/float64(c.nLeft) - lMean*lMean

	return c.iInitial - fracLeft*iL - fracRight*iR
}
