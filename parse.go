package main

import (
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"strconv"
)

type parsedInput struct {
	isRegression bool
	X            [][]float64
	YClf         []string  // will be nil when isRegression = true
	YReg         []float64 // will be nil when isRegression = false
	VarNames     []string
}

// parse csv file, detect if first row is header/has var names,
// returns X, Y, varNames, error
func parseCSV(r io.Reader) (*parsedInput, error) {
	reader := csv.NewReader(r)

	// isRegression=true, parse as regression until we hit
	// errors parsing floats, then set flag
	p := &parsedInput{isRegression: true}

	// grab first fow
	row, err := reader.Read()
	if err != nil {
		return p, err
	}

	// check if it's a header row
	varNames, err := parseHeader(row)
	if err == nil {
		p.VarNames = varNames
	} else {
		// use X1, X2,...Xn for var names
		for i := range row[1:] {
			p.VarNames = append(p.VarNames, fmt.Sprintf("X%d", i+1))
		}

		// parse row
		err = p.ParseRow(row)
		if err != nil {
			return p, err
		}
	}

	// keep reading rows until EOF
	for {
		row, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return p, err
		}

		err = p.ParseRow(row)
		if err != nil {
			return p, err
		}
	}

	// drop the y vals we aren't using
	if p.isRegression {
		p.YClf = nil
	} else {
		p.YReg = nil
	}

	return p, err
}

func (p *parsedInput) ParseRow(row []string) error {
	xi, err := parseFeatureVals(row)
	if err != nil {
		return err
	}
	p.X = append(p.X, xi)

	// parse as regression and classification until we encounter errors
	// parsing floats
	if p.isRegression {
		yi, err := strconv.ParseFloat(row[0], 64)
		if err != nil {
			p.isRegression = false
		}
		p.YReg = append(p.YReg, yi)

	}
	p.YClf = append(p.YClf, row[0])

	return nil
}

func parseFeatureVals(row []string) ([]float64, error) {
	var xi []float64
	if len(row) < 1 {
		return xi, errors.New("row only has one column")
	}
	for _, val := range row[1:] {
		fv, err := strconv.ParseFloat(val, 64)
		if err != nil {
			return xi, err
		}
		xi = append(xi, fv)
	}
	return xi, nil
}

func parseHeader(row []string) ([]string, error) {
	colNames := []string{}

	// we only accept numeric input values, so we can consider the first row
	// as a header row if one or more of the values isn't a number
	if len(row) > 1 {
		for _, val := range row[1:] {
			_, err := strconv.ParseFloat(val, 64)
			if err == nil {
				return colNames, errors.New("not a header row")
			}

			colNames = append(colNames, val)
		}
	}

	return colNames, nil
}
