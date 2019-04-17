package main

import (
  "github.com/oelmekki/matrix"
  "fmt"
  "math/rand"
  "time"
)

type Network struct {
	inputs        int
	hiddens       int
	outputs       int
	inputWeights matrix.Matrix
	hiddenWeights matrix.Matrix
}

func CreateNetwork(input, hidden, output int) (net Network) {
	net = Network{
		inputs:       input,
		hiddens:      hidden,
		outputs:      output,
	}

	net.inputWeights = matrix.RandomMatrix( hidden, input )

	net.hiddenWeights = matrix.RandomMatrix( output, hidden )

	return
}

func (net Network) Predict(inputData []float64) matrix.Matrix {
	// forward propagation
	inputs := matrix.GenerateMatrix( len(inputData), 1 )
  for i:=0; i<len(inputData); i++{
  inputs.SetAt( i,0, inputData[i])
  }

  hiddenInputs,_ := net.inputWeights.DotProduct(inputs)

  hiddenOutputs,_ := hiddenInputs.Sigmoid()

  finalInputs,_ := net.hiddenWeights.DotProduct(hiddenOutputs)

  finalOutputs,_ := finalInputs.Sigmoid()
	return finalOutputs
}





func main(){
  rand.Seed(time.Now().UTC().UnixNano())


  net := CreateNetwork(2, 4, 1)
  results := net.Predict([]float64{0,0})
  fmt.Println(results.String())
  fmt.Println("Network Working")
  InitializeTrainer(&net,500, 1, 20, 6, 5)
  fmt.Println("Trainer Initialized")

  net.Train(10000, true)

  results = net.Predict([]float64{0,0})
  fmt.Println(results.String())
  results = net.Predict([]float64{1,0})
  fmt.Println(results.String())
  results = net.Predict([]float64{0,1})
  fmt.Println(results.String())
  results = net.Predict([]float64{1,1})
  fmt.Println(results.String())



}

// InitializeTrainer(&net,200, 1, 20, 6, 5)
// net.Train(1000000, true)

// 999999 7.4307028212312165
//
// {               0.7905653521331373              }
//
//
// {               0.12381521093720718             }
//
//
// {               0.10198155509763356             }
//
//
// {               0.8659342351329194              }
