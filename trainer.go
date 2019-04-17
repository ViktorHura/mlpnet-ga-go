package main

import(
  "github.com/oelmekki/matrix"
  "math/rand"
  "sort"
  "fmt"
)

type Organism struct{
  Fitness float64
  inputWeights matrix.Matrix
	hiddenWeights matrix.Matrix
}

var Population []Organism

var (
  popsize int
  elite int
  cutoff int
  tournamentsize int
  mutationrate float64
  inputs int
  hiddens int
  outputs int
)

func InitializeTrainer(net *Network, pops int, el int, cutf int, tournsize int, mutrate float64) {
  Population = make([]Organism, pops)

  popsize = pops
  elite = el
  cutoff = cutf
  tournamentsize = tournsize
  mutationrate = mutrate

  inputs = net.inputs
  hiddens = net.hiddens
  outputs = net.outputs

}

func (net *Network) Train(maxgen int, reset bool)  {

  if reset == true {
    createInitialPopulation()
  }

  for Generation := 0; Generation < maxgen; Generation++{

    for o:=0; o<popsize; o++{
      CalcFitness(o, net)
    }

    NaturalSelection()

    for o:=0; o<popsize; o++{
      CalcFitness(o, net)
    }
    SortPopulation()
    fmt.Println(Generation, Population[0].Fitness)


  }
  
  SortPopulation()
  net.inputWeights = Population[0].inputWeights
  net.hiddenWeights = Population[0].hiddenWeights


}

func createInitialPopulation() {
  initpop := make([]Organism, popsize)

  for i := 0; i < popsize; i++ {
    o := Organism{
      Fitness: 0,
      inputWeights: matrix.RandomMatrix( hiddens, inputs ),
      hiddenWeights: matrix.RandomMatrix( outputs, hiddens ),
    }
    initpop[i] = o

  }
  Population = initpop


}

func CalcFitness (populationindex int, net *Network) {

  Population[populationindex].Fitness = 0

  net.inputWeights = Population[populationindex].inputWeights
  net.hiddenWeights = Population[populationindex].hiddenWeights

  right := 0


  results := net.Predict([]float64{ 0 , 0 })

      Population[populationindex].Fitness += results.At(0,0)
  if (results.At(0,0) > 0.5){
    right += 1
  }


  results = net.Predict([]float64{ 1 , 0 })

      Population[populationindex].Fitness += 1 - results.At(0,0)

      if (results.At(0,0) < 0.5){
        right += 1
      }


  results = net.Predict([]float64{ 0 , 1 })

      Population[populationindex].Fitness += 1 - results.At(0,0)

      if (results.At(0,0) < 0.5){
        right += 1
      }


  results = net.Predict([]float64{ 1 , 1 })

      Population[populationindex].Fitness += results.At(0,0)
      if (results.At(0,0) > 0.5){
        right += 1
      }

  Population[populationindex].Fitness += float64(right)
}

func NaturalSelection() {
	nextPopulation := make([]Organism, popsize)

  SortPopulation()

	for i := 0; i < popsize; i++ {

    if i < elite {
      nextPopulation[i] = Population[i]

    }else {

      a := TournamentSelection()
      b := TournamentSelection()


		  child := Crossover(a, b)
		  child.Mutate()


		  nextPopulation[i] = child
    }

	}

  Population = nextPopulation
}

func SortPopulation () { //sort by highest fitness
  sort.Slice(Population, func(i, j int) bool {
  return Population[i].Fitness > Population[j].Fitness
  })
}

func TournamentSelection() Organism{

  var tournament []Organism = make( []Organism,0 )

  for n:=0; n < tournamentsize; n++{

    i := rand.Intn(popsize - cutoff)
    tournament = append(tournament, Population[i])

  }

  var bestint int = 0
  var bestfit float64 = tournament[0].Fitness

  for n:=0; n < tournamentsize; n++{

    if tournament[n].Fitness > bestfit {
      bestint = n
      bestfit = tournament[n].Fitness
    }

  }

  return tournament[bestint]

}

func Crossover(p1 Organism, p2 Organism) Organism {

	child := Organism{
		Fitness: 0,
    inputWeights: matrix.GenerateMatrix( hiddens, inputs ),
    hiddenWeights: matrix.GenerateMatrix( outputs, hiddens ),
	}

	mid := rand.Intn(child.inputWeights.Rows() * child.inputWeights.Cols())

  for i := 0 ; i < child.inputWeights.Rows(); i++ {
    for j := 0 ; j < child.inputWeights.Cols(); j++ {

      if i * j < mid {

        child.inputWeights.SetAt(i,j, p1.inputWeights.At(i,j) )

      }else{

        child.inputWeights.SetAt(i,j, p2.inputWeights.At(i,j) )

      }

    }
  }

  mid = rand.Intn(child.hiddenWeights.Rows() * child.hiddenWeights.Cols())

  for i := 0 ; i < child.hiddenWeights.Rows(); i++ {
    for j := 0 ; j < child.hiddenWeights.Cols(); j++ {

      if i * j < mid {
        child.hiddenWeights.SetAt(i,j, p1.hiddenWeights.At(i,j))

      }else{
        child.hiddenWeights.SetAt(i,j, p2.hiddenWeights.At(i,j))

      }

    }
  }


	return child
}

func (o *Organism) Mutate() {

  for i := 0 ; i < o.inputWeights.Rows() ; i++ {
    for j := 0 ; j < o.inputWeights.Cols() ; j++ {

      if rand.Float64() * 100.0 < mutationrate {
        o.inputWeights.SetAt(i,j, rand.NormFloat64())
      }

    }
  }

  for i := 0 ; i < o.hiddenWeights.Rows() ; i++ {
    for j := 0 ; j < o.hiddenWeights.Cols() ; j++ {

      if rand.Float64() * 100.0 < mutationrate {
        o.hiddenWeights.SetAt(i,j, rand.NormFloat64())
      }

    }
  }



}
