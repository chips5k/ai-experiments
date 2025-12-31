package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strings"
)

type Neuron struct {
	Weights  []float64 `json:"Weights"`
	Bias     float64   `json:"Bias"`
	Output   float64   `json:"Output"`
	Gradient float64   `json:"Gradient"`
}

type Network struct {
	Rand        *rand.Rand
	HiddenLayer []*Neuron `json:"HiddenLayer"`
	OutputLayer []*Neuron `json:"OutputLayer"`
}

type TrainingRecord struct {
	Label  string    `json:"Label"`
	Input  []int     `json:"Input"`
	Target []float64 `json:"Target"`
}

// Todo understand this
func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// TODO understand this
func sigmoidDerivative(x float64) float64 {
	return x * (1.0 - x)
}

// todo support prepopulating weights
func (network *Network) Initialize(inputLayerSize int, hiddenLayerSize int, outputLayerSize int) {

	network.HiddenLayer = make([]*Neuron, hiddenLayerSize)
	network.OutputLayer = make([]*Neuron, outputLayerSize)

	for i := 0; i < len(network.HiddenLayer); i++ {

		network.HiddenLayer[i] = &Neuron{
			Weights:  initializeWeights(inputLayerSize, network.Rand),
			Bias:     network.Rand.Float64(),
			Output:   0.0,
			Gradient: 0.0,
		}

	}

	for i := 0; i < len(network.OutputLayer); i++ {
		network.OutputLayer[i] = &Neuron{
			Weights:  initializeWeights(len(network.HiddenLayer), network.Rand),
			Bias:     network.Rand.Float64(),
			Output:   0.0,
			Gradient: 0.0,
		}
	}

}

func NewNetwork(r *rand.Rand) *Network {
	return &Network{
		Rand: r,
	}
}

func initializeWeights(size int, r *rand.Rand) []float64 {

	weights := make([]float64, size)

	for i := range size {
		weights[i] = r.Float64()
	}

	return weights
}

func (network *Network) Predict(input []int) []float64 {

	for _, n := range network.HiddenLayer {
		sum := float64(0)
		for i, v := range input {
			sum += float64(v) * n.Weights[i]
		}
		n.Output = sigmoid(sum + n.Bias)
	}

	o := make([]float64, len(network.OutputLayer))
	for i, n := range network.OutputLayer {
		sum := float64(0)
		for j, v := range network.HiddenLayer {
			sum += v.Output * n.Weights[j]
		}
		n.Output = sigmoid(sum + n.Bias)
		o[i] = n.Output
	}

	return o
}

func (network *Network) Export(path string) error {

	f, err := os.Create(path)
	if err != nil {
		return err
	}

	defer f.Close()

	encoder := json.NewEncoder(f)

	encoder.SetIndent("", " ")

	err = encoder.Encode(network)

	if err != nil {
		return err
	}

	return nil

}

func (network *Network) Import(path string) error {

	f, err := os.Open(path)
	if err != nil {
		return err
	}

	defer f.Close()

	decoder := json.NewDecoder(f)

	err = decoder.Decode(network)

	if err != nil {
		return err
	}
	return nil
}

func findMax(output []float64) (index int, value float64) {

	maxValue := float64(0.0)
	maxIndex := -1
	for i, v := range output {
		if v > maxValue {
			maxValue = v
			maxIndex = i
		}
	}

	return maxIndex, maxValue

}

func (network *Network) Train(input []int, target []float64, learningRate float64) {

	network.Predict(input)

	// Calc error gradients for output layer
	for i, n := range network.OutputLayer {
		errAmt := target[i] - n.Output
		n.Gradient = errAmt * sigmoidDerivative(n.Output)
	}

	//Calc error gradients for hidden layer
	for i, n := range network.HiddenLayer {
		errAmt := float64(0)
		// use the error gradient from the output to shift blame back
		for _, on := range network.OutputLayer {
			errAmt += on.Gradient * on.Weights[i]
		}
		n.Gradient = errAmt * sigmoidDerivative(n.Output)
	}

	// update weights & bias in output layer
	for _, n := range network.OutputLayer {
		for j, on := range network.HiddenLayer {
			n.Weights[j] += n.Gradient * on.Output * learningRate
		}
		n.Bias += n.Gradient * learningRate
	}

	// update weights & bias in hidden layer
	for _, n := range network.HiddenLayer {
		for i, p := range input {
			n.Weights[i] += n.Gradient * float64(p) * learningRate
		}
		n.Bias += n.Gradient * learningRate
	}

}

func (network *Network) TrainFromFile(path string, exportPath string, epochs int, sampleInputLabel string, sampleInput []int, resultMap []string) error {

	f, err := os.Open(path)
	if err != nil {
		return err
	}

	defer f.Close()

	decoder := json.NewDecoder(f)

	data := make([]TrainingRecord, 0)

	err = decoder.Decode(&data)

	if err != nil {
		return err
	}

	for i := range epochs {
		for _, tr := range data {
			network.Train(tr.Input, tr.Target, 0.01)
		}

		if i%1000000 == 0 || i == epochs-1 {

			fmt.Printf("Predicting what '%s' is...", sampleInputLabel)
			result := network.Predict(sampleInput)
			maxIndex, _ := findMax(result)

			fmt.Println("Result: ", result)
			fmt.Printf("Closest match: %s\n", resultMap[maxIndex])

			fmt.Printf("Exporting to %s,  %d/%d epochs (%.0f%%)\n\n", exportPath, i, epochs, (float64(i)/float64(epochs))*100.00)
			err := network.Export(exportPath)
			if err != nil {
				return err
			}
		}
	}

	return nil

}

func networkFromScratch(r *rand.Rand, inputLayerSize int, hiddenLayerSize int, outputLayerSize int) *Network {
	network := NewNetwork(r)
	network.Initialize(inputLayerSize, hiddenLayerSize, outputLayerSize)

	return network
}

func networkFromFile(r *rand.Rand, path string) (*Network, error) {
	network := NewNetwork(r)
	err := network.Import(path)

	return network, err
}

func generateAlphaNumResultMap() []string {
	charMap := make([]string, 36)
	for i := range 26 {
		charMap[i] = string(rune(65 + i))
	}

	for i := range 10 {
		charMap[i+26] = string(rune(i + 48))
	}

	return charMap
}

func getShape(shape string) []int {

	switch strings.ToLower(shape) {
	case "square":
		return []int{
			1, 1, 1, 1, 1, 1, 1,
			1, 0, 0, 0, 0, 0, 1,
			1, 0, 0, 0, 0, 0, 1,
			1, 0, 0, 0, 0, 0, 1,
			1, 0, 0, 0, 0, 0, 1,
			1, 0, 0, 0, 0, 0, 1,
			1, 1, 1, 1, 1, 1, 1,
		}

	case "circle":
		return []int{
			0, 0, 1, 1, 1, 0, 0,
			0, 1, 0, 0, 0, 1, 0,
			1, 0, 0, 0, 0, 0, 1,
			1, 0, 0, 0, 0, 0, 1,
			1, 0, 0, 0, 0, 0, 1,
			0, 1, 0, 0, 0, 1, 0,
			0, 0, 1, 1, 1, 0, 0,
		}

	case "triangle":
		return []int{
			0, 0, 0, 1, 0, 0, 0,
			0, 0, 1, 0, 1, 0, 0,
			0, 0, 1, 0, 1, 0, 0,
			0, 1, 0, 0, 0, 1, 0,
			0, 1, 0, 0, 0, 1, 0,
			1, 0, 0, 0, 0, 0, 1,
			1, 1, 1, 1, 1, 1, 1,
		}

	case "cross":
		return []int{
			1, 0, 0, 0, 0, 0, 1,
			0, 1, 0, 0, 0, 1, 0,
			0, 0, 1, 0, 1, 0, 0,
			0, 0, 0, 1, 0, 0, 0,
			0, 0, 1, 0, 1, 0, 0,
			0, 1, 0, 0, 0, 1, 0,
			1, 0, 0, 0, 0, 0, 1,
		}
	default:
		return []int{
			0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0,
		}
	}

}

func main() {

	fModelPath := flag.String("model", "", "Path to your models weights in json format")
	fTrainPath := flag.String("data", "", "Path to your training data in json format")
	fTrain := flag.Bool("train", false, "If training should be run before testing")
	fEpochs := flag.Int("epochs", 10, "Number of epochs to train for")
	fShape := flag.String("shape", "", "Shape to generate and test with")
	flag.Parse()

	if *fModelPath == "" {
		log.Fatalln("-export flag must be provided")
	}

	if *fTrain && *fTrainPath == "" {
		log.Fatalln("-data flag must be provided")
	}

	r := rand.New(rand.NewSource(1))
	resultMap := []string{
		"Square", "Circle", "Triangle", "Cross", "nothing",
	}

	inputLabel := "nothing"
	if *fShape != "" {
		inputLabel = *fShape
	}

	fmt.Printf("Getting %s as input\n", *fShape)
	input := getShape(*fShape)

	fmt.Printf("Loading Network from model: %s\n", *fModelPath)
	network, err := networkFromFile(r, *fModelPath)

	if err != nil {
		fmt.Println("Failed to load model, creating from scratch...")
		network = networkFromScratch(r, len(input), 4, len(resultMap))
	}

	fmt.Printf("Exporting model to %s\n", *fModelPath)
	err = network.Export(*fModelPath)

	if *fTrain {
		fmt.Printf("Training network with data from %s, for %d epochs\n", *fTrainPath, *fEpochs)
		err := network.TrainFromFile(*fTrainPath, *fModelPath, *fEpochs, inputLabel, input, resultMap)

		if err != nil {
			log.Fatalf("Failed to traing model with error: %s\n", err)
		}
	}

	if err != nil {
		log.Fatalf("Failed to export updated model: %s\n", err)
	}

	fmt.Printf("Predicting... what '%s' is\n", inputLabel)
	result := network.Predict(input)
	maxIndex, _ := findMax(result)

	fmt.Println("Result: ", result)

	fmt.Printf("Closest match: %s\n", resultMap[maxIndex])

}
