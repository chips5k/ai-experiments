package main

import (
	"fmt"
	"log/slog"
	"math"
	"os"
)

func setLogLevel(l string) {
	if l == "" {
		l = "ERROR"
	}
	var logLevel slog.Level
	switch l {
	case "DEBUG":
		logLevel = slog.LevelDebug
	case "INFO":
		logLevel = slog.LevelInfo
	case "WARN", "WARNING":
		logLevel = slog.LevelWarn
	case "ERROR":
		logLevel = slog.LevelError
	default:
		logLevel = slog.LevelError
	}
	slog.SetLogLoggerLevel(logLevel)
}

func main() {

	setLogLevel(os.Getenv("LOG_LEVEL"))

	fruitExample()
}

func fruitExample() {

	dimensions := []string{
		"crunchyness",
		"greenness",
		"circularness",
		"veinyness",
		"bigness",
		"redness",
		"bumpyness",
	}

	embeddings := map[string][]float64{
		"red apple":   {1, -1, 1, -1, 0.5, 1, -1},
		"pear":        {1, 1, -0.5, -1, 0.5, -1, -0.5},
		"green apple": {1, 1, 1, -1, 1, -1, -1},
		"banana":      {-1, 0, -1, 1, 1, 0, -1},
		"orange":      {-0.5, -1, 1, 0.3, 1, 0.25, 0.5},
		"strawberry":  {0.8, -1, -0.6, 0, 0.15, 1, 0.5},
		"mandarin":    {-0.5, -1, 1, 0.3, 0.25, 0.25, 0.25},
	}

	// precomputed weights that identifies an apple
	// threshold for id
	bias := float64(-0.4)

	f1 := "orange"
	f2 := "mandarin"
	p, err := determineProbability(f1, f2, dimensions, embeddings, bias)

	if err != nil {
		slog.Error(fmt.Sprintf("Failed to determine probability: %s\n", err))
		fmt.Printf("Unable to determine probability due to error, consult logs for further details")
		return
	} else {
		slog.Info(fmt.Sprintf("%.2f%% chance %s is an %s\n\n", p*100, f1, f2))
	}

	s, err := determineSimilarity(f1, f2, dimensions, embeddings)
	if err != nil {
		slog.Error(fmt.Sprintf("Failed to determine similarity: %s\n", err))
		fmt.Printf("Unable to determine probability due to error, consult logs for further details")
		return
	} else {
		slog.Info(fmt.Sprintf("%s and %s are %.2f%% similar\n", f1, f2, s*100))
	}

	matchCutOff := 0.99
	if p > matchCutOff && s > matchCutOff {
		fmt.Printf("%s is a %s\n", f1, f2)
	} else {
		fmt.Printf("%s is not a %s\n", f1, f2)
	}
	similarCutOff := 0.7
	if p > similarCutOff && s > similarCutOff {
		fmt.Printf("%s is similar to %s\n", f1, f2)
	} else {
		fmt.Printf("%s is not similar to %s\n", f1, f2)
	}

}

func determineProbability(wordA string, wordB string, dimensions []string, embeddings map[string][]float64, bias float64) (probability float64, err error) {

	vecA, vecAFound := embeddings[wordA]
	if !vecAFound {
		return 0, fmt.Errorf("%s not found", wordA)
	}
	vecB, vecBFound := embeddings[wordB]
	if !vecBFound {
		return 0, fmt.Errorf("%s not found", wordB)
	}

	dims := len(dimensions)
	if len(vecA) != dims || len(vecB) != dims {
		return 0.0, fmt.Errorf("Mismatched number of dimensions")
	}

	slog.Debug(fmt.Sprintf("Word A: %s, Word B: %s", wordA, wordB))
	dot := float64(0)
	for i := 0; i < len(vecB); i++ {
		slog.Debug(fmt.Sprintf("  > %s dot = %f * %f = %f\n", dimensions[i], vecA[i], vecB[i], vecA[i]*vecB[i]))
		dot += vecA[i] * vecB[i]
	}

	score := dot + bias

	//sigmoid probability function
	e := math.Exp(-score)
	p := 1 / (1 + e)

	slog.Debug(fmt.Sprintf("  > dot: %f, Score: %f, e: %f, p: %f\n", dot, score, e, p))
	return p, nil

}

func determineSimilarity(wordA string, wordB string, dimensions []string, embeddings map[string][]float64) (similiarty float64, err error) {

	vecA, vecAFound := embeddings[wordA]
	if !vecAFound {
		return 0, fmt.Errorf("%s not found", wordA)
	}
	vecB, vecBFound := embeddings[wordB]
	if !vecBFound {
		return 0, fmt.Errorf("%s not found", wordB)
	}

	dims := len(dimensions)
	if len(vecA) != dims || len(vecB) != dims {
		return 0.0, fmt.Errorf("Mismatched number of dimensions")
	}

	slog.Debug(fmt.Sprintf("Word A: %s, Word B: %s", wordA, wordB))
	dot := float64(0)
	vecASums := float64(0)
	vecBSums := float64(0)
	for i := 0; i < dims; i++ {
		slog.Debug(fmt.Sprintf("  > %s dot = %f * %f = %f\n", dimensions[i], vecA[i], vecB[i], vecA[i]*vecB[i]))
		dot += vecA[i] * vecB[i]
		vecASums += (vecA[i] * vecA[i])
		vecBSums += (vecB[i] * vecB[i])
	}

	// Later for direct comparison
	vecAMag := math.Sqrt(vecASums)
	vecBMag := math.Sqrt(vecBSums)

	//cosign similarity
	sim := dot / (vecAMag * vecBMag)
	slog.Debug(fmt.Sprintf("Dot Product: %f, VecA Mag: %f, VecB Mag: %f", dot, vecAMag, vecBMag))
	slog.Debug(fmt.Sprintf("Cosine Similarity: %f", sim))
	return (sim + 1) / 2, nil
}
