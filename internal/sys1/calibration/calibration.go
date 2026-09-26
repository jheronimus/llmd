package calibration

import (
	"fmt"
	"math"
)

const (
	TempMin = 0.5
	TempMax = 5.0
)

// ClampTemperature constrains a temperature value to [TempMin, TempMax].
func ClampTemperature(t float64) float64 {
	if math.IsNaN(t) || math.IsInf(t, 0) || t <= 0 {
		return 1.0
	}
	return min(TempMax, max(TempMin, t))
}

// TempBucket formats the bucket key based on question type and number of options.
func TempBucket(qtypeName string, k int) string {
	var size string
	switch {
	case k <= 2:
		size = "2"
	case k <= 5:
		size = "3-5"
	case k <= 10:
		size = "6-10"
	default:
		size = "11+"
	}
	return fmt.Sprintf("%s:%s", qtypeName, size)
}

// TemperatureScale resolves temperature for a specific question type and option count.
func TemperatureScale(tempByOptions map[string]float64, defaultTemps []float64, qtypeName string, qtype int, k int) float64 {
	bucket := TempBucket(qtypeName, k)
	if tempByOptions != nil {
		if t, ok := tempByOptions[bucket]; ok {
			return ClampTemperature(t)
		}
	}
	if len(defaultTemps) > qtype {
		return ClampTemperature(defaultTemps[qtype])
	}
	return 1.0
}

// Softmax calculates temperature-scaled softmax over logits.
func Softmax(logits []float32, temp float64) []float64 {
	if len(logits) == 0 {
		return nil
	}
	if temp <= 0 {
		temp = 1.0
	}

	maxLogit := float64(logits[0]) / temp
	for _, l := range logits[1:] {
		scaled := float64(l) / temp
		if scaled > maxLogit {
			maxLogit = scaled
		}
	}

	expSum := 0.0
	res := make([]float64, len(logits))
	for i, l := range logits {
		val := math.Exp((float64(l) / temp) - maxLogit)
		res[i] = val
		expSum += val
	}

	if expSum > 0 {
		for i := range res {
			res[i] /= expSum
		}
	}
	return res
}

// ConfidenceFromProbs computes normalized Shannon entropy confidence: 1 - H(p)/log(k).
func ConfidenceFromProbs(p []float64, k int) float64 {
	if k < 2 || len(p) < 2 {
		return 1.0
	}
	if len(p) > k {
		p = p[:k]
	}

	ent := 0.0
	for _, prob := range p {
		clipped := math.Max(prob, 1e-12)
		ent += -(prob * math.Log(clipped))
	}

	logK := math.Log(float64(k))
	if logK <= 0 {
		return 1.0
	}

	conf := 1.0 - (ent / logK)
	return math.Round(min(1.0, max(0.0, conf))*10000) / 10000
}
