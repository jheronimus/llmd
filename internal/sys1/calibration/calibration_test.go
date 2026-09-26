package calibration

import (
	"math"
	"testing"
)

func TestClampTemperature(t *testing.T) {
	tests := []struct {
		input    float64
		expected float64
	}{
		{0.1, TempMin},
		{1.5, 1.5},
		{10.0, TempMax},
		{math.NaN(), 1.0},
		{math.Inf(1), 1.0},
	}
	for _, tt := range tests {
		got := ClampTemperature(tt.input)
		if math.Abs(got-tt.expected) > 1e-6 {
			t.Errorf("ClampTemperature(%v) = %v, want %v", tt.input, got, tt.expected)
		}
	}
}

func TestTempBucket(t *testing.T) {
	tests := []struct {
		qtype string
		k     int
		want  string
	}{
		{"choice", 2, "choice:2"},
		{"choice", 4, "choice:3-5"},
		{"choice", 8, "choice:6-10"},
		{"choice", 15, "choice:11+"},
		{"score", 3, "score:3-5"},
		{"noul", 2, "noul:2"},
	}
	for _, tt := range tests {
		got := TempBucket(tt.qtype, tt.k)
		if got != tt.want {
			t.Errorf("TempBucket(%v, %d) = %q, want %q", tt.qtype, tt.k, got, tt.want)
		}
	}
}

func TestSoftmax(t *testing.T) {
	logits := []float32{1.0, 2.0, 3.0}
	probs := Softmax(logits, 1.0)
	if len(probs) != 3 {
		t.Fatalf("expected 3 probabilities, got %d", len(probs))
	}

	sum := 0.0
	for _, p := range probs {
		sum += p
	}
	if math.Abs(sum-1.0) > 1e-5 {
		t.Errorf("softmax sum = %v, want 1.0", sum)
	}

	if probs[2] <= probs[1] || probs[1] <= probs[0] {
		t.Errorf("expected strictly increasing probabilities, got %v", probs)
	}
}

func TestConfidenceFromProbs(t *testing.T) {
	uniform := []float64{0.5, 0.5}
	confUniform := ConfidenceFromProbs(uniform, 2)
	if confUniform > 0.05 {
		t.Errorf("uniform distribution confidence = %v, want ~0.0", confUniform)
	}

	peaked := []float64{0.999, 0.001}
	confPeaked := ConfidenceFromProbs(peaked, 2)
	if confPeaked < 0.90 {
		t.Errorf("peaked distribution confidence = %v, want ~1.0", confPeaked)
	}
}
