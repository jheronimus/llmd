package grammar_test

import (
	"strings"
	"testing"

	"github.com/jheronimus/llm/internal/grammar"
)

func TestGenericJSON(t *testing.T) {
	g := grammar.GenericJSON()
	if !strings.Contains(g, "root   ::= object | array") {
		t.Fatalf("expected generic JSON grammar to define root object/array, got:\n%s", g)
	}
	if !strings.Contains(g, "ws ::=") {
		t.Fatalf("expected whitespace rule in grammar")
	}
}

func TestFromJSONSchema(t *testing.T) {
	tests := []struct {
		name       string
		schema     string
		wantRule   string
		wantErr    bool
		fallbackOK bool
	}{
		{
			name:       "empty schema returns generic",
			schema:     "",
			wantRule:   "root   ::= object | array",
			fallbackOK: true,
		},
		{
			name: "job ranking schema",
			schema: `{
				"type": "object",
				"properties": {
					"score": {"type": "integer"},
					"fit_reason": {"type": "string"},
					"match": {"type": "boolean"}
				},
				"required": ["score", "fit_reason"]
			}`,
			wantRule: `prop-score ::= number`,
		},
		{
			name: "enum property",
			schema: `{
				"type": "object",
				"properties": {
					"verdict": {
						"type": "string",
						"enum": ["accept", "reject", "review"]
					}
				}
			}`,
			wantRule: `prop-verdict ::= ("\"accept\"" | "\"reject\"" | "\"review\"") ws`,
		},
		{
			name:       "invalid json falls back with error",
			schema:     "{not a valid json}",
			wantRule:   "root   ::= object | array",
			wantErr:    true,
			fallbackOK: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			res, err := grammar.FromJSONSchema(tc.schema)
			if (err != nil) != tc.wantErr {
				t.Fatalf("FromJSONSchema() error = %v, wantErr = %v", err, tc.wantErr)
			}
			if !strings.Contains(res, tc.wantRule) {
				t.Fatalf("grammar does not contain expected rule %q:\n%s", tc.wantRule, res)
			}
		})
	}
}

func TestFromType(t *testing.T) {
	type Ranking struct {
		Score     int      `json:"score"`
		FitReason string   `json:"fit_reason"`
		Approved  bool     `json:"approved"`
		Tags      []string `json:"tags"`
		Decision  string   `json:"decision" enum:"yes,no,maybe"`
		Ignored   string   `json:"-"`
	}

	g, err := grammar.FromType(&Ranking{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expectedRules := []string{
		`"\"score\": " ws prop-score`,
		`"\"fit_reason\": " ws prop-fit-reason`,
		`"\"approved\": " ws prop-approved`,
		`"\"tags\": " ws prop-tags`,
		`"\"decision\": " ws prop-decision`,
		`prop-score ::= number`,
		`prop-fit-reason ::= string`,
		`prop-approved ::= boolean`,
		`prop-tags ::= "[" ws (string ("," ws string)*)? "]" ws`,
		`prop-decision ::= ("\"yes\"" | "\"no\"" | "\"maybe\"") ws`,
	}

	for _, rule := range expectedRules {
		if !strings.Contains(g, rule) {
			t.Errorf("expected grammar to contain %q, but got:\n%s", rule, g)
		}
	}

	if strings.Contains(g, "Ignored") {
		t.Errorf("expected json:\"-\" field to be excluded, but found in grammar:\n%s", g)
	}

	// Non-struct fallback
	nonStruct, err := grammar.FromType("plain string")
	if err != nil {
		t.Fatalf("unexpected error on non-struct: %v", err)
	}
	if !strings.Contains(nonStruct, "root   ::= object | array") {
		t.Fatalf("expected generic JSON fallback for non-struct, got:\n%s", nonStruct)
	}
}
