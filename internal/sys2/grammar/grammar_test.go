package grammar_test

import (
	"strings"
	"testing"

	"github.com/jheronimus/llm/internal/sys2/grammar"
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
