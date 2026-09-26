// Package grammar generates GBNF (GGML BNF) grammars to enforce token-level
// constraints on LLM generation in llama.cpp.
package grammar

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"
)

// GenericJSON returns a standard GBNF grammar enforcing valid JSON objects or arrays.
func GenericJSON() string {
	return `root   ::= object | array
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws

ws ::= ([ \t\n\r])*
`
}

// schemaDef represents a subset of JSON Schema definitions.
type schemaDef struct {
	Type        string               `json:"type"`
	Properties  map[string]schemaDef `json:"properties"`
	Required    []string             `json:"required"`
	Items       *schemaDef           `json:"items"`
	Enum        []string             `json:"enum"`
	Description string               `json:"description"`
}

// FromJSONSchema compiles a JSON Schema definition string into a GBNF grammar.
// If the schema is empty or cannot be parsed, it falls back to GenericJSON.
func FromJSONSchema(schemaStr string) (string, error) {
	trimmed := strings.TrimSpace(schemaStr)
	if trimmed == "" {
		return GenericJSON(), nil
	}

	var schema schemaDef
	if err := json.Unmarshal([]byte(trimmed), &schema); err != nil {
		return GenericJSON(), fmt.Errorf("invalid json schema: %w", err)
	}

	if schema.Type != "object" || len(schema.Properties) == 0 {
		return GenericJSON(), nil
	}

	// For complex schemas with nested objects or arrays of objects,
	// fall back to GenericJSON to avoid over-constraining the sampler.
	for _, p := range schema.Properties {
		if p.Type == "object" || (p.Type == "array" && p.Items != nil && (p.Items.Type == "object" || p.Items.Type == "")) {
			return GenericJSON(), nil
		}
	}

	var b strings.Builder
	b.WriteString("root ::= \"{\" ws ")

	// Sort property keys deterministically
	keys := make([]string, 0, len(schema.Properties))
	for k := range schema.Properties {
		keys = append(keys, k)
	}
	slices.Sort(keys)

	for i, k := range keys {
		ruleName := "prop-" + sanitizeIdent(k)

		if i > 0 {
			b.WriteString("\",\" ws ")
		}
		fmt.Fprintf(&b, "\"\\\"%s\\\": \" ws %s ", k, ruleName)
	}

	b.WriteString("\"}\" ws\n\n")

	// Emit sub-rules
	for _, k := range keys {
		prop := schema.Properties[k]
		ruleName := "prop-" + sanitizeIdent(k)
		b.WriteString(formatPropertyRule(ruleName, prop))
		b.WriteString("\n")
	}

	// Base primitives
	b.WriteString(`string ::= "\"" ([^"\\\x7F\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4}))* "\"" ws
number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws
boolean ::= ("true" | "false") ws
null ::= "null" ws
ws ::= ([ \t\n\r])*
`)

	return b.String(), nil
}

func sanitizeIdent(s string) string {
	var out []rune
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') {
			out = append(out, r)
		} else {
			out = append(out, '-')
		}
	}
	return string(out)
}

func formatPropertyRule(name string, def schemaDef) string {
	if len(def.Enum) > 0 {
		var parts []string
		for _, e := range def.Enum {
			parts = append(parts, fmt.Sprintf("\"\\\"%s\\\"\"", e))
		}
		return fmt.Sprintf("%s ::= (%s) ws", name, strings.Join(parts, " | "))
	}

	switch def.Type {
	case "string":
		return fmt.Sprintf("%s ::= string", name)
	case "integer", "number":
		return fmt.Sprintf("%s ::= number", name)
	case "boolean":
		return fmt.Sprintf("%s ::= boolean", name)
	case "array":
		itemRule := "string"
		if def.Items != nil && def.Items.Type != "" {
			switch def.Items.Type {
			case "integer", "number":
				itemRule = "number"
			case "boolean":
				itemRule = "boolean"
			}
		}
		return fmt.Sprintf("%s ::= \"[\" ws (%s (\",\" ws %s)*)? \"]\" ws", name, itemRule, itemRule)
	default:
		return fmt.Sprintf("%s ::= string | number | boolean | null", name)
	}
}
