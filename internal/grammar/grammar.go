// Package grammar generates GBNF (GGML BNF) grammars to enforce token-level
// constraints on LLM generation in llama.cpp.
package grammar

import (
	"encoding/json"
	"fmt"
	"reflect"
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

// FromType generates a GBNF grammar enforcing the structure of the provided Go value (struct or pointer to struct).
// If target is not a struct or has unsupported/nested complex types, it falls back to GenericJSON.
func FromType(target any) (string, error) {
	if target == nil {
		return GenericJSON(), nil
	}

	t := reflect.TypeOf(target)
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}

	if t.Kind() != reflect.Struct {
		return GenericJSON(), nil
	}

	schema := schemaDef{
		Type:       "object",
		Properties: make(map[string]schemaDef),
	}

	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		if !f.IsExported() {
			continue
		}

		jsonTag := f.Tag.Get("json")
		if jsonTag == "-" {
			continue
		}

		name := f.Name
		if jsonTag != "" {
			parts := strings.Split(jsonTag, ",")
			if parts[0] != "" {
				name = parts[0]
			}
		}

		ft := f.Type
		for ft.Kind() == reflect.Pointer {
			ft = ft.Elem()
		}

		var prop schemaDef
		enumTag := f.Tag.Get("enum")
		if enumTag != "" {
			parts := strings.Split(enumTag, ",")
			var vals []string
			for _, p := range parts {
				p = strings.TrimSpace(p)
				if p != "" {
					vals = append(vals, p)
				}
			}
			if len(vals) > 0 {
				prop.Type = "string"
				prop.Enum = vals
				schema.Properties[name] = prop
				continue
			}
		}

		switch ft.Kind() {
		case reflect.String:
			prop.Type = "string"
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
			reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
			prop.Type = "integer"
		case reflect.Float32, reflect.Float64:
			prop.Type = "number"
		case reflect.Bool:
			prop.Type = "boolean"
		case reflect.Slice, reflect.Array:
			prop.Type = "array"
			elemKind := ft.Elem().Kind()
			switch elemKind {
			case reflect.String:
				prop.Items = &schemaDef{Type: "string"}
			case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
				reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
				prop.Items = &schemaDef{Type: "integer"}
			case reflect.Float32, reflect.Float64:
				prop.Items = &schemaDef{Type: "number"}
			case reflect.Bool:
				prop.Items = &schemaDef{Type: "boolean"}
			default:
				return GenericJSON(), nil
			}
		default:
			return GenericJSON(), nil
		}

		schema.Properties[name] = prop
	}

	if len(schema.Properties) == 0 {
		return GenericJSON(), nil
	}

	return compileSchema(schema)
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

	return compileSchema(schema)
}

func compileSchema(schema schemaDef) (string, error) {
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
