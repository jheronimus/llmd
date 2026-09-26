package llm_test

import (
	"context"
	"fmt"
	"log"

	"github.com/jheronimus/llm"
)

type TriageRequest struct {
	IsUrgent bool   `llm:"bool,prompt:Does this request need immediate assistance?"`
	Category string `llm:"choice,technical|billing|sales|general,prompt:What is the issue category?"`
}

func ExampleDecide() {
	ctx := context.Background()

	res, err := llm.Decide[TriageRequest](ctx, "My production database is down and customers are affected!")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Urgent: %v, Category: %s\n", res.IsUrgent, res.Category)
}

type UserProfile struct {
	Name  string `json:"name"`
	Role  string `json:"role"`
	Years int    `json:"years_experience"`
}

func ExampleExtract() {
	ctx := context.Background()

	profile, err := llm.Extract[UserProfile](ctx, "Extract user: Alice is a Principal Engineer with 12 years of experience.")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Name: %s, Role: %s, Experience: %d\n", profile.Name, profile.Role, profile.Years)
}

func ExampleGenerate() {
	ctx := context.Background()

	resp, err := llm.Generate(ctx, llm.Request{
		Prompt:      "Say hello in French.",
		Temperature: 0.1,
		MaxTokens:   16,
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(resp.Text)
}
