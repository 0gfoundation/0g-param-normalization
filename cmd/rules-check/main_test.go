package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func write(t *testing.T, name, body string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestValidate(t *testing.T) {
	good := write(t, "good.json", `{"schema":1,"version":3,"env":"staging","families":[],"rows":[{"identity":"a","model":"m","endpoint":"chat","params":{"stop":"rejected"}}]}`)
	bad := write(t, "bad.json", `{"schema":1,"version":3,"env":"staging","families":[],"rows":[{"identity":"a","model":"m","params":{"messages":"rejected"}}]}`)
	var out, errb bytes.Buffer
	if code := run([]string{good}, &out, &errb); code != 0 || !strings.Contains(out.String(), "ok version 3 env staging") {
		t.Fatalf("good: %d %q %q", code, out.String(), errb.String())
	}
	out.Reset()
	errb.Reset()
	if code := run([]string{good, bad}, &out, &errb); code != 1 || !strings.Contains(errb.String(), "bad.json") {
		t.Fatalf("bad: %d %q", code, errb.String())
	}
	errb.Reset()
	if code := run([]string{"-env", "mainnet", good}, &out, &errb); code != 1 || !strings.Contains(errb.String(), `want "mainnet"`) {
		t.Fatalf("env: %d %q", code, errb.String())
	}
	if code := run(nil, &out, &errb); code != 2 {
		t.Fatalf("no files: %d", code)
	}
}

func TestCatalog(t *testing.T) {
	var out, errb bytes.Buffer
	if code := run([]string{"-catalog"}, &out, &errb); code != 0 {
		t.Fatalf("%d %s", code, errb.String())
	}
	var got struct {
		Params map[string]struct {
			Category      string              `json:"category"`
			Classes       []string            `json:"classes"`
			ClassCategory map[string]string   `json:"class_category"`
			Translations  map[string][]string `json:"translations"`
			Limits        bool                `json:"limits"`
		} `json:"params"`
		Structural []string `json:"structural"`
		Containers []string `json:"containers"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if p := got.Params["reasoning_effort"]; p.Category != "translate" || p.ClassCategory["none"] != "reject" || len(p.Translations["none"]) == 0 {
		t.Errorf("reasoning_effort: %+v", p)
	}
	if p := got.Params["max_tokens"]; !p.Limits {
		t.Errorf("max_tokens: %+v", p)
	}
	if p, ok := got.Params["user"]; !ok || p.Category != "passthrough" || len(p.Classes) != 0 {
		t.Errorf("user: %+v %v", p, ok)
	}
	if len(got.Structural) == 0 || len(got.Containers) == 0 {
		t.Errorf("structural %v containers %v", got.Structural, got.Containers)
	}
}
