// Command rules-check validates rule bundles against this library's
// parameter table, and prints that table for tools that generate rules.
//
//	rules-check [-env ENV] FILE...   validate bundles; with -env, each must
//	                                 name ENV, or "any" (the built-in rules)
//
// Flags come before the files.
//	rules-check -catalog             print the parameter table as JSON
//
// It exits 1 if any bundle fails to load: the same check a Store makes, so a
// bundle that passes here loads in the router and the broker built from the
// same library version.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	paramnorm "github.com/0gfoundation/0g-param-normalization"
)

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

func run(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("rules-check", flag.ContinueOnError)
	fs.SetOutput(stderr)
	env := fs.String("env", "", "require each bundle to be for this environment (or \"any\")")
	catalog := fs.Bool("catalog", false, "print the parameter table as JSON and exit")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *catalog {
		if fs.NArg() > 0 {
			fmt.Fprintln(stderr, "rules-check: -catalog takes no files")
			return 2
		}
		params, structural, containers := paramnorm.Catalog()
		out := map[string]any{"params": params, "structural": structural, "containers": containers}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(out); err != nil {
			fmt.Fprintln(stderr, "rules-check:", err)
			return 1
		}
		return 0
	}
	if fs.NArg() == 0 {
		fmt.Fprintln(stderr, "usage: rules-check [-env ENV] FILE... | rules-check -catalog")
		return 2
	}
	failed := false
	for _, path := range fs.Args() {
		data, err := os.ReadFile(path)
		if err != nil {
			fmt.Fprintf(stderr, "%s: %v\n", path, err)
			failed = true
			continue
		}
		rs, err := paramnorm.ParseRules(data)
		if err != nil {
			fmt.Fprintf(stderr, "%s: %v\n", path, err)
			failed = true
			continue
		}
		if *env != "" && rs.Env() != paramnorm.EnvAny && rs.Env() != strings.ToLower(*env) { // rs.Env() is lower-cased by ParseRules
			fmt.Fprintf(stderr, "%s: bundle is for %q, want %q\n", path, rs.Env(), *env)
			failed = true
			continue
		}
		fmt.Fprintf(stdout, "%s: ok version %d env %s sha256 %s\n", path, rs.Version(), rs.Env(), rs.Hash())
	}
	if failed {
		return 1
	}
	return 0
}
