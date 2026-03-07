Perform a code review on current changes.

Steps:
1. Run `git diff` to see unstaged changes
2. Run `git diff --cached` to see staged changes
3. If no changes, check `git diff HEAD~1` for the last commit
4. For each changed file:
   a. Read the full file for context
   b. Review changes against the checklist:
      - Correctness: Do invariants hold after every operation?
      - Complexity: Do Big-O annotations match implementation?
      - Library safety: Allocator-first, comptime config, no panic, no debug.print?
      - API quality: Iterator protocol, validate() method, standard container pattern?
      - Tests: Are there tests for new/changed functionality?
      - Performance: Efficient memory layout, no unnecessary allocations?
5. Report findings as CRITICAL / WARNING / SUGGESTION

Context: $ARGUMENTS (optional description of what to focus on)
