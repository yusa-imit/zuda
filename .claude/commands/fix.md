Fix a bug in the zuda library.

Bug description: $ARGUMENTS

Workflow:
1. **Reproduce**: Understand the bug. If a test reproduces it, run it. Check if invariant validation (`validate()`) catches it.
2. **Locate**: Use Grep/Glob to find relevant code. Read the source files.
3. **Analyze**: Identify root cause. Check `.claude/memory/debugging.md` for similar past issues. Verify Big-O contracts are not violated.
4. **Fix**: Apply the minimal fix needed. Don't refactor unrelated code. Ensure invariants are maintained.
5. **Test**: Ensure existing tests still pass. Add a regression test for this bug.
6. **Validate**: Run `validate()` on the data structure with various inputs to confirm invariant holds.
7. **Verify**: Run `zig build test` to confirm everything passes.
8. **Memory**: Record the bug and fix in `.claude/memory/debugging.md`.
9. **Report**: Summarize the root cause, the fix, and the regression test added.
