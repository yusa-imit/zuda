Implement a data structure or algorithm for the zuda library.

Feature description: $ARGUMENTS

Workflow:
1. **Understand**: Read `docs/PRD.md` and `CLAUDE.md` for context. Read `.claude/memory/` files for past decisions and patterns.
2. **Plan**: Enter plan mode. Identify which files need to be created/modified. Design the approach — consider API surface, comptime parameters, iterator protocol, Big-O contracts. Get user approval.
3. **Implement**: Write the code following the container template from CLAUDE.md (allocator-first, comptime config, iterator, validate).
4. **Test**: Write comprehensive tests — lifecycle, operations, invariants, edge cases, memory safety. Run `zig build test`.
5. **Review**: Self-review against library safety checklist and complexity contracts.
6. **Memory**: Update `.claude/memory/` with any architectural decisions or patterns discovered.
7. **Report**: Summarize what was implemented, files changed, tests added, Big-O characteristics.

For complex features (3+ files), consider spawning a team:
- Use `zig-developer` agent for implementation
- Use `test-writer` agent for tests
- Use `code-reviewer` agent for review
- Use `architect` agent for design decisions
