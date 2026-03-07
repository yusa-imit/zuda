# zuda Decision Log

## Decision: Project Scope — Complement std, Don't Replace
- **Date**: 2026-03-07
- **Context**: Need to decide relationship with Zig standard library containers
- **Decision**: zuda complements std. It does not reimplement ArrayList, HashMap, etc. It offers *alternative* structures with different trade-offs.
- **Rationale**: Avoids ecosystem fragmentation. Users import zuda for structures std doesn't provide.
- **Consequences**: Must document when to use std vs zuda for each category.

## Decision: Module-per-File Organization
- **Date**: 2026-03-07
- **Context**: How to organize 100+ data structures
- **Decision**: One data structure per file, grouped in category directories (containers/trees/, algorithms/sorting/, etc.)
- **Rationale**: Easy navigation, clear ownership, manageable file sizes (< 800 lines).
- **Consequences**: Root module must re-export all public types. More files to manage.

## Decision: Zig 0.15.2 as Minimum Version
- **Date**: 2026-03-07
- **Context**: Which Zig version to target
- **Decision**: Minimum Zig 0.15.2
- **Rationale**: Latest stable. Use modern APIs (unmanaged ArrayList, etc.)
- **Consequences**: Must follow 0.15.x patterns. Cannot use unreleased features.
