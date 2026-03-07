Show the current status of the zuda project.

Gather and display:

1. **Git Status**:
   - Current branch
   - Uncommitted changes count
   - Last commit message and date
   - Ahead/behind remote

2. **Build Status**:
   - Run `zig build` and report success/failure

3. **Test Status**:
   - Run `zig build test` and report pass/fail count

4. **Project Progress**:
   - Read `CLAUDE.md` phase checklists
   - Report completion percentage per phase
   - List next unchecked items

5. **Data Structure Coverage**:
   - Count implemented containers vs PRD target
   - Count implemented algorithms vs PRD target

6. **Memory Summary**:
   - Read `.claude/memory/` files
   - Report key recent decisions and known issues

Format output as a clear dashboard.
