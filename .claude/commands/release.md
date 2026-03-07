Prepare and execute a release for the zuda library.

Version: $ARGUMENTS (e.g., "v0.1.0")

Workflow:
1. **Pre-flight checks**:
   - Run `zig build test` — all tests must pass (0 failures)
   - Run `git status` — working tree must be clean
   - Cross-compile for all 6 targets to verify builds:
     - `zig build -Dtarget=x86_64-linux-gnu`
     - `zig build -Dtarget=aarch64-linux-gnu`
     - `zig build -Dtarget=x86_64-macos-none`
     - `zig build -Dtarget=aarch64-macos-none`
     - `zig build -Dtarget=x86_64-windows-msvc`
     - `zig build -Dtarget=aarch64-windows-msvc`
   - If ANY check fails, abort release and fix the issue first
2. **Version bump**: Update version in `build.zig.zon`
3. **Changelog**: Generate summary of changes since last tag using `git log --oneline <last-tag>..HEAD`
4. **Commit**: `chore: bump version to <version>`
5. **Tag**: `git tag -a <version> -m "Release <version>"`
6. **Push**: `git push && git push origin <version>`
7. **GitHub Release**: `gh release create <version> --title "<version>" --notes "<changelog>"`
8. **Close resolved issues**: If the release addresses any open GitHub issues, close them:
   - `gh issue close <number> --comment "Resolved in <version>"`
9. **Report**: Summary of release steps completed
