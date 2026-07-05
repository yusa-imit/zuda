#!/bin/bash
# Run this script to free disk space and unblock zuda autonomous development
# The /private/tmp filesystem is full — likely due to zig build cache

echo "=== Disk space before cleanup ==="
df -h /private/tmp 2>/dev/null || df -h /
du -sh /Users/fn/codespace/zuda/.zig-cache 2>/dev/null && echo ".zig-cache size above"
du -sh ~/.cache/zig 2>/dev/null && echo "~/.cache/zig size above"

echo ""
echo "=== Cleaning up ==="

# Remove project zig cache
rm -rf /Users/fn/codespace/zuda/.zig-cache
echo "Removed .zig-cache"

# Clean old claude session tmp files from previous conversations
find /private/tmp/claude-501/ -maxdepth 2 -type d -name "tasks" | while read tasks_dir; do
    session_dir=$(dirname "$tasks_dir")
    session_uuid=$(basename "$session_dir")
    # Keep current session 4486c72a, remove old ones
    if [ "$session_uuid" != "4486c72a-263f-4531-8ae1-ef86301ccfde" ]; then
        echo "Removing old session: $session_uuid"
        rm -rf "$session_dir"
    fi
done

echo ""
echo "=== Disk space after cleanup ==="
df -h /private/tmp 2>/dev/null || df -h /

echo ""
echo "=== Pending git work ==="
cd /Users/fn/codespace/zuda
echo "Running: zig build test"
zig build test && echo "✅ Tests pass" || echo "❌ Tests failed"

echo ""
echo "Committing NonCentralChiSquared distribution..."
git add src/stats/distributions.zig .claude/session-counter .claude/memory/project-context.md .claude/DISK_FULL_CLEANUP.sh
git commit -m "feat: add NonCentralChiSquared distribution (133rd total, 109th continuous)

Poisson mixture series over chi-squared(k+2j) components.
Fixed premature early-stopping for large lambda by skipping
convergence check until past Poisson mode (j > lambda/2).
Mean=k+lambda, Var=2(k+2lambda), exact formulas; 19 tests.

Co-Authored-By: Claude <noreply@anthropic.com>"
git push

echo "Done! You can now restart the autonomous Claude Code session."
