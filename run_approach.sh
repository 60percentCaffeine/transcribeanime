#!/bin/bash
set -e

# Agent selection:
# - Set RALPH_AGENT=claude or RALPH_AGENT=codex to force.
# - Otherwise auto-detect (prefers Claude if available).
resolve_agent() {
    if [[ -n "${RALPH_AGENT:-}" ]]; then
        echo "$RALPH_AGENT"
        return 0
    fi
    if command -v claude >/dev/null 2>&1; then
        echo "claude"
        return 0
    fi
    if command -v codex >/dev/null 2>&1; then
        echo "codex"
        return 0
    fi
    return 1
}

run_agent() {
    local agent="$1"
    local prompt="$2"
    case "$agent" in
        claude)
            claude --dangerously-skip-permissions -p "$prompt"
            ;;
        codex)
            local output_file
            output_file="$(mktemp -t ralph_codex.XXXXXX)"
            codex exec --full-auto --color never -C "$PWD" --output-last-message "$output_file" "$prompt" >/dev/null
            cat "$output_file"
            rm -f "$output_file"
            ;;
        *)
            echo "Unsupported agent: $agent" >&2
            return 1
            ;;
    esac
}

agent=$(resolve_agent) || {
    echo "No supported agent found. Install 'claude' or 'codex', or set RALPH_AGENT." >&2
    exit 1
}

has_untested() {
    grep -q 'TOBETESTED' APPROACHES.md 2>/dev/null
}

prompt=$(cat <<'EOF'
@APPROACHES.md

Test the first untested approach from APPROACHES.md. Run test.sh to check it works and improves the results. Don't be afraid to iterate on it 2-3 times to get it right.

When you're done, update the entry in APPROACHES.md with the results. If the approach was successful (i.e. black box score is improved) add a section that explains the results. If only white box results are improved, the approach is considered not successful. In the section include: what was done, key numbers or findings, any surprises or quality issues to watch for in downstream tasks.

Also if the approach was successful, leave it in the code and commit the results. Commit message should include approach description and test numbers. If the approach was unsuccessful, revert changes but commit APPROACHES.md results.

WORK ON ONLY ONE APPROACH.
If while implementing the task you determine that it is fully complete, output <promise>COMPLETE</promise>
EOF
)

iteration=1
while has_untested; do
    echo "==================================="
    echo "Iteration $iteration"
    untested=$(grep -c 'TOBETESTED' APPROACHES.md 2>/dev/null || echo "0")
    echo "Untested approaches remaining: $untested"
    echo "Running agent: $agent"
    echo "==================================="

    result=$(run_agent "$agent" "$prompt")
    echo "$result"

    echo "-----------------------------------"
    echo "Iteration $iteration complete."
    ((iteration++))
done

echo "==================================="
echo "All approaches tested! Total iterations: $((iteration-1))"
