#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    printf '%s\n' \
        'Usage:' \
        '  managed_job.sh launch NAME WORKDIR STATE_DIR -- COMMAND [ARG...]' \
        '  managed_job.sh status NAME STATE_DIR' \
        '  managed_job.sh tail NAME STATE_DIR [LINES]'
    exit 2
}

action="${1:-}"
case "$action" in
    launch)
        [[ $# -ge 6 ]] || usage
        name="$2"
        workdir="$3"
        state_dir="$4"
        [[ "$5" == -- ]] || usage
        shift 5
        [[ "$name" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || {
            printf 'Invalid job name: %s\n' "$name" >&2
            exit 2
        }
        [[ -d "$workdir" ]] || {
            printf 'Workdir does not exist: %s\n' "$workdir" >&2
            exit 2
        }
        command -v tmux >/dev/null 2>&1 || {
            printf 'tmux is required.\n' >&2
            exit 2
        }
        if tmux has-session -t "$name" 2>/dev/null; then
            printf 'Session already exists: %s\n' "$name" >&2
            exit 1
        fi

        mkdir -p "$state_dir"
        runner="$state_dir/$name.runner.sh"
        status="$state_dir/$name.status"
        log="$state_dir/$name.log"
        {
            printf '#!/usr/bin/env bash\nset -Eeuo pipefail\n'
            printf 'status=%q\nlog=%q\nworkdir=%q\n' "$status" "$log" "$workdir"
            printf 'job_command=(' 
            printf ' %q' "$@"
            printf ' )\n'
            printf '%s\n' \
                'finish() {' \
                '    rc=$?' \
                '    if [[ $rc -eq 0 ]]; then' \
                '        printf "complete exit=0 time=%s\\n" "$(date --iso-8601=seconds)" > "$status"' \
                '    else' \
                '        printf "failed exit=%s time=%s\\n" "$rc" "$(date --iso-8601=seconds)" > "$status"' \
                '    fi' \
                '}' \
                'trap finish EXIT' \
                'cd "$workdir"' \
                'printf "running pid=%s time=%s\\n" "$$" "$(date --iso-8601=seconds)" > "$status"' \
                'exec > >(tee -a "$log") 2>&1' \
                'printf "command="' \
                'printf "%q " "${job_command[@]}"' \
                'printf "\\n"' \
                '"${job_command[@]}"'
        } > "$runner"
        chmod 700 "$runner"
        tmux new-session -d -s "$name" "$runner"
        printf 'launched session=%s status=%s log=%s\n' "$name" "$status" "$log"
        ;;
    status)
        [[ $# -eq 3 ]] || usage
        name="$2"
        state_dir="$3"
        cat "$state_dir/$name.status" 2>/dev/null || printf 'status file absent\n'
        if tmux has-session -t "$name" 2>/dev/null; then
            printf 'session=alive\n'
        else
            printf 'session=absent\n'
        fi
        ;;
    tail)
        [[ $# -ge 3 && $# -le 4 ]] || usage
        name="$2"
        state_dir="$3"
        lines="${4:-80}"
        [[ "$lines" =~ ^[0-9]+$ ]] || usage
        tail -n "$lines" "$state_dir/$name.log"
        ;;
    *)
        usage
        ;;
esac
