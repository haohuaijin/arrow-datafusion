#!/usr/bin/env bash
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

set -euo pipefail

SCRIPT_DIR=$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
    pwd
)
WORKSPACE_ROOT="${SCRIPT_DIR}"

TIME_BIN=${TIME_BIN:-/usr/bin/time}
RUNS=${RUNS:-10}
OUTPUT_CSV=${1:-"${WORKSPACE_ROOT}/target/window_topk_rss.csv"}
if [[ "${OUTPUT_CSV}" == *.* ]]; then
    OUTPUT_PRETTY="${OUTPUT_CSV%.*}.txt"
else
    OUTPUT_PRETTY="${OUTPUT_CSV}.txt"
fi

usage() {
    cat <<EOF
Collect max RSS for all cases in datafusion/core/benches/window_topk.rs.

Usage:
  $(basename "$0") [output_csv]

Examples:
  $(basename "$0")
  $(basename "$0") /tmp/window_topk_rss.csv

Notes:
  - This script uses \`${TIME_BIN} -l\`, which is expected on macOS.
  - The CSV stores RSS values formatted in MB.
  - RSS and time use a trimmed average over ${RUNS} runs, dropping one max and one min run.
  - A pretty text summary is written next to the CSV.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

require_command() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Missing required command: $cmd" >&2
        exit 1
    fi
}

find_example_binary() {
    local examples_dir="${WORKSPACE_ROOT}/target/release/examples"
    local direct_binary="${examples_dir}/query"

    if [[ -x "${direct_binary}" ]]; then
        printf '%s\n' "${direct_binary}"
        return 0
    fi

    find "${examples_dir}" -maxdepth 1 -type f -name 'query-*' -perm -111 2>/dev/null \
        | head -n 1
}

csv_escape() {
    local value="$1"
    value=${value//\"/\"\"}
    printf '"%s"' "$value"
}

collect_metrics() {
    local example_binary="$1"
    local num_categories="$2"
    local rows_per_category="$3"
    local use_window_topk="$4"
    local stderr_file
    local rss
    local real_time
    local rss_values=""
    local real_time_values=""
    local avg_rss
    local avg_real_time_ms
    local run

    for ((run = 1; run <= RUNS; run++)); do
        stderr_file=$(mktemp)
        "${TIME_BIN}" -l \
            "${example_binary}" \
            "${num_categories}" \
            "${rows_per_category}" \
            "${use_window_topk}" \
            >/dev/null 2>"${stderr_file}"

        rss=$(awk '/maximum resident set size/ { print $1; exit }' "${stderr_file}")
        real_time=$(awk '/ real .* user .* sys/ { print $1; exit }' "${stderr_file}")
        if [[ -z "${rss}" || -z "${real_time}" ]]; then
            echo "Failed to parse metrics from /usr/bin/time -l output:" >&2
            cat "${stderr_file}" >&2
            rm -f "${stderr_file}"
            exit 1
        fi

        rss_values+="${rss}"$'\n'
        real_time_values+="${real_time}"$'\n'
        rm -f "${stderr_file}"
    done

    avg_rss=$(printf '%s' "${rss_values}" | awk '
        {
            values[++count] = $1
            sum += $1
            if (count == 1 || $1 < min) {
                min = $1
            }
            if (count == 1 || $1 > max) {
                max = $1
            }
        }
        END {
            if (count < 3) {
                printf "%.2f", sum / count
            } else {
                printf "%.2f", (sum - min - max) / (count - 2)
            }
        }
    ')

    avg_real_time_ms=$(printf '%s' "${real_time_values}" | awk '
        {
            sum += $1
            if (count == 0 || $1 < min) {
                min = $1
            }
            if (count == 0 || $1 > max) {
                max = $1
            }
            count += 1
        }
        BEGIN {
        }
        END {
            if (count < 3) {
                printf "%.2f", (sum / count) * 1000
            } else {
                printf "%.2f", ((sum - min - max) / (count - 2)) * 1000
            }
        }
    ')

    printf '%s|%s\n' "${avg_rss}" "${avg_real_time_ms}"
}

compare_percent() {
    local rss_disable="$1"
    local rss_enable="$2"

    awk -v disable="${rss_disable}" -v enable="${rss_enable}" '
        BEGIN {
            if (disable == 0) {
                printf "0.00"
            } else {
                printf "%.2f", ((enable - disable) / disable) * 100
            }
        }
    '
}

bytes_to_mb() {
    local bytes="$1"

    awk -v value="${bytes}" '
        BEGIN {
            printf "%.2f", value / (1024 * 1024)
        }
    '
}

require_command cargo
require_command awk
require_command find
require_command mktemp

if ((RUNS < 3)); then
    echo "RUNS must be at least 3 to drop one max and one min value" >&2
    exit 1
fi

mkdir -p "$(dirname -- "${OUTPUT_CSV}")"

echo "Building datafusion-examples query example in release mode..."
(
    cd "${WORKSPACE_ROOT}"
    cargo build -p datafusion-examples --example query --release --quiet
)

EXAMPLE_BINARY=$(find_example_binary)
if [[ -z "${EXAMPLE_BINARY}" ]]; then
    echo "Could not find the built query example binary in target/release/examples" >&2
    exit 1
fi

echo "Writing RSS results to ${OUTPUT_CSV}"
echo "Writing pretty RSS summary to ${OUTPUT_PRETTY}"
printf '%s\n' \
    'label,num_categories,rows_per_category,rss_disable_mb_avg,rss_enable_mb_avg,rss_enable_vs_disable_pct,time_disable_ms_avg,time_enable_ms_avg,time_enable_vs_disable_pct' \
    >"${OUTPUT_CSV}"
printf '%-32s %-16s %-18s %-14s %-14s %-18s %-14s %-14s %-18s\n' \
    'label' \
    'num_categories' \
    'rows_per_category' \
    'disable_mb' \
    'enable_mb' \
    'rss_pct' \
    'disable_ms' \
    'enable_ms' \
    'time_pct' \
    >"${OUTPUT_PRETTY}"
printf '%-32s %-16s %-18s %-14s %-14s %-18s %-14s %-14s %-18s\n' \
    '--------------------------------' \
    '----------------' \
    '------------------' \
    '--------------' \
    '--------------' \
    '------------------' \
    '--------------' \
    '--------------' \
    '------------------' \
    >>"${OUTPUT_PRETTY}"

CASES=(
    'few categories, many rows|10|100000'
    'few categories, many rows|50|100000'
    'few categories, many rows|100|100000'
    'few categories, many rows|1000|100000'
    'many categories, many rows|10000|10000'
    'many categories, few rows|100000|1000'
    'many categories, few rows|100000|100'
    'many categories, few rows|100000|50'
    'many categories, few rows|100000|10'
    'many categories, few rows|100000|1'
)

for case in "${CASES[@]}"; do
    IFS='|' read -r label num_categories rows_per_category <<<"${case}"

    IFS='|' read -r rss_disable time_disable_raw <<<"$(collect_metrics \
        "${EXAMPLE_BINARY}" \
        "${num_categories}" \
        "${rows_per_category}" \
        false)"
    IFS='|' read -r rss_enable time_enable_raw <<<"$(collect_metrics \
        "${EXAMPLE_BINARY}" \
        "${num_categories}" \
        "${rows_per_category}" \
        true)"
    rss_disable_mb=$(bytes_to_mb "${rss_disable}")
    rss_enable_mb=$(bytes_to_mb "${rss_enable}")
    rss_enable_vs_disable_pct=$(compare_percent "${rss_disable}" "${rss_enable}")
    time_enable_vs_disable_pct=$(compare_percent "${time_disable_raw}" "${time_enable_raw}")

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$(csv_escape "${label}")" \
        "${num_categories}" \
        "${rows_per_category}" \
        "${rss_disable_mb}" \
        "${rss_enable_mb}" \
        "${rss_enable_vs_disable_pct}" \
        "${time_disable_raw}" \
        "${time_enable_raw}" \
        "${time_enable_vs_disable_pct}" \
        >>"${OUTPUT_CSV}"
    printf '%-32s %-16s %-18s %-14s %-14s %-18s %-14s %-14s %-18s\n' \
        "${label}" \
        "${num_categories}" \
        "${rows_per_category}" \
        "${rss_disable_mb}" \
        "${rss_enable_mb}" \
        "${rss_enable_vs_disable_pct}%" \
        "${time_disable_raw}" \
        "${time_enable_raw}" \
        "${time_enable_vs_disable_pct}%" \
        >>"${OUTPUT_PRETTY}"

    echo "Collected ${label} / ${num_categories} / ${rows_per_category}: rss disable=${rss_disable_mb} MB, rss enable=${rss_enable_mb} MB, rss pct=${rss_enable_vs_disable_pct}%, time disable=${time_disable_raw} ms avg, time enable=${time_enable_raw} ms avg, time pct=${time_enable_vs_disable_pct}%"
done

echo "Done."
