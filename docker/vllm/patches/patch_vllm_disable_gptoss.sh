#!/bin/bash
# Patch script to disable openai_gptoss reasoning parser and harmony in vllm,
# and route vllm's logging to stderr so --enable-log-requests / --enable-log-outputs are visible.
#
# Every target below is a file vllm 0.26.0 actually ships. Nothing is existence-gated: a missing
# target or an unmatched form is a hard failure, so an upstream move is caught at build time
# instead of silently shipping an unpatched server.
#
# Usage:
#   ./docker/vllm/patches/patch_vllm_disable_gptoss.sh                      # auto-detect (interpreter's site-packages)
#   ./docker/vllm/patches/patch_vllm_disable_gptoss.sh /path/to/site-packages/vllm  # explicit path

set -e

# ---------------------------------------------------------------------------
# Determine python command: prefer "python3", fall back to "python"
# ---------------------------------------------------------------------------
if command -v python3 &>/dev/null; then
    PY="python3"
elif command -v python &>/dev/null; then
    PY="python"
else
    echo "ERROR: no python interpreter on PATH" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Resolve vllm path: use argument or auto-detect
# ---------------------------------------------------------------------------
if [ -n "$1" ]; then
    VLLM_PATH="$1"
else
    echo "No path argument given — auto-detecting..."
    SITE_PACKAGES="$($PY -c "import sysconfig; print(sysconfig.get_path('purelib'))" 2>/dev/null)" || {
        echo "Error: could not detect site-packages. Pass the vllm path explicitly."
        exit 1
    }
    VLLM_PATH="$SITE_PACKAGES/vllm"
fi

if [ ! -d "$VLLM_PATH" ]; then
    echo "Error: vllm directory not found at $VLLM_PATH"
    exit 1
fi

# ---------------------------------------------------------------------------
# Detect vllm version
# ---------------------------------------------------------------------------
VLLM_VERSION="$($PY -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")"
echo "============================================"
echo "  vllm version : $VLLM_VERSION"
echo "  vllm path    : $VLLM_PATH"
echo "============================================"
echo ""

ERRORS=0   # count validation failures

# Every patch targets a file this vllm must ship. Absence means upstream moved it, which has to fail
# the build rather than soft-skip into an unpatched server.
require_file() {
    local label="$1" file="$2"
    if [ -f "$file" ]; then
        return 0
    fi
    echo "  FAIL  $label: target file not found ($file)"
    ERRORS=$((ERRORS + 1))
    return 1
}

# ===========================================================================
# Patch 1: force the harmony gates off in the serving + rendering files
# ===========================================================================
# Three spellings coexist in 0.26.0, so all three are applied to every file:
#   A: `self.use_harmony = self.model_config...`  (responses/serving.py)
#   B: `self.use_harmony = model_config...`       (renderers/online_{,de}renderer.py — local arg)
#   C: `is_harmony=self.model_config...` kwarg    (chat_completion/serving.py, responses/serving.py)
HARMONY_PATTERN_A='self\.use_harmony = self\.model_config\.hf_config\.model_type == "gpt_oss"'
HARMONY_REPLACE_A='self.use_harmony = False  # self.model_config.hf_config.model_type == "gpt_oss"'
HARMONY_PATTERN_B='self\.use_harmony = model_config\.hf_config\.model_type == "gpt_oss"'
HARMONY_REPLACE_B='self.use_harmony = False  # model_config.hf_config.model_type == "gpt_oss"'
# The optional trailing comma is captured and re-emitted BEFORE the comment: leaving it at the end
# would bury it inside the comment, dropping the separator if a kwarg is ever added after this one.
HARMONY_PATTERN_C='is_harmony=self\.model_config\.hf_config\.model_type == "gpt_oss"\(,\)\{0,1\}'
HARMONY_REPLACE_C='is_harmony=False\1  # patched: was model_type == "gpt_oss"'
# Idempotency markers:
HARMONY_CHECK='self\.use_harmony = False.*# .*model_config\.hf_config'
HARMONY_CHECK_C='is_harmony=False,\{0,1\}  # patched'
# Any harmony gate keyed on gpt_oss, in whatever comparison form (==, in (...), startswith, …). Used
# to tell "this file had nothing to patch" from "this file has a gate the sed forms did not match".
HARMONY_GATE_ANY='(use|is)_harmony[[:space:]]*=[^#]*gpt_oss'

# Unpatched, the gpt-oss path renders through openai_harmony and ignores the jinja chat template
# (train/serve prompt mismatch).
HARMONY_FILES=(
    "$VLLM_PATH/entrypoints/openai/chat_completion/serving.py"
    "$VLLM_PATH/entrypoints/openai/responses/serving.py"
    "$VLLM_PATH/renderers/online_renderer.py"
    "$VLLM_PATH/renderers/online_derenderer.py"
)

for HFILE in "${HARMONY_FILES[@]}"; do
    require_file "harmony gate" "$HFILE" || continue
    echo "Patching $HFILE..."
    if grep -q -e "$HARMONY_CHECK" -e "$HARMONY_CHECK_C" "$HFILE"; then
        echo "  - Already patched, skipping..."
        continue
    fi
    sed -i -e "s/$HARMONY_PATTERN_A/$HARMONY_REPLACE_A/g" \
           -e "s/$HARMONY_PATTERN_B/$HARMONY_REPLACE_B/g" \
           -e "s/$HARMONY_PATTERN_C/$HARMONY_REPLACE_C/g" "$HFILE"
    # Report what the sed actually did, per file — never announce success unconditionally.
    if grep -q -e "$HARMONY_CHECK" -e "$HARMONY_CHECK_C" "$HFILE"; then
        echo "  - Disabled harmony (use_harmony/is_harmony forced False)"
    else
        echo "  FAIL  harmony gate present but none of the known forms matched"
        ERRORS=$((ERRORS + 1))
    fi
done

# ===========================================================================
# Patch 2: reasoning/__init__.py - Comment out openai_gptoss parser
# ===========================================================================
REASONING_INIT="$VLLM_PATH/reasoning/__init__.py"
if require_file "reasoning parser registry" "$REASONING_INIT"; then
    echo "Patching $REASONING_INIT..."
    if grep -q '# "openai_gptoss"' "$REASONING_INIT"; then
        echo "  - Already patched, skipping..."
    else
        awk '
        /^    "openai_gptoss": \(/ {
            print "    # \"openai_gptoss\": ("
            getline; print "    #     " substr($0, 9)
            getline; print "    #     " substr($0, 9)
            getline; print "    # ),"
            next
        }
        { print }
        ' "$REASONING_INIT" > "${REASONING_INIT}.tmp" && mv "${REASONING_INIT}.tmp" "$REASONING_INIT"
        echo "  - Commented out openai_gptoss parser registration"
    fi
fi

# ===========================================================================
# Patch 3: envs.py - Default logging stream from stdout → stderr
# vLLM defaults to stdout, but uvicorn and most servers log to stderr.
# ===========================================================================
ENVS_PY="$VLLM_PATH/envs.py"
if require_file "logging stream default" "$ENVS_PY"; then
    echo "Patching $ENVS_PY..."
    if grep -q 'VLLM_LOGGING_STREAM.*ext://sys.stderr' "$ENVS_PY"; then
        echo "  - Already patched, skipping..."
    else
        sed -i 's|VLLM_LOGGING_STREAM: str = "ext://sys.stdout"|VLLM_LOGGING_STREAM: str = "ext://sys.stderr"  # patched: was stdout|' "$ENVS_PY"
        sed -i 's|"VLLM_LOGGING_STREAM": lambda: os.getenv("VLLM_LOGGING_STREAM", "ext://sys.stdout")|"VLLM_LOGGING_STREAM": lambda: os.getenv("VLLM_LOGGING_STREAM", "ext://sys.stderr")|' "$ENVS_PY"
        echo "  - Changed default logging stream from stdout to stderr"
    fi
fi

# ===========================================================================
# Patch 4: model_executor/models/config.py - Disable auto-setting reasoning_parser
# ===========================================================================
CONFIG_PY="$VLLM_PATH/model_executor/models/config.py"
if require_file "reasoning_parser auto-set" "$CONFIG_PY"; then
    echo "Patching $CONFIG_PY..."
    if grep -q '# Disabled: do not force openai_gptoss' "$CONFIG_PY"; then
        echo "  - Already patched, skipping..."
    else
        awk '
        /if structured_outputs_config\.reasoning_parser == "":/ {
            print "        # Disabled: do not force openai_gptoss reasoning parser"
            print "        # " substr($0, 9)
            getline
            print "        # " substr($0, 9)
            next
        }
        { print }
        ' "$CONFIG_PY" > "${CONFIG_PY}.tmp" && mv "${CONFIG_PY}.tmp" "$CONFIG_PY"
        echo "  - Disabled auto-setting of openai_gptoss reasoning parser"
    fi
fi

# ===========================================================================
# Patch 5: unquantized MoE backend oracle — demote FlashInfer CUTLASS
# ===========================================================================
# Why: FlashInfer CUTLASS unquantized MoE has known bf16 correctness issues
# (vLLM authors already demote it for Qwen3.5 + DP>1 a few lines down in the
# same file). It produces pure garbage output for gpt-oss-20b bf16 unquantized
# MoE at any DP/TP. Demote it to last priority so models that can use TRTLLM
# still get the fast kernel and models whose layout TRTLLM rejects (gpt-oss)
# fall through to TRITON instead of the broken CUTLASS path.
MOE_ORACLE="$VLLM_PATH/model_executor/layers/fused_moe/oracle/unquantized.py"
if require_file "unquantized MoE oracle" "$MOE_ORACLE"; then
    echo "Patching $MOE_ORACLE..."
    if grep -q 'universal-benchmarks' "$MOE_ORACLE"; then
        echo "  - Already patched, skipping..."
    else
        $PY - "$MOE_ORACLE" <<'PY' || ERRORS=$((ERRORS + 1))
import sys
p = sys.argv[1]
src = open(p).read()
marker = (
    "    elif current_platform.is_cuda():\n"
    "        _AVAILABLE_BACKENDS = [\n"
    "            UnquantizedMoeBackend.FLASHINFER_TRTLLM,\n"
    "            UnquantizedMoeBackend.FLASHINFER_CUTLASS,\n"
    "            UnquantizedMoeBackend.TRITON,\n"
    "            UnquantizedMoeBackend.BATCHED_TRITON,\n"
    "        ]"
)
patch = marker + """

        # WORKAROUND (universal-benchmarks): FlashInfer CUTLASS BF16 has
        # correctness issues for unquantized MoE (vLLM authors already
        # demote it for Qwen3.5 + DP>1 below — same kernel produces pure
        # garbage for gpt-oss-20b bf16 unquantized MoE at any DP/TP).
        # Demote *only* CUTLASS: TRTLLM stays in priority slot 1 (correct
        # for Qwen3.5/3.6, faster than TRITON), and models whose layout
        # TRTLLM rejects (e.g. gpt-oss-20b) fall through to TRITON instead
        # of the broken CUTLASS kernel."""
patch += "\n        _move_to_back(_AVAILABLE_BACKENDS, UnquantizedMoeBackend.FLASHINFER_CUTLASS)"
if marker not in src:
    sys.stderr.write("  FAIL  expected backend priority block not found\n")
    sys.exit(2)
open(p, "w").write(src.replace(marker, patch))
print("  - Demoted FlashInfer CUTLASS to last priority for unquantized MoE")
PY
    fi
fi

# ===========================================================================
# Clean __pycache__ to ensure patched .py files take effect
# ===========================================================================
echo "Cleaning __pycache__ for patched modules..."
find "$VLLM_PATH" -type d -name __pycache__ -exec sh -c '
    for d; do
        rm -f "$d"/serving*.pyc "$d"/__init__*.pyc "$d"/config*.pyc "$d"/envs*.pyc \
              "$d"/online_renderer*.pyc "$d"/online_derenderer*.pyc "$d"/unquantized*.pyc 2>/dev/null
    done
' _ {} +
echo "  - Cleaned stale .pyc files"

# ===========================================================================
# Validation: verify every patch was applied correctly
# ===========================================================================
echo ""
echo "============================================"
echo "  Validating patches..."
echo "============================================"

validate() {
    local label="$1" file="$2" pattern="$3"
    if [ -f "$file" ] && grep -q "$pattern" "$file"; then
        echo "  OK  $label"
    else
        echo "  FAIL  $label  ($file)"
        ERRORS=$((ERRORS + 1))
    fi
}

# Harmony: every serving/rendering file must carry the patch
harmony_ok=0
for HFILE in "${HARMONY_FILES[@]}"; do
    if [ -f "$HFILE" ] && grep -q -e "$HARMONY_CHECK" -e "$HARMONY_CHECK_C" "$HFILE"; then
        harmony_ok=$((harmony_ok + 1))
    fi
done
if [ "$harmony_ok" -eq "${#HARMONY_FILES[@]}" ]; then
    echo "  OK  use_harmony = False ($harmony_ok/${#HARMONY_FILES[@]} file(s))"
else
    echo "  FAIL  use_harmony = False ($harmony_ok/${#HARMONY_FILES[@]} file(s) patched)"
    ERRORS=$((ERRORS + 1))
fi

# Belt: fail if any UNPATCHED `*_harmony = ... == "gpt_oss"` gate survives (matches the code form only,
# not the patched `= False  # ...` comment) — so a stale HARMONY_FILES list fails loud.
grep_rc=0
# `|| grep_rc=$?` keeps `set -e` from aborting on grep's rc 1 ("no match") while still capturing a
# real failure rc — a bare `|| true` would hide rc 2 and read an unreadable tree as a clean one.
residual=$(grep -rEln "$HARMONY_GATE_ANY" "$VLLM_PATH" 2>/dev/null) || grep_rc=$?
if [ "$grep_rc" -gt 1 ]; then
    # rc 1 is "no match" (the good case); anything above is a real grep failure, which must not be
    # mistaken for a clean tree.
    echo "  FAIL  residual harmony scan could not run (grep exit $grep_rc)"
    ERRORS=$((ERRORS + 1))
elif [ -n "$residual" ]; then
    echo "  FAIL  unpatched harmony gate still present in:"
    echo "$residual" | sed 's/^/          /'
    ERRORS=$((ERRORS + 1))
else
    echo "  OK  no unpatched use_harmony/is_harmony gpt_oss gate remains"
fi

validate "openai_gptoss commented out" "$REASONING_INIT" '# "openai_gptoss"'
validate "envs.py logging stream = stderr" "$ENVS_PY" 'ext://sys.stderr'
validate "config.py reasoning_parser disabled" "$CONFIG_PY" '# Disabled: do not force openai_gptoss'
validate "unquantized MoE oracle: CUTLASS demoted" "$MOE_ORACLE" 'universal-benchmarks'

# Smoke-test the patched tree. This must be unconditional: the seds rewrite Python source in place,
# so a malformed edit has to surface here rather than at server start.
echo ""
echo "  Smoke-testing patched vllm import..."
if $PY -c "import vllm; print('    vllm', vllm.__version__, 'imported OK')"; then
    echo "  OK  vllm import"
else
    echo "  FAIL  vllm import (a patch produced invalid Python)"
    ERRORS=$((ERRORS + 1))
fi

# Byte-compile every file the seds touched — import alone does not reach the lazily-imported ones.
for PATCHED in "${HARMONY_FILES[@]}" "$ENVS_PY" "$CONFIG_PY" "$MOE_ORACLE"; do
    [ -f "$PATCHED" ] || continue
    if ! $PY -m py_compile "$PATCHED" 2>&1; then
        echo "  FAIL  py_compile $PATCHED"
        ERRORS=$((ERRORS + 1))
    fi
done

echo ""
if [ $ERRORS -eq 0 ]; then
    echo "============================================"
    echo "  All patches validated successfully!"
    echo "============================================"
else
    echo "============================================"
    echo "  WARNING: $ERRORS validation(s) failed"
    echo "============================================"
    exit 1
fi

echo ""
echo "Patch summary (vllm $VLLM_VERSION):"
echo "  1. use_harmony/is_harmony = False in serving + rendering files (custom templates work)"
echo "  2. Commented out openai_gptoss parser in reasoning/__init__.py"
echo "  3. Default logging stream → stderr in envs.py"
echo "  4. Disabled auto-setting of reasoning_parser in config.py"
echo "  5. Demoted FlashInfer CUTLASS unquantized MoE (buggy bf16 → falls"
echo "     through to TRITON; TRTLLM stays in priority for Qwen3.5/3.6)"
echo "  6. Cleaned __pycache__ bytecode for patched modules"
echo ""
echo "Tip: set PYTHONUNBUFFERED=1 if logs are still not visible, and --max-log-len to bound"
echo "     request/output logging (vLLM's own default is unlimited)."
