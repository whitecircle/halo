#!/bin/zsh
# Render a social image page to a 2x PNG with headless Chrome, then optimize it losslessly with OptiPNG.
# Usage: render.sh page.html out.png [width height]   (CSS px; default: the size the page's body rule fixes)
# The brand fonts are installed system fonts. A page links the skill's kit/ and any logo in its assets/ by a
# path relative to the page.
set -euo pipefail

if (( $# != 2 && $# != 4 )); then
  echo "Usage: render.sh page.html out.png [width height]" >&2
  exit 2
fi

page=${1:A}
out=${2:A}

if (( $# == 4 )); then
  width=$3
  height=$4
else
  size=$(perl -0ne 'print "$1 $2" if /(?:^|[\s,}])body\s*\{[^}]*?\bwidth:\s*(\d+)px;[^}]*?\bheight:\s*(\d+)px;/s' "$page")
  if [[ -z "$size" ]]; then
    echo "$page fixes no canvas size: give its body rule width and height in px, or pass them after out.png." >&2
    exit 2
  fi
  width=${size% *}
  height=${size#* }
fi

chrome=""
for candidate in "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  google-chrome google-chrome-stable chromium chromium-browser; do
  if [[ -x "$candidate" ]] || command -v "$candidate" >/dev/null 2>&1; then
    chrome=$candidate
    break
  fi
done

if [[ -z "$chrome" ]]; then
  echo "Google Chrome or Chromium is needed to render the image, and neither was found." >&2
  exit 1
fi

if ! command -v optipng >/dev/null 2>&1; then
  echo "OptiPNG is needed to optimize the image, and it was not found: install it (brew install optipng, apt install optipng), then render again." >&2
  exit 1
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

# A family that does not resolve measures exactly like its generic fallback.
cat > "$work/probe.html" <<'EOF'
<!doctype html>
<body><script>
  const ctx = document.createElement("canvas").getContext("2d");
  const sample = "Halo 0123456789";
  const roles = [
    ["ABC Otto", "ABC Otto Variable"],
    ["ABC Monument Grotesk", "ABC Monument Grotesk Variable"],
  ];
  const resolves = (name) => ["monospace", "serif"].every((generic) => {
    ctx.font = `40px ${generic}`;
    const fallback = ctx.measureText(sample).width;
    ctx.font = `40px "${name}", ${generic}`;
    return ctx.measureText(sample).width !== fallback;
  });
  const missing = roles.filter((names) => !names.some(resolves)).map((names) => names[0]);
  document.body.textContent = `FONTS-MISSING:${missing.join(", ")};`;
</script></body>
EOF

probe=$("$chrome" --headless=new --disable-gpu --dump-dom "file://$work/probe.html" 2>/dev/null \
  | grep -o 'FONTS-MISSING:[^;]*' || true)

if [[ -z "$probe" ]]; then
  echo "The font check page did not run in Chrome, so the brand fonts could not be verified." >&2
  exit 1
fi

if [[ -n "${probe#FONTS-MISSING:}" ]]; then
  echo "Not installed as a system font: ${probe#FONTS-MISSING:}. Install ABC Otto and ABC Monument Grotesk, then render again." >&2
  exit 1
fi

# The page renders where it sits, so its relative links (the kit, a logo, a data file) resolve. The virtual time
# budget lets the page's script build the chart before the shot.
"$chrome" --headless=new --disable-gpu --hide-scrollbars --allow-file-access-from-files \
  --force-device-scale-factor=2 --window-size="$width,$height" --virtual-time-budget=4000 \
  --screenshot="$out" "file://$page" >/dev/null 2>&1

if [[ ! -s "$out" ]]; then
  echo "Chrome did not write $out." >&2
  exit 1
fi

# Lossless: the pixels stay exactly as Chrome drew them. -o2 is OptiPNG's default, and higher levels save
# well under 1% more on these flat images at several times the time.
optipng -quiet -o2 -strip all "$out"

if command -v sips >/dev/null 2>&1; then
  sips -g pixelWidth -g pixelHeight "$out" | tail -2
else
  echo "Wrote $out"
fi
