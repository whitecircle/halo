#!/bin/zsh
# Print a figure page's canvas size as "width height" in CSS px. The width comes from the page's body rule. The
# height does too when the rule fixes one; otherwise the page is laid out in Chrome and its content height is
# measured, so a page whose body sets only a width grows as tall as its content.
# Usage: canvas-size.sh page.html
set -euo pipefail

page=${1:A}
chrome="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
body=$(perl -0ne 'print $1 if /(?:^|[\s,}])body\s*\{([^}]*)\}/s' "$page")
width=$(perl -ne 'print $1 if /(?<![\w-])width:\s*(\d+)px;/' <<< "$body")
height=$(perl -ne 'print $1 if /(?<![\w-])height:\s*(\d+)px;/' <<< "$body")

if [[ -z "$width" ]]; then
  echo "$page fixes no canvas width: give its body rule a width in px." >&2
  exit 2
fi

if [[ -n "$height" ]]; then
  echo "$width $height"
  exit 0
fi

# The probe sits beside the page so its relative links (the kit, data files) resolve.
probe="${page:h}/.${page:t:r}.size.html"
trap 'rm -f "$probe"' EXIT
sed 's#</body>##g; s#</html>##g' "$page" > "$probe"
cat >> "$probe" <<'EOF'
<script>
window.addEventListener("load", () => setTimeout(() => {
  const out = document.createElement("pre");
  out.textContent = "SIZE" + `-H:${Math.ceil(document.body.getBoundingClientRect().height)};`;
  document.documentElement.append(out);
}, 300));
</script>
</body></html>
EOF

measured=$("$chrome" --headless=new --disable-gpu --hide-scrollbars --allow-file-access-from-files \
  --window-size="$width,2400" --virtual-time-budget=4000 --dump-dom "file://$probe" 2>/dev/null \
  | grep -o "SIZE-H:[0-9][0-9]*" | head -1 || true)

if [[ -z "$measured" ]]; then
  echo "Chrome did not lay out $page, so its content height could not be measured." >&2
  exit 1
fi

echo "$width ${measured#SIZE-H:}"
