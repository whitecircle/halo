#!/bin/zsh
# Check a figure page as Chrome lays it out: any text or mark crossing the canvas margins, text overlapping
# other text, text crowding text in another column (a gap under 12px on a 1600 canvas), text spilling out
# of its own box, a drawn line running through text, and any error the page's script throws. Prints each
# problem and exits 1, or prints "ok" and the gaps around the figure.
# Usage: check.sh page.html   (canvas size read from the page's body rule)
set -euo pipefail
page=${1:A}
chrome="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
size=$(perl -0ne 'print "$1 $2" if /(?:^|[\s,}])body\s*\{[^}]*?\bwidth:\s*(\d+)px;[^}]*?\bheight:\s*(\d+)px;/s' "$page")
width=${size% *}
height=${size#* }
if (( width >= 1600 )); then side=64; top=52; bottom=48; else side=48; top=40; bottom=36; fi
# The instrumented copy sits beside the page so its relative paths (data files) still resolve.
probe="${page:h}/.${page:t:r}.check.html"
trap 'rm -f "$probe"' EXIT

# Errors are collected from the start of <head>, before the page's own script runs, since a page whose
# script throws draws an incomplete figure that may still sit inside its margins.
perl -0pe 's{<head>}{<head><script>window.__pageErrors = []; addEventListener("error", (e) => __pageErrors.push(e.message)); addEventListener("unhandledrejection", (e) => __pageErrors.push(String(e.reason)));</script>}' "$page" \
  | sed 's#</body>##g; s#</html>##g' > "$probe"
if ! grep -q '__pageErrors' "$probe"; then
  echo "$page has no <head> tag to hook the error capture into." >&2
  exit 1
fi
cat >> "$probe" <<EOF
<script>
window.addEventListener("load", () => setTimeout(() => {
  const W = $width, H = $height;
  const box = { l: $side - 0.5, t: $top - 0.5, r: W - $side + 0.5, b: H - $bottom + 0.5 };
  const minGap = 12 * W / 1600;
  const hidden = (n) => { for (let e = n; e && e.nodeType === 1; e = e.parentElement) { const s = getComputedStyle(e); if (s.display === "none" || s.visibility === "hidden" || s.opacity === "0") return true; } return false; };
  const blockOf = (n) => { for (let e = n.parentElement; e; e = e.parentElement) { const d = getComputedStyle(e).display; if (!d.startsWith("inline")) return e; } return document.body; };
  const name = (n) => (n.textContent || "").trim().replace(/\s+/g, " ").slice(0, 38);
  const items = [];
  for (const n of document.body.querySelectorAll("*")) {
    if (["SCRIPT", "STYLE"].includes(n.tagName) || hidden(n)) continue;
    const texts = [...n.childNodes].filter((c) => c.nodeType === 3 && c.textContent.trim() !== "");
    if (!texts.length) continue;
    const fs = parseFloat(getComputedStyle(n).fontSize) || 16;
    const slack = 0.16 * fs;
    for (const t of texts) {
      const range = document.createRange();
      range.selectNodeContents(t);
      for (const r of range.getClientRects()) {
        if (r.width < 1 || r.height < 1) continue;
        // Inline text (sentences, spans) flows within its block, where a space is the gap. Grid and flex
        // items are blockified, so two columns of one grid still count as separate.
        const inline = getComputedStyle(n).display.startsWith("inline");
        items.push({ n, fs, flow: inline ? blockOf(n) : n, l: r.left, r: r.right, t: r.top + slack, b: r.bottom - slack });
      }
    }
  }
  const bad = new Set();
  const report = (msg) => bad.add(msg);
  for (const message of window.__pageErrors) report(\`SCRIPT   \${message}\`);
  // Whether a computed colour paints anything: its alpha is read, so opaque black is not mistaken for
  // transparent rgba(0, 0, 0, 0).
  const opaque = (c) => {
    if (!c || c === "transparent") return false;
    const slash = c.match(/\/\s*([\d.]+)(%?)\s*\)$/);
    if (slash) return parseFloat(slash[1]) > 0;
    const rgba = c.match(/^rgba\(([^)]+)\)$/);
    if (rgba) return parseFloat(rgba[1].split(",")[3]) > 0;
    return true;
  };
  // Marks: drawn SVG shapes and boxes with a fill or border, whatever the canvas and its column hold.
  for (const n of document.body.querySelectorAll("*")) {
    if (hidden(n) || ["svg", "g", "defs", "marker", "text", "tspan"].includes(n.tagName)) continue;
    const s = getComputedStyle(n);
    const isMark = n instanceof SVGGraphicsElement;
    const isBox = !(n instanceof SVGElement) && !n.classList.contains("canvas")
      && (opaque(s.backgroundColor) || s.borderTopWidth !== "0px" || s.borderLeftWidth !== "0px");
    if (!isMark && !isBox) continue;
    const r = n.getBoundingClientRect();
    if (r.width === 0 && r.height === 0) continue;
    if (r.left < box.l || r.top < box.t || r.right > box.r || r.bottom > box.b) {
      report(\`MARGIN   <\${n.tagName.toLowerCase()} class="\${n.getAttribute("class") || ""}"> "\${name(n)}" [\${Math.round(r.left)},\${Math.round(r.top)} → \${Math.round(r.right)},\${Math.round(r.bottom)}]\`);
    }
  }
  const related = (a, b) => a.n === b.n || a.n.contains(b.n) || b.n.contains(a.n);
  // A text's outline as a convex polygon in page coordinates: an SVG text keeps its rotation, where its
  // client rect would be the rotated text's whole upright bounding box.
  const shapeOf = (item) => {
    if (item.n instanceof SVGGraphicsElement && item.n.getScreenCTM()) {
      const b = item.n.getBBox(), m = item.n.getScreenCTM(), s = 0.16 * item.fs;
      return [[b.x + 1, b.y + s], [b.x + b.width - 1, b.y + s], [b.x + b.width - 1, b.y + b.height - s], [b.x + 1, b.y + b.height - s]]
        .map(([x, y]) => { const p = new DOMPoint(x, y).matrixTransform(m); return [p.x, p.y]; });
    }
    return [[item.l + 1, item.t], [item.r - 1, item.t], [item.r - 1, item.b], [item.l + 1, item.b]];
  };
  const rotated = (n) => { const m = n instanceof SVGGraphicsElement && n.getScreenCTM(); return !!m && (Math.abs(m.b) > 1e-6 || Math.abs(m.c) > 1e-6); };
  // Separating axes: whether two convex polygons overlap by more than a pixel.
  const polysOverlap = (P, Q) => {
    for (const poly of [P, Q]) {
      for (let i = 0; i < poly.length; i++) {
        const [x1, y1] = poly[i], [x2, y2] = poly[(i + 1) % poly.length];
        const len = Math.hypot(x2 - x1, y2 - y1) || 1, nx = (y2 - y1) / len, ny = (x1 - x2) / len;
        const a = P.map(([x, y]) => x * nx + y * ny), b = Q.map(([x, y]) => x * nx + y * ny);
        if (Math.max(...a) < Math.min(...b) + 1 || Math.max(...b) < Math.min(...a) + 1) return false;
      }
    }
    return true;
  };
  for (const a of items) {
    if (a.l < box.l || a.r > box.r || a.t < box.t || a.b > box.b) report(\`MARGIN   "\${name(a.n)}" [\${Math.round(a.l)},\${Math.round(a.t)} → \${Math.round(a.r)},\${Math.round(a.b)}]\`);
    // Text wider than its own element spills into whatever sits beside it.
    const own = a.n instanceof SVGElement ? null : a.n.getBoundingClientRect();
    if (own && getComputedStyle(a.n).display !== "inline" && (a.r > own.right + 1 || a.l < own.left - 1)) report(\`SPILL    "\${name(a.n)}" text \${Math.round(a.l)}–\${Math.round(a.r)} past its box \${Math.round(own.left)}–\${Math.round(own.right)}\`);
  }
  for (let i = 0; i < items.length; i++) {
    for (let j = i + 1; j < items.length; j++) {
      const a = items[i], b = items[j];
      if (related(a, b)) continue;
      if (rotated(a.n) || rotated(b.n)) {
        if (polysOverlap(shapeOf(a), shapeOf(b))) report(\`OVERLAP  "\${name(a.n)}" × "\${name(b.n)}"\`);
        continue;
      }
      const vOverlap = Math.min(a.b, b.b) - Math.max(a.t, b.t);
      const hOverlap = Math.min(a.r, b.r) - Math.max(a.l, b.l);
      if (vOverlap > 1 && hOverlap > 1) { report(\`OVERLAP  "\${name(a.n)}" × "\${name(b.n)}"\`); continue; }
      // Side by side on one line, in different columns: the gap between them must read as a gap.
      const sameLine = vOverlap > 0.5 * Math.min(a.b - a.t, b.b - b.t);
      if (sameLine && a.flow !== b.flow && hOverlap <= 1 && -hOverlap < minGap) report(\`CROWD    "\${name(a.n)}" | "\${name(b.n)}" gap \${Math.round(-hOverlap)} px\`);
    }
  }
  // Lines: no drawn line (connector, curve, ceiling, gridline, hairline rule) may run through text, and no box
  // behind a label may cover one.
  const segments = [];
  for (const n of document.body.querySelectorAll("line, polyline, path")) {
    if (hidden(n) || n.closest("defs, pattern, marker, mask, clipPath")) continue;
    const s = getComputedStyle(n);
    if (s.stroke === "none" || parseFloat(s.strokeWidth) === 0 || s.strokeOpacity === "0") continue;
    const m = n.getScreenCTM();
    if (!m) continue;
    const pt = (x, y) => { const p = new DOMPoint(x, y).matrixTransform(m); return [p.x, p.y]; };
    let pts = [];
    if (n.tagName === "line") pts = [pt(n.x1.baseVal.value, n.y1.baseVal.value), pt(n.x2.baseVal.value, n.y2.baseVal.value)];
    else if (n.tagName === "polyline") pts = [...n.points].map((p) => pt(p.x, p.y));
    else { const len = n.getTotalLength(), k = Math.max(2, Math.ceil(len / 6)); for (let i = 0; i <= k; i++) { const p = n.getPointAtLength((len * i) / k); pts.push(pt(p.x, p.y)); } }
    for (let i = 1; i < pts.length; i++) segments.push({ n, a: pts[i - 1], b: pts[i] });
  }
  // HTML hairlines: a gridline or rule drawn as a 1–2px box with a border or fill.
  for (const n of document.body.querySelectorAll("*")) {
    if (n instanceof SVGElement || hidden(n) || [...n.childNodes].some((c) => c.nodeType === 3 && c.textContent.trim())) continue;
    const r = n.getBoundingClientRect(), s = getComputedStyle(n);
    const painted = opaque(s.backgroundColor) || parseFloat(s.borderLeftWidth) > 0 || parseFloat(s.borderTopWidth) > 0;
    if (!painted) continue;
    if (r.width <= 2.5 && r.height > 6) segments.push({ n, a: [r.left + r.width / 2, r.top], b: [r.left + r.width / 2, r.bottom] });
    else if (r.height <= 2.5 && r.width > 6) segments.push({ n, a: [r.left, r.top + r.height / 2], b: [r.right, r.top + r.height / 2] });
  }
  // Cyrus–Beck: the part of a segment inside a convex polygon, as [t0, t1], or null.
  const clip = ({ a: [x0, y0], b: [x1, y1] }, poly) => {
    const sign = Math.sign(poly.reduce((sum, [px, py], i) => { const [qx, qy] = poly[(i + 1) % poly.length]; return sum + (px * qy - qx * py); }, 0)) || 1;
    let t0 = 0, t1 = 1;
    for (let i = 0; i < poly.length; i++) {
      const [px, py] = poly[i], [qx, qy] = poly[(i + 1) % poly.length];
      const nx = sign * (qy - py), ny = -sign * (qx - px);
      const num = nx * (px - x0) + ny * (py - y0), den = nx * (x1 - x0) + ny * (y1 - y0);
      if (den === 0) { if (num < 0) return null; continue; }
      const t = num / den;
      // The edge normals point outward: a segment heading out (den > 0) leaves at t, one heading in enters at t.
      if (den > 0) t1 = Math.min(t1, t); else t0 = Math.max(t0, t);
      if (t0 > t1) return null;
    }
    return t1 - t0 > 1e-6 ? [t0, t1] : null;
  };
  const probeColor = document.createElement("i");
  probeColor.style.color = "var(--line)";
  document.body.append(probeColor);
  const gridRgb = getComputedStyle(probeColor).color;
  probeColor.remove();
  const lineColor = (n) => {
    const s = getComputedStyle(n);
    if (n instanceof SVGElement) return s.stroke;
    return parseFloat(s.borderLeftWidth) > 0 ? s.borderLeftColor : parseFloat(s.borderTopWidth) > 0 ? s.borderTopColor : s.backgroundColor;
  };
  const kind = (seg) => (lineColor(seg.n) === gridRgb ? "gridline" : \`<\${seg.n.tagName.toLowerCase()}> line\`);
  for (const item of items) {
    const poly = shapeOf(item);
    const hit = segments.find((seg) => !(seg.n === item.n || seg.n.contains(item.n) || item.n.contains(seg.n)) && clip(seg, poly));
    if (hit) report(\`CROSS    a \${kind(hit)} runs through "\${name(item.n)}": move the label off it, or drop the gridlines where labels sit on the plot\`);
  }
  // A box behind a label (the rect drawn just before an SVG label, or an HTML label's own background) is no
  // fix: it cuts a gap into every line it covers, which reads as a broken line.
  const knockouts = new Map();
  const knockoutPolys = new Map();
  for (const item of items) {
    const prev = item.n.previousElementSibling;
    // A rect counts as the label's box only where it wraps the label closely: a region fill or bar that
    // happens to come just before it in the document is not one.
    if (item.n instanceof SVGGraphicsElement && prev && prev.tagName === "rect") {
      const t = item.n.getBBox(), b = prev.getBBox();
      const wraps = b.x <= t.x && b.y <= t.y && b.x + b.width >= t.x + t.width && b.y + b.height >= t.y + t.height;
      if (wraps && b.width - t.width <= 40 && b.height - t.height <= 24) knockouts.set(prev, item.n);
    }
    else if (!(item.n instanceof SVGElement) && opaque(getComputedStyle(item.n).backgroundColor)) knockouts.set(item.n, item.n);
  }
  for (const [k, label] of knockouts) {
    let poly;
    if (k instanceof SVGGraphicsElement) {
      const b = k.getBBox(), m = k.getScreenCTM();
      poly = [[b.x + 1, b.y + 1], [b.x + b.width - 1, b.y + 1], [b.x + b.width - 1, b.y + b.height - 1], [b.x + 1, b.y + b.height - 1]]
        .map(([x, y]) => { const p = new DOMPoint(x, y).matrixTransform(m); return [p.x, p.y]; });
    } else {
      const r = k.getBoundingClientRect();
      poly = [[r.left + 1, r.top + 1], [r.right - 1, r.top + 1], [r.right - 1, r.bottom - 1], [r.left + 1, r.bottom - 1]];
    }
    const cut = segments.find((seg) => (seg.n.compareDocumentPosition(k) & Node.DOCUMENT_POSITION_FOLLOWING) && clip(seg, poly));
    if (cut) report(\`GAP      the box behind "\${name(label)}" cuts a gap into a \${kind(cut)}: move the label off the line, or drop the gridlines where labels sit on the plot\`);
    knockoutPolys.set(label, poly);
  }
  // Dots: no label, nor the box behind it, may sit on a point, ring or marker.
  const dots = [...document.body.querySelectorAll("circle, ellipse")].filter((c) => !hidden(c) && !c.closest("defs, pattern, marker, mask, clipPath"));
  for (const item of items) {
    const poly = knockoutPolys.get(item.n) ?? shapeOf(item);
    const dot = dots.find((c) => {
      const r = c.getBoundingClientRect();
      return polysOverlap(poly, [[r.left, r.top], [r.right, r.top], [r.right, r.bottom], [r.left, r.bottom]]);
    });
    if (dot) report(\`COVERS   "\${name(item.n)}" sits on a <\${dot.tagName}> mark: move the label off it\`);
  }
  const fig = document.querySelector(".figure"), head = document.querySelector("header"), foot = document.querySelector("footer");
  let gaps = "";
  if (fig && head) gaps = \`gap above figure \${Math.round(fig.getBoundingClientRect().top - head.getBoundingClientRect().bottom)} px · below \${Math.round((foot ? foot.getBoundingClientRect().top : H - $bottom) - fig.getBoundingClientRect().bottom)} px\`;
  const out = document.createElement("pre");
  out.textContent = "CHK" + "-BEGIN\n" + (bad.size ? [...bad].join("\n") : "ok") + "\n" + gaps + "\nCHK" + "-END";
  document.body.append(out);
}, 300));
</script>
</body></html>
EOF

result=$("$chrome" --headless=new --disable-gpu --hide-scrollbars --allow-file-access-from-files \
  --window-size="$width,$height" --virtual-time-budget=4000 --dump-dom "file://$probe" 2>/dev/null \
  | sed -n '/CHK-BEGIN/,/CHK-END/p' | sed '1d;$d' | sed 's/&gt;/>/g; s/&lt;/</g; s/&amp;/\&/g; s/&quot;/"/g')

if [[ -z "$result" ]]; then
  echo "The check did not run in Chrome on $page." >&2
  exit 1
fi

echo "$result"
[[ "${result%%$'\n'*}" == "ok" ]]
