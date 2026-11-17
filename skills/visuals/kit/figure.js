/* The figure kit's helpers, shared by every figure page: element builders, the connector arrow, {name}
   placeholders and the zero-based value axis. A classic script, since a module does not load from file://. */

const SVG_NS = "http://www.w3.org/2000/svg";
/** An arrowhead's length along its line and its reach to each side, in CSS px. */
const ARROWHEAD = { len: 8, half: 5 };

function el(tag, className, text) {
  const node = document.createElement(tag);

  if (className !== undefined) {
    node.className = className;
  }

  if (text !== undefined) {
    node.textContent = text;
  }

  return node;
}

function svgEl(tag, attrs, text) {
  const node = document.createElementNS(SVG_NS, tag);

  for (const [key, value] of Object.entries(attrs)) {
    node.setAttribute(key, value);
  }

  if (text !== undefined) {
    node.textContent = text;
  }

  return node;
}

/** Draws into `svg` a straight 2px line from (x1, y1) to (x2, y2) with an open chevron at its end, or at both. */
function arrow(svg, x1, y1, x2, y2, hue, both = false) {
  const head = (x, y, angle) => {
    const bx = x - ARROWHEAD.len * Math.cos(angle);
    const by = y - ARROWHEAD.len * Math.sin(angle);
    const px = ARROWHEAD.half * Math.sin(angle);
    const py = -ARROWHEAD.half * Math.cos(angle);
    svg.append(svgEl("polyline", {
      points: `${bx + px},${by + py} ${x},${y} ${bx - px},${by - py}`,
      fill: "none",
      stroke: hue,
      "stroke-width": 2,
      "stroke-linecap": "round",
      "stroke-linejoin": "round",
    }));
  };
  const angle = Math.atan2(y2 - y1, x2 - x1);
  svg.append(svgEl("line", { x1, y1, x2, y2, stroke: hue, "stroke-width": 2 }));
  head(x2, y2, angle);

  if (both) {
    head(x1, y1, angle + Math.PI);
  }
}

/** Replaces each {name} in the text under `root` with `values[name]`. Raises on a name `values` lacks. */
function fillValues(values, root = document.querySelector(".canvas")) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);

  for (let node = walker.nextNode(); node !== null; node = walker.nextNode()) {
    node.nodeValue = node.nodeValue.replace(/\{(\w+)\}/g, (_, name) => {
      if (!(name in values)) {
        throw new Error(`no value named ${name}`);
      }

      return values[name];
    });
  }
}

/** Rounds off the binary noise a float picks up in step arithmetic, so 0.30000000000000004 prints as 0.3. */
const roundTick = (value) => Number(value.toPrecision(12));

/** The smallest nice step (1, 2, 2.5 or 5 × 10^k) that puts at most `most` steps at or below `max`. */
function niceStep(max, most = 5) {
  const unit = 10 ** Math.floor(Math.log10(max / (most + 1)));

  return [1, 2, 2.5, 5, 10].map((m) => roundTick(m * unit)).find((s) => max / s < most + 1);
}

/** Ticks from zero at niceStep(max, most), 4 to 6 at the default, the last at or short of `max`. */
function niceTicks(max, most = 5) {
  const step = niceStep(max, most);

  return Array.from({ length: Math.floor(max / step + 1e-9) + 1 }, (_, i) => roundTick(i * step));
}

/**
 * A zero-based axis that ends where the data ends: the smallest maximum from `top` up, in steps of `growth` of
 * `top`, for which `fits(max, ticks)` holds on niceTicks(max, most), so a `fits` that draws the chart leaves it
 * drawn on that axis. Returns { max, ticks, digits }, `digits` being the step's decimal places for formatting the
 * ticks. Raises past ten times `top`.
 */
function niceAxis(top, fits = () => true, { most = 5, growth = 0.02 } = {}) {
  for (let grown = 0; grown * growth <= 9; grown += 1) {
    const max = top * (1 + growth * grown);
    const ticks = niceTicks(max, most);

    if (fits(max, ticks)) {
      return { max, ticks, digits: (String(niceStep(max, most)).split(".")[1] ?? "").length };
    }
  }

  throw new Error(`No axis up to ${10 * top} fits every mark and label inside the plot: widen the plot or shorten a label.`);
}

/** Draws an axis into `row`: its line, and at each tick a mark and the tick's value, kept inside the row. */
function drawAxis(row, axis, format) {
  row.append(el("div", "axis"));

  for (const tick of axis.ticks) {
    const at = `${(tick / axis.max) * 100}%`;
    const mark = el("i");
    mark.style.left = `min(${at}, 100% - 1px)`;
    const label = el("span", undefined, format(tick));
    row.append(mark, label);
    label.style.left = `min(${at}, 100% - ${label.offsetWidth / 2}px)`;
  }
}
