/* ------------------------------------------------------------------
   Price chart — hand-drawn SVG, no charting library.
   Reads two JSON blobs the server rendered into the page, so the chart
   costs zero extra requests.
   ------------------------------------------------------------------ */

(function () {
  "use strict";

  const svg = document.getElementById("price-chart");
  const tooltip = document.getElementById("chart-tooltip");
  const dataNode = document.getElementById("chart-data");
  if (!svg || !dataNode) return;

  const SERIES = JSON.parse(dataNode.textContent || "[]");
  const predNode = document.getElementById("prediction-data");
  const PREDICTION = predNode ? JSON.parse(predNode.textContent || "null") : null;

  if (!SERIES.length) return;

  const NS = "http://www.w3.org/2000/svg";
  const W = 900, H = 340;
  const M = { top: 16, right: 74, bottom: 30, left: 58 };
  const PLOT_W = W - M.left - M.right;
  const PLOT_H = H - M.top - M.bottom;

  const UP = "#2ecc8f", DOWN = "#f2635f";

  let visible = SERIES;          // current slice after range filtering
  let scaleX = null, scaleY = null;

  const el = (name, attrs) => {
    const node = document.createElementNS(NS, name);
    for (const key in attrs) node.setAttribute(key, attrs[key]);
    return node;
  };

  const money = (n) =>
    "$" + n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

  const shortDate = (iso) => {
    const [y, m, d] = iso.split("-");
    const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    return `${months[+m - 1]} ${+d}`;
  };

  const longDate = (iso) => {
    const [y, m, d] = iso.split("-");
    return `${shortDate(iso)} ${y}`;
  };

  /* ---------- drawing ---------- */

  function render() {
    while (svg.firstChild) svg.removeChild(svg.firstChild);

    // The forecast occupies one extra slot to the right of the last bar.
    const showPrediction = PREDICTION !== null;
    const slots = visible.length + (showPrediction ? 1 : 0);

    let lo = Math.min(...visible.map((p) => p.c));
    let hi = Math.max(...visible.map((p) => p.c));
    if (showPrediction) {
      lo = Math.min(lo, PREDICTION.interval_low);
      hi = Math.max(hi, PREDICTION.interval_high);
    }
    const pad = (hi - lo) * 0.08 || Math.max(hi * 0.02, 0.5);
    lo -= pad;
    hi += pad;

    scaleX = (i) => M.left + (slots === 1 ? PLOT_W / 2 : (i / (slots - 1)) * PLOT_W);
    scaleY = (v) => M.top + PLOT_H - ((v - lo) / (hi - lo)) * PLOT_H;

    const rising = visible[visible.length - 1].c >= visible[0].c;
    const stroke = rising ? UP : DOWN;
    const gradientId = "area-gradient";

    // --- gradient under the line ---
    const defs = el("defs", {});
    const gradient = el("linearGradient", {
      id: gradientId, x1: "0", y1: "0", x2: "0", y2: "1",
    });
    gradient.appendChild(el("stop", { offset: "0%", "stop-color": stroke, "stop-opacity": ".28" }));
    gradient.appendChild(el("stop", { offset: "100%", "stop-color": stroke, "stop-opacity": "0" }));
    defs.appendChild(gradient);
    svg.appendChild(defs);

    // --- horizontal gridlines + price axis ---
    const TICKS = 5;
    for (let t = 0; t <= TICKS; t++) {
      const value = lo + ((hi - lo) * t) / TICKS;
      const y = scaleY(value);
      svg.appendChild(el("line", {
        class: "chart-grid-line", x1: M.left, y1: y, x2: M.left + PLOT_W, y2: y,
      }));
      const label = el("text", {
        class: "chart-axis-text", x: M.left - 8, y: y + 4, "text-anchor": "end",
      });
      label.textContent = value.toFixed(2);
      svg.appendChild(label);
    }

    // --- date axis ---
    const labelCount = Math.min(6, visible.length);
    for (let t = 0; t < labelCount; t++) {
      const i = Math.round((t / Math.max(labelCount - 1, 1)) * (visible.length - 1));
      const label = el("text", {
        class: "chart-axis-text",
        x: scaleX(i),
        y: H - 10,
        "text-anchor": t === 0 ? "start" : t === labelCount - 1 ? "middle" : "middle",
      });
      label.textContent = shortDate(visible[i].d);
      svg.appendChild(label);
    }

    // --- area + line ---
    const points = visible.map((p, i) => [scaleX(i), scaleY(p.c)]);
    const linePath = points.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)}`).join("");
    const baseY = M.top + PLOT_H;

    svg.appendChild(el("path", {
      d: `${linePath}L${points[points.length - 1][0].toFixed(2)},${baseY}L${points[0][0].toFixed(2)},${baseY}Z`,
      fill: `url(#${gradientId})`,
      stroke: "none",
    }));
    svg.appendChild(el("path", { class: "chart-line", d: linePath, stroke: stroke }));

    // --- the forecast ---
    if (showPrediction) {
      const px = scaleX(slots - 1);
      const bandTop = scaleY(PREDICTION.interval_high);
      const bandBottom = scaleY(PREDICTION.interval_low);
      const last = points[points.length - 1];

      // 80% error band, drawn as a capped vertical bar.
      svg.appendChild(el("rect", {
        class: "chart-pred-band",
        x: px - 9, y: bandTop, width: 18,
        height: Math.max(bandBottom - bandTop, 1), rx: 3,
      }));
      svg.appendChild(el("path", {
        class: "chart-pred-line",
        d: `M${last[0].toFixed(2)},${last[1].toFixed(2)}L${px.toFixed(2)},${scaleY(PREDICTION.predicted_close).toFixed(2)}`,
      }));
      svg.appendChild(el("circle", {
        cx: px, cy: scaleY(PREDICTION.predicted_close), r: 4.5,
        fill: "#5b9dff", stroke: "#0b0f16", "stroke-width": 2,
      }));

      const tag = el("text", {
        class: "chart-axis-text",
        x: px + 12,
        y: scaleY(PREDICTION.predicted_close) + 4,
        "text-anchor": "start",
        fill: "#5b9dff",
      });
      tag.textContent = money(PREDICTION.predicted_close);
      svg.appendChild(tag);
    }

    // --- last actual close ---
    const last = points[points.length - 1];
    svg.appendChild(el("circle", {
      cx: last[0], cy: last[1], r: 3.5, fill: stroke,
      stroke: "#0b0f16", "stroke-width": 2,
    }));

    // --- hover layer ---
    const crosshair = el("line", {
      class: "chart-crosshair", x1: 0, y1: M.top, x2: 0, y2: M.top + PLOT_H, opacity: "0",
    });
    const marker = el("circle", {
      r: 4, fill: stroke, stroke: "#0b0f16", "stroke-width": 2, opacity: "0",
    });
    svg.appendChild(crosshair);
    svg.appendChild(marker);

    svg.appendChild(el("rect", {
      x: M.left, y: M.top, width: PLOT_W, height: PLOT_H,
      fill: "transparent", "pointer-events": "all", id: "hover-target",
    }));

    attachHover(crosshair, marker);
  }

  /* ---------- interaction ---------- */

  function attachHover(crosshair, marker) {
    const target = svg.querySelector("#hover-target");
    if (!target) return;

    target.addEventListener("mousemove", (event) => {
      const rect = svg.getBoundingClientRect();
      const scale = rect.width / W;                 // SVG units -> screen px
      const svgX = (event.clientX - rect.left) / scale;

      // Nearest data point to the cursor.
      const slots = visible.length + (PREDICTION ? 1 : 0);
      const ratio = (svgX - M.left) / PLOT_W;
      let i = Math.round(ratio * (slots - 1));
      i = Math.max(0, Math.min(visible.length - 1, i));

      const point = visible[i];
      const x = scaleX(i);
      const y = scaleY(point.c);

      crosshair.setAttribute("x1", x);
      crosshair.setAttribute("x2", x);
      crosshair.setAttribute("opacity", "1");
      marker.setAttribute("cx", x);
      marker.setAttribute("cy", y);
      marker.setAttribute("opacity", "1");

      const previous = i > 0 ? visible[i - 1].c : point.c;
      const delta = point.c - previous;
      const pct = previous ? (delta / previous) * 100 : 0;
      const colour = delta >= 0 ? UP : DOWN;

      tooltip.innerHTML =
        `<div class="tt-date">${longDate(point.d)}</div>` +
        `<div><strong>${money(point.c)}</strong> ` +
        `<span style="color:${colour}">${delta >= 0 ? "+" : ""}${delta.toFixed(2)} ` +
        `(${pct >= 0 ? "+" : ""}${pct.toFixed(2)}%)</span></div>`;
      tooltip.hidden = false;
      tooltip.style.left = `${x * scale}px`;
      tooltip.style.top = `${y * scale}px`;
    });

    target.addEventListener("mouseleave", () => {
      crosshair.setAttribute("opacity", "0");
      marker.setAttribute("opacity", "0");
      tooltip.hidden = true;
    });
  }

  /* ---------- range buttons ---------- */

  const buttons = document.getElementById("range-buttons");
  if (buttons) {
    buttons.addEventListener("click", (event) => {
      const button = event.target.closest("button");
      if (!button) return;

      const days = parseInt(button.dataset.days, 10);
      visible = days > 0 ? SERIES.slice(-days) : SERIES;
      if (visible.length < 2) visible = SERIES;

      buttons.querySelectorAll("button").forEach((b) => b.classList.remove("active"));
      button.classList.add("active");
      render();
    });
  }

  window.addEventListener("resize", () => { tooltip.hidden = true; });

  render();
})();
