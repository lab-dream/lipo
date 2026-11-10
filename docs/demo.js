/* LiPo interactive demo — runs the actual pipeline in the browser.
   chunk scheduling  ->  linear blending over the overlap  ->  jerk minimization
   inside a bounded perturbation space.                                        */
(function () {
  var root = document.getElementById('demo');
  if (!root) return;

  var N = 240, DT = 1 / 30;
  var SV = document.getElementById('demo-svg');
  var gPos = document.getElementById('demo-pos');
  var gVel = document.getElementById('demo-vel');
  var out = document.getElementById('demo-out');
  var seed = 7;

  var ctrl = {
    H: document.getElementById('d-h'),
    ov: document.getElementById('d-ov'),
    delay: document.getElementById('d-delay'),
    noise: document.getElementById('d-noise'),
    eps: document.getElementById('d-eps')
  };

  function rng(s) {
    return function () {
      s |= 0; s = s + 0x6D2B79F5 | 0;
      var t = Math.imul(s ^ s >>> 15, 1 | s);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  function gauss(r) {
    var u = Math.max(1e-9, r()), v = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // the motion the policy is trying to produce
  var ref = new Float64Array(N);
  for (var i = 0; i < N; i++) {
    var t = i * DT;
    ref[i] = 0.6 * Math.sin(2 * Math.PI * 0.45 * t) + 0.25 * Math.sin(2 * Math.PI * 0.9 * t + 1.0);
  }

  // each chunk is predicted on its own, so consecutive chunks disagree a little
  function makeChunks(H, overlap, noise, r) {
    var step = Math.max(1, H - overlap), segs = [], s = 0;
    while (s < N) {
      var L = Math.min(H, N - s);
      var bias = gauss(r) * noise, drift = gauss(r) * noise * 0.5;
      var seg = new Float64Array(L);
      for (var k = 0; k < L; k++) seg[k] = ref[s + k] + bias + drift * (L > 1 ? k / (L - 1) : 0);
      segs.push({ s: s, q: seg });
      s += step;
    }
    return { segs: segs, step: step };
  }

  // no blending: play each chunk until the next one is ready, holding still while it computes
  function rawExec(c, delay) {
    var q = new Float64Array(N), filled = new Uint8Array(N);
    for (var i = 0; i < c.segs.length; i++) {
      var sg = c.segs[i], e = Math.min(sg.s + c.step, N), k;
      for (k = sg.s; k < e; k++) { q[k] = sg.q[k - sg.s]; filled[k] = 1; }
      var hold = q[e - 1];
      for (k = e; k < Math.min(e + delay, N); k++) { q[k] = hold; filled[k] = 1; }
    }
    for (var j = 0; j < N; j++) if (!filled[j]) q[j] = ref[j];
    return q;
  }

  // step 2 — linear blending across the overlapping region
  function blend(c, overlap) {
    var num = new Float64Array(N), den = new Float64Array(N);
    for (var i = 0; i < c.segs.length; i++) {
      var sg = c.segs[i], L = sg.q.length, r = Math.min(overlap, L);
      for (var k = 0; k < L; k++) {
        var w = 1;
        if (r > 1) {
          if (k < r) w = k / (r - 1);
          if (k >= L - r) w = Math.min(w, (L - 1 - k) / (r - 1));
        }
        num[sg.s + k] += w * sg.q[k];
        den[sg.s + k] += w;
      }
    }
    var q = new Float64Array(N);
    for (var j = 0; j < N; j++) q[j] = den[j] > 1e-9 ? num[j] / den[j] : ref[j];
    return q;
  }

  // step 3 — minimize jerk without straying further than eps from the blended trajectory
  function jerkMin(q0, eps, iters) {
    var q = Float64Array.from(q0), g = new Float64Array(N), lr = 0.004, it, i;
    for (it = 0; it < iters; it++) {
      g.fill(0);
      for (i = 0; i + 3 < N; i++) {
        var r = q[i + 3] - 3 * q[i + 2] + 3 * q[i + 1] - q[i];
        g[i] += -2 * r; g[i + 1] += 6 * r; g[i + 2] += -6 * r; g[i + 3] += 2 * r;
      }
      for (i = 0; i < N; i++) {
        var v = q[i] - lr * g[i], lo = q0[i] - eps, hi = q0[i] + eps;
        q[i] = v < lo ? lo : (v > hi ? hi : v);
      }
    }
    return q;
  }

  function vel(q) {
    var v = new Float64Array(N - 1);
    for (var i = 0; i < N - 1; i++) v[i] = (q[i + 1] - q[i]) / DT;
    return v;
  }
  function maxAccelJump(q) {
    var v = vel(q), m = 0;
    for (var i = 0; i < v.length - 1; i++) m = Math.max(m, Math.abs(v[i + 1] - v[i]));
    return m;
  }
  function maxJerk(q) {
    var m = 0;
    for (var i = 0; i + 3 < N; i++)
      m = Math.max(m, Math.abs((q[i + 3] - 3 * q[i + 2] + 3 * q[i + 1] - q[i]) / (DT * DT * DT)));
    return m;
  }

  // ---- drawing -------------------------------------------------------------
  var PAD = { l: 16, r: 14, t: 16, b: 14 }, W = 920, HP = 150, HV = 120, GAP = 30;

  function path(arr, x0, y0, h, lo, hi) {
    var n = arr.length, d = '', i, x, y;
    for (i = 0; i < n; i++) {
      x = x0 + (W - PAD.l - PAD.r) * i / (n - 1);
      y = y0 + h - h * (arr[i] - lo) / (hi - lo);
      d += (i ? 'L' : 'M') + x.toFixed(1) + ' ' + y.toFixed(1);
    }
    return d;
  }
  function el(tag, attrs) {
    var n = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    return n;
  }
  function range(arrs) {
    var lo = Infinity, hi = -Infinity;
    arrs.forEach(function (a) { for (var i = 0; i < a.length; i++) { if (a[i] < lo) lo = a[i]; if (a[i] > hi) hi = a[i]; } });
    var pad = (hi - lo) * 0.12 || 1;
    return [lo - pad, hi + pad];
  }

  function render() {
    var H = +ctrl.H.value, ov = Math.min(+ctrl.ov.value, Math.floor(H / 2) - 1);
    var delay = +ctrl.delay.value, noise = +ctrl.noise.value / 1000, eps = +ctrl.eps.value / 1000;

    var rawC = makeChunks(H, 0, noise, rng(seed));
    var raw = rawExec(rawC, delay);
    var lipoC = makeChunks(H, Math.max(ov, 1), noise, rng(seed));
    var bl = blend(lipoC, Math.max(ov, 1));
    var lipo = jerkMin(bl, eps, 320);

    gPos.innerHTML = ''; gVel.innerHTML = '';
    var y0 = PAD.t, y1 = PAD.t + HP + GAP;

    var pr = range([raw, lipo, ref]);
    gPos.appendChild(el('rect', { x: PAD.l, y: y0, width: W - PAD.l - PAD.r, height: HP, fill: '#fffefb', stroke: '#e4e0d9' }));
    gPos.appendChild(el('path', { d: path(ref, PAD.l, y0, HP, pr[0], pr[1]), fill: 'none', stroke: '#c9c5bd', 'stroke-width': 1.5, 'stroke-dasharray': '4 4' }));
    gPos.appendChild(el('path', { d: path(raw, PAD.l, y0, HP, pr[0], pr[1]), fill: 'none', stroke: '#c8372f', 'stroke-width': 1.8, 'stroke-opacity': .85 }));
    gPos.appendChild(el('path', { d: path(lipo, PAD.l, y0, HP, pr[0], pr[1]), fill: 'none', stroke: '#2f6fd0', 'stroke-width': 2.2 }));

    var vr = vel(raw), vl = vel(lipo), vv = range([vr, vl]);
    gVel.appendChild(el('rect', { x: PAD.l, y: y1, width: W - PAD.l - PAD.r, height: HV, fill: '#fffefb', stroke: '#e4e0d9' }));
    gVel.appendChild(el('path', { d: path(vr, PAD.l, y1, HV, vv[0], vv[1]), fill: 'none', stroke: '#c8372f', 'stroke-width': 1.6, 'stroke-opacity': .85 }));
    gVel.appendChild(el('path', { d: path(vl, PAD.l, y1, HV, vv[0], vv[1]), fill: 'none', stroke: '#2f6fd0', 'stroke-width': 2 }));

    // chunk seams
    var s = 0;
    while (s < N) {
      var x = PAD.l + (W - PAD.l - PAD.r) * s / (N - 1);
      gPos.appendChild(el('line', { x1: x, y1: y0, x2: x, y2: y0 + HP, stroke: '#e0dbd2', 'stroke-width': 1 }));
      s += rawC.step;
    }

    function lab(x, y, txt, fill, anchor) {
      var n = el('text', { x: x, y: y, fill: fill, 'font-size': 11.5, 'font-family': 'Inter, sans-serif', 'text-anchor': anchor || 'start' });
      n.textContent = txt; return n;
    }
    gPos.appendChild(lab(PAD.l + 10, y0 + 17, 'position', '#9a968e'));
    gVel.appendChild(lab(PAD.l + 10, y1 + 17, 'velocity', '#9a968e'));

    var aR = maxAccelJump(raw), aL = maxAccelJump(lipo);
    var jR = maxJerk(raw), jL = maxJerk(lipo);
    var dR = 0, i;
    for (i = 0; i < N; i++) dR = Math.max(dR, Math.abs(lipo[i] - ref[i]));

    function pct(a, b) { return a > 0 ? (100 * (1 - b / a)).toFixed(0) + '% lower' : '—'; }
    out.innerHTML =
      '<div class="stat"><span class="k">Peak acceleration step</span>' +
      '<span class="v v--red">' + aR.toFixed(1) + '</span><span class="arrow">&rarr;</span>' +
      '<span class="v v--blue">' + aL.toFixed(1) + '</span><span class="d">' + pct(aR, aL) + '</span></div>' +
      '<div class="stat"><span class="k">Peak jerk</span>' +
      '<span class="v v--red">' + jR.toFixed(0) + '</span><span class="arrow">&rarr;</span>' +
      '<span class="v v--blue">' + jL.toFixed(0) + '</span><span class="d">' + pct(jR, jL) + '</span></div>' +
      '<div class="stat"><span class="k">Deviation from the policy</span>' +
      '<span class="v v--blue">' + dR.toFixed(3) + '</span><span class="d">bounded by the &epsilon; you set</span></div>';

    renderSweep(lipoC, Math.max(ov, 1), eps);

    document.getElementById('d-hv').textContent = H;
    document.getElementById('d-ovv').textContent = Math.max(ov, 1);
    document.getElementById('d-delayv').textContent = delay;
    document.getElementById('d-noisev').textContent = (noise * 1000).toFixed(0);
    document.getElementById('d-epsv').textContent = (eps * 1000).toFixed(0);
  }

  // ---- epsilon sweep: how much smoothing does a wider bound buy you? --------
  var gSweep = document.getElementById('demo-sweep-g');
  var SW = 920, SH = 150, SP = { l: 64, r: 70, t: 16, b: 34 };

  function renderSweep(chunkSet, overlap, curEps) {
    if (!gSweep) return;
    var bl = blend(chunkSet, overlap);
    var EPS_MAX = 0.03, M = 15, pts = [], i;
    for (i = 0; i < M; i++) {
      var e = EPS_MAX * i / (M - 1);
      var q = e > 0 ? jerkMin(bl, e, 200) : bl;
      var dev = 0;
      for (var k = 0; k < N; k++) dev = Math.max(dev, Math.abs(q[k] - bl[k]));
      pts.push({ e: e, j: maxJerk(q), dev: dev });
    }
    var jMax = 0;
    pts.forEach(function (p) { if (p.j > jMax) jMax = p.j; });

    var x0 = SP.l, x1 = SW - SP.r, y0 = SP.t, y1 = SH - SP.b;
    var X = function (e) { return x0 + (x1 - x0) * e / EPS_MAX; };
    var YJ = function (j) { return y1 - (y1 - y0) * j / (jMax || 1); };
    var YD = function (d) { return y1 - (y1 - y0) * d / EPS_MAX; };

    gSweep.innerHTML = '';
    gSweep.appendChild(el('rect', { x: x0, y: y0, width: x1 - x0, height: y1 - y0, fill: '#fffefb', stroke: '#e4e0d9' }));

    var dj = '', dd = '';
    pts.forEach(function (p, n) {
      dj += (n ? 'L' : 'M') + X(p.e).toFixed(1) + ' ' + YJ(p.j).toFixed(1);
      dd += (n ? 'L' : 'M') + X(p.e).toFixed(1) + ' ' + YD(p.dev).toFixed(1);
    });
    gSweep.appendChild(el('path', { d: dd, fill: 'none', stroke: '#b9b5ac', 'stroke-width': 1.6, 'stroke-dasharray': '5 4' }));
    gSweep.appendChild(el('path', { d: dj, fill: 'none', stroke: '#2f6fd0', 'stroke-width': 2.4 }));

    var cx = X(Math.min(curEps, EPS_MAX));
    gSweep.appendChild(el('line', { x1: cx, y1: y0, x2: cx, y2: y1, stroke: '#1b1d1f', 'stroke-width': 1.2, 'stroke-dasharray': '3 3' }));
    var near = pts.reduce(function (a, b) { return Math.abs(b.e - curEps) < Math.abs(a.e - curEps) ? b : a; });
    gSweep.appendChild(el('circle', { cx: cx, cy: YJ(near.j), r: 4.5, fill: '#2f6fd0' }));

    function tx(x, y, t, fill, anchor, size) {
      var n = el('text', { x: x, y: y, fill: fill, 'font-size': size || 11.5, 'font-family': 'Inter, sans-serif', 'text-anchor': anchor || 'start' });
      n.textContent = t; return n;
    }
    gSweep.appendChild(tx(x0 - 8, y0 + 11, 'peak jerk', '#2f6fd0', 'end'));
    gSweep.appendChild(tx(x1 + 8, y0 + 11, 'deviation', '#8b877f'));
    gSweep.appendChild(tx(x0, y1 + 20, '0', '#6b6e72'));
    gSweep.appendChild(tx((x0 + x1) / 2, y1 + 20, 'bound  ε  (mm)', '#6b6e72', 'middle'));
    gSweep.appendChild(tx(x1, y1 + 20, '30', '#6b6e72', 'end'));
    gSweep.appendChild(tx(cx + 6, y1 - 6, 'ε = ' + (curEps * 1000).toFixed(0), '#1b1d1f'));
  }

  // drag anywhere on the sweep chart to set the bound
  var sweepSvg = document.getElementById('demo-sweep');
  if (sweepSvg) {
    var dragging = false;
    var setFromX = function (clientX) {
      var r = sweepSvg.getBoundingClientRect();
      var vx = (clientX - r.left) * (SW / r.width);
      var f = (vx - SP.l) / (SW - SP.r - SP.l);
      f = f < 0 ? 0 : (f > 1 ? 1 : f);
      ctrl.eps.value = Math.round(f * 30);
      render();
    };
    sweepSvg.addEventListener('pointerdown', function (e) {
      dragging = true; sweepSvg.setPointerCapture(e.pointerId); setFromX(e.clientX);
    });
    sweepSvg.addEventListener('pointermove', function (e) { if (dragging) setFromX(e.clientX); });
    sweepSvg.addEventListener('pointerup', function () { dragging = false; });
    sweepSvg.addEventListener('pointercancel', function () { dragging = false; });
  }

  Object.keys(ctrl).forEach(function (k) { ctrl[k].addEventListener('input', render); });
  document.getElementById('d-reseed').addEventListener('click', function () {
    seed = (seed * 7919 + 13) % 100000; render();
  });
  render();
})();
