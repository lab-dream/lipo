/* LiPo interactive demo — the pipeline of the paper, run in the browser.
   Reference construction follows Eq. (4)-(5): the trajectory is frozen through the
   inference delay, linearly blended over the blending horizon, then handed to the new
   chunk. The refinement follows Eq. (6): minimize jerk over a perturbation eps(t) that
   is zero in the delay window, bounded by eps_b inside the blending zone and by the
   tighter eps_p everywhere else.                                                      */
(function () {
  if (!document.getElementById('demo')) return;

  var N = 300, DT = 1 / 30;
  var gBack = document.getElementById('demo-back');
  var gPos = document.getElementById('demo-pos');
  var gVel = document.getElementById('demo-vel');
  var out = document.getElementById('demo-out');
  var seed = 7;

  var ctrl = {
    stride: document.getElementById('d-stride'),
    delay: document.getElementById('d-delay'),
    blend: document.getElementById('d-blend'),
    noise: document.getElementById('d-noise'),
    eb: document.getElementById('d-eb'),
    ep: document.getElementById('d-ep')
  };

  // chunk colours, cycled so neighbouring chunks never share one
  var CHUNK = ['#ffb454', '#4fd1c5', '#c79bff', '#7fd36b', '#ff8da1'];

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

  var ref = new Float64Array(N);
  for (var i0 = 0; i0 < N; i0++) {
    var tt = i0 * DT;
    ref[i0] = 0.6 * Math.sin(2 * Math.PI * 0.42 * tt) + 0.22 * Math.sin(2 * Math.PI * 0.95 * tt + 1.0);
  }

  // every chunk is inferred on its own, so neighbours disagree a little
  function makeChunks(H, stride, noise, r) {
    var segs = [], s = 0;
    while (s < N) {
      var L = Math.min(H, N - s);
      var bias = gauss(r) * noise, drift = gauss(r) * noise * 0.6;
      var q = new Float64Array(L);
      for (var k = 0; k < L; k++) q[k] = ref[s + k] + bias + drift * (L > 1 ? k / (L - 1) : 0);
      segs.push({ s: s, q: q });
      s += stride;
    }
    return segs;
  }

  /* Eq. (4)-(5). Returns the reference plus the zone map used by Eq. (6):
     zone 0 = frozen (delay), 1 = blending, 2 = path.                        */
  function buildReference(segs, stride, td, tb) {
    var q = new Float64Array(N), zone = new Uint8Array(N), i, k;
    for (i = 0; i < segs[0].q.length && i < N; i++) { q[i] = segs[0].q[i]; zone[i] = 2; }

    for (i = 1; i < segs.length; i++) {
      var sNew = segs[i].s, qNew = segs[i].q;
      var dEnd = Math.min(sNew + td, N);          // frozen through the inference delay
      for (k = sNew; k < dEnd; k++) zone[k] = 0;  // q stays whatever the previous chunk left

      var bEnd = Math.min(sNew + tb, N);          // linear blending over (td, tb]
      for (k = dEnd; k < bEnd; k++) {
        var a = tb > td ? (k - sNew - td) / (tb - td) : 1;
        q[k] = (1 - a) * q[k] + a * qNew[k - sNew];
        zone[k] = 1;
      }
      var end = Math.min(sNew + qNew.length, N);  // then the new chunk alone
      for (k = bEnd; k < end; k++) { q[k] = qNew[k - sNew]; zone[k] = 2; }
    }
    return { q: q, zone: zone };
  }

  // what a controller would track with no post-optimization: switch chunks outright,
  // and hold still while the next one is being inferred
  function rawExec(segs, stride, td) {
    var q = new Float64Array(N), i, k;
    for (i = 0; i < segs[0].q.length && i < N; i++) q[i] = segs[0].q[i];
    for (i = 1; i < segs.length; i++) {
      var s = segs[i].s, hold = q[Math.max(0, s - 1)];
      for (k = s; k < Math.min(s + td, N); k++) q[k] = hold;
      for (k = Math.min(s + td, N); k < Math.min(s + segs[i].q.length, N); k++) q[k] = segs[i].q[k - s];
    }
    return q;
  }

  /* Eq. (6): minimize the jerk of q_ref + eps subject to the per-zone bounds,
     by projected gradient descent.                                           */
  function jerkMin(q0, zone, eb, ep, iters) {
    var q = Float64Array.from(q0), g = new Float64Array(N);
    var lo = new Float64Array(N), hi = new Float64Array(N), i, it;
    for (i = 0; i < N; i++) {
      var e = zone[i] === 0 ? 0 : (zone[i] === 1 ? eb : ep);
      lo[i] = q0[i] - e; hi[i] = q0[i] + e;
    }
    for (it = 0; it < iters; it++) {
      g.fill(0);
      for (i = 0; i + 3 < N; i++) {
        var r = q[i + 3] - 3 * q[i + 2] + 3 * q[i + 1] - q[i];
        g[i] += -2 * r; g[i + 1] += 6 * r; g[i + 2] += -6 * r; g[i + 3] += 2 * r;
      }
      for (i = 0; i < N; i++) {
        var v = q[i] - 0.004 * g[i];
        q[i] = v < lo[i] ? lo[i] : (v > hi[i] ? hi[i] : v);
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
  var W = 920, PAD = { l: 14, r: 14, t: 14 }, HP = 190, HV = 110, GAP = 26, HZ = 150;
  var NS = 'http://www.w3.org/2000/svg';
  function el(tag, a) { var n = document.createElementNS(NS, tag); for (var k in a) n.setAttribute(k, a[k]); return n; }
  function X(i) { return PAD.l + (W - PAD.l - PAD.r) * i / (N - 1); }

  function line(arr, y0, h, lo, hi, from, to) {
    var d = '', i, first = true;
    for (i = from; i <= to; i++) {
      if (arr[i] === undefined) continue;
      var y = y0 + h - h * (arr[i] - lo) / (hi - lo);
      d += (first ? 'M' : 'L') + X(i).toFixed(1) + ' ' + y.toFixed(1);
      first = false;
    }
    return d;
  }
  function range(arrs) {
    var lo = Infinity, hi = -Infinity;
    arrs.forEach(function (a) { for (var i = 0; i < a.length; i++) { if (a[i] < lo) lo = a[i]; if (a[i] > hi) hi = a[i]; } });
    var p = (hi - lo) * 0.14 || 1;
    return [lo - p, hi + p];
  }

  function render() {
    var stride = +ctrl.stride.value, td = +ctrl.delay.value, bh = +ctrl.blend.value;
    var noise = +ctrl.noise.value / 1000, eb = +ctrl.eb.value / 1000, ep = +ctrl.ep.value / 1000;
    var tb = td + bh, H = stride + tb + 6;

    var segs = makeChunks(H, stride, noise, rng(seed));
    var raw = rawExec(segs, stride, td);
    var built = buildReference(segs, stride, td, tb);
    var lipo = jerkMin(built.q, built.zone, eb, ep, 320);

    gBack.innerHTML = ''; gPos.innerHTML = ''; gVel.innerHTML = '';
    var y0 = PAD.t, y1 = PAD.t + HP + GAP;
    var pr = range([raw, built.q, ref]);
    var toY = function (v) { return y0 + HP - HP * (v - pr[0]) / (pr[1] - pr[0]); };

    gBack.appendChild(el('rect', { x: PAD.l, y: y0, width: W - PAD.l - PAD.r, height: HP, fill: '#171a1d', rx: 6 }));
    gBack.appendChild(el('rect', { x: PAD.l, y: y1, width: W - PAD.l - PAD.r, height: HV, fill: '#171a1d', rx: 6 }));

    // zone bands — light blue for the frozen delay, orange for the blending horizon (Fig. 2)
    var i, k;
    for (i = 1; i < segs.length; i++) {
      var s = segs[i].s;
      if (s >= N) break;
      var xd0 = X(s), xd1 = X(Math.min(s + td, N - 1)), xb1 = X(Math.min(s + tb, N - 1));
      if (td > 0) gBack.appendChild(el('rect', { x: xd0, y: y0, width: Math.max(1, xd1 - xd0), height: HP, fill: '#8ab4ff', 'fill-opacity': .16 }));
      if (bh > 0) gBack.appendChild(el('rect', { x: xd1, y: y0, width: Math.max(1, xb1 - xd1), height: HP, fill: '#ffb454', 'fill-opacity': .17 }));
    }

    // the epsilon envelope around q_ref, drawn as a band whose width changes by zone
    var up = '', dn = '';
    for (i = 0; i < N; i++) {
      var e = built.zone[i] === 0 ? 0 : (built.zone[i] === 1 ? eb : ep);
      up += (i ? 'L' : 'M') + X(i).toFixed(1) + ' ' + toY(built.q[i] + e).toFixed(1);
    }
    for (i = N - 1; i >= 0; i--) {
      var e2 = built.zone[i] === 0 ? 0 : (built.zone[i] === 1 ? eb : ep);
      dn += 'L' + X(i).toFixed(1) + ' ' + toY(built.q[i] - e2).toFixed(1);
    }
    gPos.appendChild(el('path', { d: up + dn + 'Z', fill: '#ffffff', 'fill-opacity': .20, stroke: 'none' }));

    // each chunk as its own line, so the overlaps are visible
    for (i = 0; i < segs.length; i++) {
      var sg = segs[i], arr = new Array(N), L = sg.q.length;
      for (k = 0; k < L && sg.s + k < N; k++) arr[sg.s + k] = sg.q[k];
      gPos.appendChild(el('path', {
        d: line(arr, y0, HP, pr[0], pr[1], sg.s, Math.min(sg.s + L - 1, N - 1)),
        fill: 'none', stroke: CHUNK[i % CHUNK.length], 'stroke-width': 1.6, 'stroke-opacity': .95
      }));
    }

    // the executed result: white, on top
    gPos.appendChild(el('path', { d: line(lipo, y0, HP, pr[0], pr[1], 0, N - 1), fill: 'none', stroke: '#ffffff', 'stroke-width': 2.6 }));

    var vr = vel(raw), vl = vel(lipo), vv = range([vr, vl]);
    gVel.appendChild(el('path', { d: line(vr, y1, HV, vv[0], vv[1], 0, vr.length - 1), fill: 'none', stroke: '#ff6b6b', 'stroke-width': 1.5, 'stroke-opacity': .9 }));
    gVel.appendChild(el('path', { d: line(vl, y1, HV, vv[0], vv[1], 0, vl.length - 1), fill: 'none', stroke: '#ffffff', 'stroke-width': 2 }));

    function tx(x, y, t, fill, size, anchor) {
      var n = el('text', { x: x, y: y, fill: fill, 'font-size': size || 11.5, 'font-family': 'Inter, sans-serif', 'text-anchor': anchor || 'start' });
      n.textContent = t; return n;
    }
    gPos.appendChild(tx(PAD.l + 12, y0 + 20, 'position — one line per inferred chunk, white is what gets executed', '#8e9299'));
    gVel.appendChild(tx(PAD.l + 12, y1 + 18, 'velocity — red: raw chunk switching, white: after LiPo', '#8e9299'));

    var aR = maxAccelJump(raw), aL = maxAccelJump(lipo);
    var jR = maxJerk(raw), jL = maxJerk(lipo);
    var dev = 0;
    for (i = 0; i < N; i++) dev = Math.max(dev, Math.abs(lipo[i] - built.q[i]));
    function pct(a, b) { if (!(a > 0)) return '\u2014'; var v = 100 * (1 - b / a); return (v > 99 ? v.toFixed(1) : v.toFixed(0)) + '% lower'; }

    out.innerHTML =
      '<div class="stat"><span class="k">Peak acceleration step</span><span class="v v--red">' + aR.toFixed(1) +
      '</span><span class="arrow">&rarr;</span><span class="v v--blue">' + aL.toFixed(1) + '</span><span class="d">' + pct(aR, aL) + '</span></div>' +
      '<div class="stat"><span class="k">Peak jerk</span><span class="v v--red">' + jR.toFixed(0) +
      '</span><span class="arrow">&rarr;</span><span class="v v--blue">' + jL.toFixed(0) + '</span><span class="d">' + pct(jR, jL) + '</span></div>' +
      '<div class="stat"><span class="k">Largest perturbation used</span><span class="v v--blue">' + dev.toFixed(4) +
      '</span><span class="d">rad &middot; bounded by &epsilon;&#772;<sub>b</sub> = ' + eb.toFixed(3) + ' and &epsilon;&#772;<sub>p</sub> = ' + ep.toFixed(3) + '</span></div>';

    drawZoom(segs, built, lipo, stride, td, tb, eb, ep, y1 + HV + GAP);

    document.getElementById('d-stridev').textContent = stride;
    document.getElementById('d-delayv').textContent = td;
    document.getElementById('d-blendv').textContent = bh;
    document.getElementById('d-noisev').textContent = (noise * 1000).toFixed(0);
    document.getElementById('d-ebv').textContent = eb.toFixed(3);
    document.getElementById('d-epv').textContent = ep.toFixed(3);
    if (eb < ep) document.getElementById('d-warn').style.display = 'block';
    else document.getElementById('d-warn').style.display = 'none';

    renderSweep(segs, stride, td, tb, eb, ep);
  }


  /* a window around one chunk boundary, scaled so the epsilon bands are readable */
  function drawZoom(segs, built, lipo, stride, td, tb, eb, ep, yz) {
    var gZ = document.getElementById('demo-zoom');
    if (!gZ || segs.length < 2) return;
    gZ.innerHTML = '';
    var c = segs[1].s, from = Math.max(0, c - 14), to = Math.min(N - 1, c + tb + 20);
    var span = to - from;
    var ZX = function (i) { return PAD.l + (W - PAD.l - PAD.r) * (i - from) / span; };

    var lo = Infinity, hi = -Infinity, i, k;
    for (i = from; i <= to; i++) {
      var e0 = built.zone[i] === 0 ? 0 : (built.zone[i] === 1 ? eb : ep);
      lo = Math.min(lo, built.q[i] - e0 * 1.6, lipo[i]);
      hi = Math.max(hi, built.q[i] + e0 * 1.6, lipo[i]);
    }
    for (i = 0; i < segs.length; i++)
      for (k = 0; k < segs[i].q.length; k++) {
        var gi = segs[i].s + k;
        if (gi >= from && gi <= to) { lo = Math.min(lo, segs[i].q[k]); hi = Math.max(hi, segs[i].q[k]); }
      }
    var pd = (hi - lo) * 0.18 || 0.01; lo -= pd; hi += pd;
    var ZY = function (v) { return yz + HZ - HZ * (v - lo) / (hi - lo); };

    gZ.appendChild(el('rect', { x: PAD.l, y: yz, width: W - PAD.l - PAD.r, height: HZ, fill: '#171a1d', rx: 6 }));
    var xd0 = ZX(c), xd1 = ZX(Math.min(c + td, to)), xb1 = ZX(Math.min(c + tb, to));
    if (td > 0) gZ.appendChild(el('rect', { x: xd0, y: yz, width: Math.max(1, xd1 - xd0), height: HZ, fill: '#8ab4ff', 'fill-opacity': .18 }));
    if (tb > td) gZ.appendChild(el('rect', { x: xd1, y: yz, width: Math.max(1, xb1 - xd1), height: HZ, fill: '#ffb454', 'fill-opacity': .19 }));

    var up = '', dn = '';
    for (i = from; i <= to; i++) {
      var e = built.zone[i] === 0 ? 0 : (built.zone[i] === 1 ? eb : ep);
      up += (i === from ? 'M' : 'L') + ZX(i).toFixed(1) + ' ' + ZY(built.q[i] + e).toFixed(1);
    }
    for (i = to; i >= from; i--) {
      var e2 = built.zone[i] === 0 ? 0 : (built.zone[i] === 1 ? eb : ep);
      dn += 'L' + ZX(i).toFixed(1) + ' ' + ZY(built.q[i] - e2).toFixed(1);
    }
    gZ.appendChild(el('path', { d: up + dn + 'Z', fill: '#ffffff', 'fill-opacity': .26, stroke: '#ffffff', 'stroke-opacity': .35, 'stroke-width': 1 }));

    for (i = 0; i < segs.length; i++) {
      var sg = segs[i], d = '', first = true;
      for (k = 0; k < sg.q.length; k++) {
        var g2 = sg.s + k;
        if (g2 < from || g2 > to) continue;
        d += (first ? 'M' : 'L') + ZX(g2).toFixed(1) + ' ' + ZY(sg.q[k]).toFixed(1); first = false;
      }
      if (d) gZ.appendChild(el('path', { d: d, fill: 'none', stroke: CHUNK[i % CHUNK.length], 'stroke-width': 1.8, 'stroke-opacity': .95 }));
    }

    var dl = '';
    for (i = from; i <= to; i++) dl += (i === from ? 'M' : 'L') + ZX(i).toFixed(1) + ' ' + ZY(lipo[i]).toFixed(1);
    gZ.appendChild(el('path', { d: dl, fill: 'none', stroke: '#ffffff', 'stroke-width': 2.8 }));

    function zt(x, y, t, fill, anchor) {
      var n = el('text', { x: x, y: y, fill: fill, 'font-size': 11.5, 'font-family': 'Inter, sans-serif', 'text-anchor': anchor || 'start' });
      n.textContent = t; return n;
    }
    gZ.appendChild(zt(PAD.l + 12, yz + 19, 'one chunk boundary, magnified \u2014 the shaded band is the perturbation \u03b5(t) allowed by Eq. (6)', '#8e9299'));
    if (td > 0) gZ.appendChild(zt((xd0 + xd1) / 2, yz + HZ - 10, 'frozen', '#8ab4ff', 'middle'));
    if (tb > td) gZ.appendChild(zt((xd1 + xb1) / 2, yz + HZ - 10, 'blending', '#ffb454', 'middle'));
    gZ.appendChild(zt(xb1 + 10, yz + HZ - 10, 'path', '#b9b5ac'));
  }

  // ---- what the two bounds buy you ----------------------------------------
  var gSweep = document.getElementById('demo-sweep-g');
  var SW = 920, SH = 160, SP = { l: 64, r: 72, t: 16, b: 36 };

  function renderSweep(segs, stride, td, tb, curEb, curEp) {
    if (!gSweep) return;
    var built = buildReference(segs, stride, td, tb);
    var MAXB = 0.04, M = 13, pts = [], i;
    for (i = 0; i < M; i++) {
      var e = MAXB * i / (M - 1);
      var q = jerkMin(built.q, built.zone, e, curEp, 200);
      var dev = 0;
      for (var k = 0; k < N; k++) dev = Math.max(dev, Math.abs(q[k] - built.q[k]));
      pts.push({ e: e, j: maxJerk(q), dev: dev });
    }
    var jMax = 0; pts.forEach(function (p) { if (p.j > jMax) jMax = p.j; });

    var x0 = SP.l, x1 = SW - SP.r, y0 = SP.t, y1 = SH - SP.b;
    var XX = function (e) { return x0 + (x1 - x0) * e / MAXB; };
    var YJ = function (j) { return y1 - (y1 - y0) * j / (jMax || 1); };
    var YD = function (d) { return y1 - (y1 - y0) * d / MAXB; };

    gSweep.innerHTML = '';
    gSweep.appendChild(el('rect', { x: x0, y: y0, width: x1 - x0, height: y1 - y0, fill: '#fffefb', stroke: '#e4e0d9' }));

    var dj = '', dd = '';
    pts.forEach(function (p, n) {
      dj += (n ? 'L' : 'M') + XX(p.e).toFixed(1) + ' ' + YJ(p.j).toFixed(1);
      dd += (n ? 'L' : 'M') + XX(p.e).toFixed(1) + ' ' + YD(p.dev).toFixed(1);
    });
    gSweep.appendChild(el('path', { d: dd, fill: 'none', stroke: '#b9b5ac', 'stroke-width': 1.6, 'stroke-dasharray': '5 4' }));
    gSweep.appendChild(el('path', { d: dj, fill: 'none', stroke: '#2f6fd0', 'stroke-width': 2.4 }));

    var cx = XX(Math.min(curEb, MAXB));
    gSweep.appendChild(el('line', { x1: cx, y1: y0, x2: cx, y2: y1, stroke: '#1b1d1f', 'stroke-width': 1.2, 'stroke-dasharray': '3 3' }));
    var near = pts.reduce(function (a, b) { return Math.abs(b.e - curEb) < Math.abs(a.e - curEb) ? b : a; });
    gSweep.appendChild(el('circle', { cx: cx, cy: YJ(near.j), r: 4.5, fill: '#2f6fd0' }));

    function tx(x, y, t, fill, anchor) {
      var n = el('text', { x: x, y: y, fill: fill, 'font-size': 11.5, 'font-family': 'Inter, sans-serif', 'text-anchor': anchor || 'start' });
      n.textContent = t; return n;
    }
    gSweep.appendChild(tx(x0 - 8, y0 + 11, 'peak jerk', '#2f6fd0', 'end'));
    gSweep.appendChild(tx(x1 + 8, y0 + 11, 'perturbation', '#8b877f'));
    gSweep.appendChild(tx(x0, y1 + 21, '0', '#6b6e72'));
    gSweep.appendChild(tx((x0 + x1) / 2, y1 + 21, 'blending bound  ε̄b  (rad),  path bound held at ' + curEp.toFixed(3), '#6b6e72', 'middle'));
    gSweep.appendChild(tx(x1, y1 + 21, MAXB.toFixed(2), '#6b6e72', 'end'));
    gSweep.appendChild(tx(cx + 6, y1 - 6, 'ε̄b = ' + curEb.toFixed(3), '#1b1d1f'));
  }

  var sweepSvg = document.getElementById('demo-sweep');
  if (sweepSvg) {
    var dragging = false;
    var setFromX = function (clientX) {
      var r = sweepSvg.getBoundingClientRect();
      var f = ((clientX - r.left) * (SW / r.width) - SP.l) / (SW - SP.r - SP.l);
      f = f < 0 ? 0 : (f > 1 ? 1 : f);
      ctrl.eb.value = Math.round(f * 40);
      render();
    };
    sweepSvg.addEventListener('pointerdown', function (e) { dragging = true; sweepSvg.setPointerCapture(e.pointerId); setFromX(e.clientX); });
    sweepSvg.addEventListener('pointermove', function (e) { if (dragging) setFromX(e.clientX); });
    sweepSvg.addEventListener('pointerup', function () { dragging = false; });
    sweepSvg.addEventListener('pointercancel', function () { dragging = false; });
  }

  Object.keys(ctrl).forEach(function (k) { ctrl[k].addEventListener('input', render); });
  document.getElementById('d-reseed').addEventListener('click', function () { seed = (seed * 7919 + 13) % 100000; render(); });
  render();
})();
