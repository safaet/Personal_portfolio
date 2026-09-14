/* ============================================================
   bg-network.js — Site-wide animated background
   Neural-network / particle-graph effect, with a subtle DNA-helix
   accent confined to each page's hero section.

   - No external libraries.
   - Colors are read from the CSS custom properties defined in
     assets/css/style.css (--violet-light, --cyan-light,
     --text-secondary) via getComputedStyle, so the effect follows
     the light/dark theme toggle in main.js (which sets
     document.documentElement[data-theme]).
   - Pauses when the tab is hidden, renders a single static frame
     for prefers-reduced-motion, and never intercepts pointer events.
   ============================================================ */

(function () {
  'use strict';

  var HERO_SELECTOR =
    '.hero, .tools-hero, .contact-hero, .projects-hero, .ai-hero, .blog-hero, .research-hero';

  var CONNECT_DIST = 140;      // px — nodes closer than this get a line
  var MAX_DPR = 2;             // cap devicePixelRatio to avoid huge canvases
  var RESIZE_DEBOUNCE = 200;   // ms

  var canvas, ctx;
  var width = 0, height = 0, dpr = 1;
  var nodes = [];
  var colors = { node: '#7c3aed', node2: '#06b6d4', line: '#8b9cc8' };
  var rafId = null;
  var running = false;
  var reduceMotionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
  var helixPhase = 0;

  /* ── Setup ───────────────────────────────────────────────── */

  function createCanvas() {
    canvas = document.createElement('canvas');
    canvas.id = 'bg-network';
    canvas.setAttribute('aria-hidden', 'true');
    canvas.style.cssText = [
      'position:fixed',
      'inset:0',
      'width:100%',
      'height:100%',
      'z-index:-1',
      'pointer-events:none',
      'opacity:0.4',
      'display:block'
    ].join(';');

    if (document.body.firstChild) {
      document.body.insertBefore(canvas, document.body.firstChild);
    } else {
      document.body.appendChild(canvas);
    }

    ctx = canvas.getContext('2d');
  }

  function readColors() {
    var styles = getComputedStyle(document.documentElement);
    var violetLight = styles.getPropertyValue('--violet-light').trim();
    var cyanLight = styles.getPropertyValue('--cyan-light').trim();
    var textSecondary = styles.getPropertyValue('--text-secondary').trim();

    colors = {
      node: violetLight || '#a78bfa',
      node2: cyanLight || '#67e8f9',
      line: textSecondary || '#8b9cc8'
    };
  }

  function nodeCountForViewport(w) {
    if (w < 480) return 20;
    if (w < 768) return 30;
    if (w < 1200) return 45;
    return 60;
  }

  function createNodes() {
    var count = nodeCountForViewport(width);
    nodes = [];
    for (var i = 0; i < count; i++) {
      nodes.push({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * 0.3,
        vy: (Math.random() - 0.5) * 0.3,
        alt: Math.random() < 0.5
      });
    }
  }

  function resize() {
    width = window.innerWidth;
    height = window.innerHeight;
    dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR);

    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    createNodes();
    renderFrame(); // keep something on screen immediately after resize
  }

  function debounce(fn, wait) {
    var t;
    return function () {
      clearTimeout(t);
      t = setTimeout(fn, wait);
    };
  }

  /* ── Update ──────────────────────────────────────────────── */

  function update() {
    for (var i = 0; i < nodes.length; i++) {
      var n = nodes[i];
      n.x += n.vx;
      n.y += n.vy;
      if (n.x <= 0 || n.x >= width) n.vx *= -1;
      if (n.y <= 0 || n.y >= height) n.vy *= -1;
      n.x = Math.max(0, Math.min(width, n.x));
      n.y = Math.max(0, Math.min(height, n.y));
    }
  }

  /* ── Draw ────────────────────────────────────────────────── */

  function drawNetwork() {
    ctx.clearRect(0, 0, width, height);

    // Connections
    ctx.lineWidth = 1.2;
    for (var i = 0; i < nodes.length; i++) {
      for (var j = i + 1; j < nodes.length; j++) {
        var a = nodes[i], b = nodes[j];
        var dx = a.x - b.x, dy = a.y - b.y;
        var dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < CONNECT_DIST) {
          var alpha = 1 - dist / CONNECT_DIST;
          ctx.strokeStyle = colors.line;
          ctx.globalAlpha = alpha * 0.85;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }

    // Nodes
    ctx.globalAlpha = 1;
    for (var k = 0; k < nodes.length; k++) {
      var node = nodes[k];
      ctx.fillStyle = node.alt ? colors.node2 : colors.node;
      ctx.beginPath();
      ctx.arc(node.x, node.y, 2.6, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.globalAlpha = 1;
  }

  function drawHelix() {
    var hero = document.querySelector(HERO_SELECTOR);
    if (!hero) return;

    var rect = hero.getBoundingClientRect();
    var clipX = Math.max(rect.left, 0);
    var clipY = Math.max(rect.top, 0);
    var clipRight = Math.min(rect.right, width);
    var clipBottom = Math.min(rect.bottom, height);
    var clipW = clipRight - clipX;
    var clipH = clipBottom - clipY;
    if (clipW <= 0 || clipH <= 0) return;

    var anchorX = clipX + clipW * 0.86;
    var amplitude = Math.min(clipW * 0.05, 26);

    ctx.save();
    ctx.beginPath();
    ctx.rect(clipX, clipY, clipW, clipH);
    ctx.clip();
    ctx.globalAlpha = 0.5;
    ctx.lineWidth = 1.5;

    var step = 10;
    ctx.strokeStyle = colors.node;
    ctx.beginPath();
    for (var y = clipY; y <= clipBottom; y += step) {
      var t = (y - clipY) / 34 + helixPhase;
      var x = anchorX + Math.sin(t) * amplitude;
      if (y === clipY) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.strokeStyle = colors.node2;
    ctx.beginPath();
    for (var y2 = clipY; y2 <= clipBottom; y2 += step) {
      var t2 = (y2 - clipY) / 34 + helixPhase;
      var x2 = anchorX + Math.sin(t2 + Math.PI) * amplitude;
      if (y2 === clipY) ctx.moveTo(x2, y2); else ctx.lineTo(x2, y2);
    }
    ctx.stroke();

    // Rungs connecting the two strands
    ctx.globalAlpha = 0.3;
    ctx.strokeStyle = colors.line;
    for (var y3 = clipY; y3 <= clipBottom; y3 += step * 4) {
      var t3 = (y3 - clipY) / 34 + helixPhase;
      var rx1 = anchorX + Math.sin(t3) * amplitude;
      var rx2 = anchorX + Math.sin(t3 + Math.PI) * amplitude;
      ctx.beginPath();
      ctx.moveTo(rx1, y3);
      ctx.lineTo(rx2, y3);
      ctx.stroke();
    }

    ctx.restore();
    ctx.globalAlpha = 1;
  }

  function renderFrame() {
    drawNetwork();
    drawHelix();
  }

  /* ── Animation loop ──────────────────────────────────────── */

  function tick() {
    if (!running) return;
    update();
    helixPhase += 0.01;
    renderFrame();
    rafId = requestAnimationFrame(tick);
  }

  function start() {
    if (running || reduceMotionQuery.matches) return;
    running = true;
    rafId = requestAnimationFrame(tick);
  }

  function stop() {
    running = false;
    if (rafId) cancelAnimationFrame(rafId);
    rafId = null;
  }

  /* ── Reduced motion ──────────────────────────────────────── */

  function applyMotionPreference() {
    if (reduceMotionQuery.matches) {
      stop();
      renderFrame(); // static single frame
    } else {
      start();
    }
  }

  /* ── Wire up ─────────────────────────────────────────────── */

  function init() {
    createCanvas();
    readColors();
    resize();
    applyMotionPreference();

    window.addEventListener('resize', debounce(resize, RESIZE_DEBOUNCE));

    document.addEventListener('visibilitychange', function () {
      if (document.hidden) {
        stop();
      } else {
        applyMotionPreference();
      }
    });

    if (typeof reduceMotionQuery.addEventListener === 'function') {
      reduceMotionQuery.addEventListener('change', applyMotionPreference);
    } else if (typeof reduceMotionQuery.addListener === 'function') {
      reduceMotionQuery.addListener(applyMotionPreference); // Safari < 14
    }

    // Re-read colors whenever the theme toggle flips data-theme on <html>
    var themeObserver = new MutationObserver(function () {
      readColors();
      renderFrame();
    });
    themeObserver.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-theme']
    });
  }

  if (document.body) {
    init();
  } else {
    document.addEventListener('DOMContentLoaded', init);
  }
})();
