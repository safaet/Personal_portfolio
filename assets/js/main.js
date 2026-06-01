/* ============================================================
   main.js — Shared JavaScript for all pages
   ============================================================ */

/* ──────────────────────────────────────────────
   1. THEME TOGGLE (dark / light)
   ────────────────────────────────────────────── */

const THEME_KEY = 'portfolio-theme';

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem(THEME_KEY, theme);
}

function initTheme() {
  const saved = localStorage.getItem(THEME_KEY);
  const preferred = window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
  applyTheme(saved || preferred);
}

function bindThemeToggle() {
  const btn = document.querySelector('.theme-toggle');
  if (!btn) return;
  btn.addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme');
    applyTheme(current === 'dark' ? 'light' : 'dark');
  });
}


/* ──────────────────────────────────────────────
   2. MOBILE NAV TOGGLE
   ────────────────────────────────────────────── */

function initMobileNav() {
  const toggle  = document.querySelector('.nav__toggle');
  const drawer  = document.querySelector('.nav__mobile');
  if (!toggle || !drawer) return;

  toggle.addEventListener('click', () => {
    const open = drawer.classList.toggle('open');
    toggle.setAttribute('aria-expanded', open);
  });

  // Close on nav link click
  drawer.querySelectorAll('a').forEach(link => {
    link.addEventListener('click', () => {
      drawer.classList.remove('open');
      toggle.setAttribute('aria-expanded', false);
    });
  });

  // Close on outside click
  document.addEventListener('click', e => {
    if (!toggle.contains(e.target) && !drawer.contains(e.target)) {
      drawer.classList.remove('open');
      toggle.setAttribute('aria-expanded', false);
    }
  });
}


/* ──────────────────────────────────────────────
   3. ACTIVE NAV LINK
   Sets .active on the link matching the current page
   ────────────────────────────────────────────── */

function initActiveNav() {
  const path = window.location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('.nav__link, .nav__mobile a').forEach(link => {
    const href = link.getAttribute('href');
    if (href && (href === path || (path === '' && href === 'index.html'))) {
      link.classList.add('active');
    }
  });
}


/* ──────────────────────────────────────────────
   4. SCROLL REVEAL
   Adds .visible to .reveal elements on scroll
   ────────────────────────────────────────────── */

function initScrollReveal() {
  const items = document.querySelectorAll('.reveal');
  if (!items.length) return;

  const observer = new IntersectionObserver(
    entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.1, rootMargin: '0px 0px -40px 0px' }
  );

  items.forEach(el => observer.observe(el));
}


/* ──────────────────────────────────────────────
   5. PROJECT FILTER
   Filters .card[data-tags] by active .filter-btn
   ────────────────────────────────────────────── */

function initProjectFilter() {
  const filterBar = document.querySelector('.filter-bar');
  if (!filterBar) return;

  filterBar.addEventListener('click', e => {
    const btn = e.target.closest('.filter-btn');
    if (!btn) return;

    filterBar.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    const filter = btn.dataset.filter || 'all';
    const cards = document.querySelectorAll('[data-tags]');

    cards.forEach(card => {
      const tags = card.dataset.tags || '';
      const show = filter === 'all' || tags.includes(filter);
      card.style.display = show ? '' : 'none';
    });
  });
}


/* ──────────────────────────────────────────────
   6. CONTACT FORM (basic validation + submit)
   ────────────────────────────────────────────── */

function initContactForm() {
  const form = document.querySelector('.js-contact-form');
  if (!form) return;

  form.addEventListener('submit', async e => {
    e.preventDefault();
    let valid = true;

    // Validate required fields
    form.querySelectorAll('[required]').forEach(field => {
      const group = field.closest('.form-group');
      if (!field.value.trim()) {
        group.classList.add('has-error');
        group.classList.remove('is-valid');
        valid = false;
      } else {
        group.classList.remove('has-error');
        group.classList.add('is-valid');
      }
    });

    if (!valid) return;

    const submitBtn = form.querySelector('[type="submit"]');
    const successMsg = form.querySelector('.form-success');
    const failMsg    = form.querySelector('.form-fail');

    submitBtn.disabled = true;
    submitBtn.textContent = 'Sending…';

    try {
      // Replace action URL with your Formspree/Netlify endpoint
      const res = await fetch(form.action, {
        method: 'POST',
        body: new FormData(form),
        headers: { 'Accept': 'application/json' }
      });

      if (res.ok) {
        form.reset();
        form.querySelectorAll('.form-group').forEach(g => g.classList.remove('is-valid'));
        if (successMsg) successMsg.classList.add('show');
      } else {
        throw new Error('Server error');
      }
    } catch {
      if (failMsg) failMsg.classList.add('show');
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = 'Send message';
    }
  });

  // Clear error on input
  form.querySelectorAll('input, textarea').forEach(field => {
    field.addEventListener('input', () => {
      field.closest('.form-group')?.classList.remove('has-error');
    });
  });
}


/* ──────────────────────────────────────────────
   7. SKILL BARS (index.html)
   Maps data-level → --skill-level for .skill-bar::after width
   ────────────────────────────────────────────── */

function initSkillBars() {
  const bars = document.querySelectorAll('.skill-bar[data-level]');
  if (!bars.length) return;

  bars.forEach(bar => {
    const level = bar.getAttribute('data-level');
    if (!level) return;

    bar.style.setProperty('--skill-level', '0%');

    const item = bar.closest('.skill-item');
    if (item) {
      let pct = item.querySelector('.skill-pct');
      if (!pct) {
        pct = document.createElement('span');
        pct.className = 'skill-pct';
        item.appendChild(pct);
      }
      pct.textContent = `${level}%`;
    }
  });

  const observer = new IntersectionObserver(
    entries => {
      entries.forEach(entry => {
        if (!entry.isIntersecting) return;
        const level = entry.target.getAttribute('data-level');
        if (level) entry.target.style.setProperty('--skill-level', `${level}%`);
        observer.unobserve(entry.target);
      });
    },
    { threshold: 0.15, rootMargin: '0px 0px -20px 0px' }
  );

  bars.forEach(bar => observer.observe(bar));
}


/* ──────────────────────────────────────────────
   8. COPY CODE BLOCKS (blog posts)
   ────────────────────────────────────────────── */

function initCopyCode() {
  document.querySelectorAll('pre > code').forEach(block => {
    const btn = document.createElement('button');
    btn.className = 'btn btn--ghost btn--sm btn--copy';
    btn.textContent = 'Copy';
    btn.setAttribute('aria-label', 'Copy code');
    block.parentElement.style.position = 'relative';
    block.parentElement.appendChild(btn);

    btn.addEventListener('click', () => {
      navigator.clipboard.writeText(block.textContent).then(() => {
        btn.textContent = 'Copied!';
        setTimeout(() => { btn.textContent = 'Copy'; }, 2000);
      });
    });
  });
}


/* ──────────────────────────────────────────────
   INIT — runs on every page
   ────────────────────────────────────────────── */

document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  bindThemeToggle();
  initMobileNav();
  initActiveNav();
  initScrollReveal();
  initSkillBars();
  initProjectFilter();
  initContactForm();
  initCopyCode();
});
