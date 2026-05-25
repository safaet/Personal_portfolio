# Portfolio Design System — Reference Guide

## Folder Structure

```
portfolio/
├── index.html                     ← Home (hero, about, skills, resume, contact)
├── projects.html                  ← Software projects (DevSearch, AI Lesson Planner, School Mgmt)
├── ai-portfolio.html              ← AI/ML projects (AMR, House Price, Mask Detection)
├── research.html                  ← Research interests, reading list, collaboration CTA
├── blog.html                      ← Blog post list + featured post
├── tools.html                     ← Developer tools, stack recommendations
├── contact.html                   ← Contact form (standalone)
│
├── css/
│   ├── style.css                  ← Full design system (variables, layout, buttons, cards)
│   └── form.css                   ← Input, textarea, validation styles (contact page only)
│
├── js/
│   └── main.js                    ← Theme toggle, nav, scroll reveal, filter, form submit
│
├── images/
│   ├── avatar.jpg                 ← Profile photo (500×500 square, WebP preferred)
│   ├── og-image.jpg               ← Social preview card (1200×630)
│   ├── projects/                  ← One screenshot per software project
│   │   ├── devsearch.webp         ← 800×500 project thumbnail
│   │   ├── ai-lesson-planner.webp
│   │   └── school-management.webp
│   ├── ai-ml/                     ← AI/ML project visuals (charts, demo screenshots)
│   │   ├── amr-prediction.webp
│   │   ├── house-price.webp
│   │   └── mask-detection.webp
│   ├── blog/                      ← Blog post cover images (1200×630)
│   └── icons/                     ← Tech stack SVG icons (React, Java, Docker, etc.)
│
├── blog/
│   └── posts/                     ← Individual blog post HTML files
│       └── post-template.html
│
├── artifacts/
│   └── public-summary/            ← Public-facing learning summary JSON/HTML snippets
│       └── progress.json          ← { currentFocus, weeklyGoal, skills: [...] }
│
├── resume/
│   └── safaet-jaman-resume.pdf
│
└── README.md                      ← This file
```

[Visit the live site](https://safaet.github.io/Personal_portfolio/)

---

## CSS Naming Convention (BEM-lite)

```
Block:    .card          .btn          .nav          .section
Element:  .card__title   .btn--icon    .nav__link    .section-title
Modifier: .card--project .btn--primary .nav--open
State:    .is-active     .has-error    .is-valid
JS hook:  .js-*          (never styled — only used in JS querySelector)
Utility:  .text-center   .flex-center  .gap-4
```

### Pattern rules:
- Block = standalone component
- `__` = child element of that block
- `--` = visual variant/modifier
- `is-` / `has-` = JS-toggled state classes
- `js-` prefix = selector used only in JavaScript, never in CSS

---

## Color Tag Convention

| Page / Content Type | CSS variable          | Tag class       |
|---------------------|-----------------------|-----------------|
| Software projects   | `--color-software`    | `.tag--software`|
| AI/ML projects      | `--color-ai`          | `.tag--ai`      |
| Research            | `--color-research`    | `.tag--research`|
| Blog                | `--color-blog`        | `.tag--blog`    |
| Tools               | `--color-tools`       | `.tag--tools`   |

---

## Image Asset Plan

| Image               | Format  | Dimensions  | Where used                      |
|---------------------|---------|-------------|----------------------------------|
| avatar.jpg          | WebP/JPG| 500×500     | About section, meta OG           |
| og-image.jpg        | JPG     | 1200×630    | `<meta property="og:image">`     |
| projects/*.webp     | WebP    | 800×500     | Project card thumbnails          |
| ai-ml/*.webp        | WebP    | 800×500     | AI/ML card thumbnails            |
| blog/*.webp         | WebP    | 1200×630    | Blog post cover (open graph too) |
| icons/*.svg         | SVG     | 24×24       | Skills section, tech badges      |

**Always include `width` + `height` attributes on `<img>` to prevent layout shift.**

---

## How to Use the Design System

### Every page must include:
```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=Outfit:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="css/style.css">
<!-- contact.html only: -->
<link rel="stylesheet" href="css/form.css">

<script src="js/main.js" defer></script>
```

### Page shell template:
```html
<nav class="nav">
  <div class="nav__inner">
    <a href="index.html" class="nav__brand">SJ<span>.</span></a>
    <ul class="nav__links"> <!-- desktop links --> </ul>
    <div class="nav__actions">
      <button class="theme-toggle" aria-label="Toggle theme">
        <span class="icon-moon">🌙</span>
        <span class="icon-sun">☀️</span>
      </button>
      <button class="nav__toggle" aria-label="Open menu" aria-expanded="false">
        <span></span><span></span><span></span>
      </button>
    </div>
  </div>
</nav>
<nav class="nav__mobile"> <!-- mobile links --> </nav>

<main>
  <!-- page content -->
</main>

<footer class="footer"> ... </footer>
```
