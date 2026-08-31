# CCPT Webpage Integration Guide

Welcome! This package contains the complete, standalone, portable interactive research article explaining the **Constitutional Control-Plane Transformer (CCPT)** experiment.

It has been built with zero external runtime dependencies (vanilla semantic HTML5, modern CSS, and plain JavaScript) so you can easily drop it into almost any personal website, static blog, or documentation site.

---

## 1. Directory Structure

```text
ccpt_webpage_handoff/
├── README_INTEGRATION.md       <- This guide
├── SOURCE_PROVENANCE.md        <- Research provenance tracking every number to raw artifacts
├── SCIENTIFIC_CLAIMS.md        <- Evidentiary classifications (SUPPORTED / NOT CLAIMED)
├── ccpt-article.html           <- Main semantic HTML5 article
├── ccpt-article.css            <- Clean, responsive, accessible stylesheet
├── ccpt-article.js             <- Vanilla JavaScript interactive controllers
├── data/
│   └── ccpt-results.json       <- Authoritative public dataset (dynamically fetched)
├── assets/
│   ├── architecture-fallback.svg  <- No-JS fallback for architecture diagram
│   ├── persistence-fallback.svg   <- No-JS fallback for persistence chart
│   └── controller-fallback.svg    <- No-JS fallback for causal ablation chart
└── scripts/
    └── build_public_results.py    <- Python script to re-derive data from raw research repo
```

---

## 2. Quick Local Preview

To test and preview the webpage immediately on your local machine:

```bash
# Navigate to the package directory
cd ccpt_webpage_handoff

# Start an ordinary static HTTP server (required for fetch('data/ccpt-results.json'))
python3 -m http.server 8000
```

Open your browser to: `http://localhost:8000/ccpt-article.html`

---

## 3. Recommended Public URL & Routing

- **Recommended Public Path / Slug:** `/research/ccpt/` (or `/projects/ccpt/`)
- **Index File:** `ccpt-article.html` (can be renamed to `index.html` inside `/research/ccpt/`)

---

## 4. Integration Instructions

### Option A: Plain Static Site / Folder Copy (Simplest)
1. Copy the entire folder into your public web root as `research/ccpt/`:
   ```bash
   cp -r ccpt_webpage_handoff/ /path/to/my-website/public/research/ccpt/
   mv /path/to/my-website/public/research/ccpt/ccpt-article.html /path/to/my-website/public/research/ccpt/index.html
   ```
2. Wrap or replace the top `<header>` and bottom `<footer>` elements in `index.html` with your website's universal navigation bar and footer.

### Option B: Static Site Generators (Astro, Hugo, 11ty, Jekyll)
1. **HTML Body:** Copy the contents inside `<main class="ccpt-wrapper">...</main>` from `ccpt-article.html` into your site layout/template.
2. **Styles:** Link or import `ccpt-article.css`.
3. **Script:** Link `ccpt-article.js` at the bottom of the page.
4. **Data & Assets:** Ensure `data/ccpt-results.json` and `assets/*.svg` are copied to your static public assets directory so `fetch('data/ccpt-results.json')` resolves correctly.

### Option C: Modern Web Frameworks (Next.js, Vite, SvelteKit)
- You can either serve this as a static pre-rendered route in your `public/` directory or port the semantic HTML markup into a component. All JavaScript logic in `ccpt-article.js` uses standard browser DOM APIs and does not require Node.js or bundler polyfills.

---

## 5. Critical Scientific Data Rule

> ⚠️ **IMPORTANT:** Do NOT manually edit numeric values inside `data/ccpt-results.json` or hardcode scientific numbers in the HTML/JS.

`data/ccpt-results.json` is mathematically derived from frozen, audited checkpoint evaluations. If you need to regenerate the data, run:
```bash
python scripts/build_public_results.py
```

---

## 6. Verification Checklist After Integration

After deploying or embedding into your site, verify:
- [ ] **Architecture Explorer:** Clicking `1. LM Training`, `2. Safety Training`, and `3. Inference` dynamically highlights active vs. frozen pathways.
- [ ] **Persistence Explorer:** Toggling `Harmful Refusal Retention` and `Benign Over-Refusal` correctly updates the three-seed bar chart.
- [ ] **Controller Explorer:** Clicking `Seed 1`, `Seed 2`, and `Seed 3` updates the active vs. ablated (off) bars and sensitivity intervals.
- [ ] **Responsive Test:** Layout scales gracefully across mobile (320px), tablet (768px), and desktop (1440px) without horizontal scrolling.
- [ ] **No-JS Fallback:** Disabling JavaScript in browser devtools displays the high-resolution fallback SVGs in place of the interactive charts.
