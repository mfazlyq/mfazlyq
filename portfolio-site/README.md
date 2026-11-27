# Academic & Research Portfolio Website

A responsive personal portfolio website for M. Fazly Qusyairy, focused on Artificial Intelligence, Deep Learning Research, Biomedical Image Analysis, and Education Technology.

## Run locally

No build tools or package installation are required.

### Option 1 — VS Code Live Server

1. Clone the repository:
   ```bash
   git clone --branch feature/portfolio-website https://github.com/mfazlyq/mfazlyq.git
   ```
2. Open the repository in VS Code.
3. Open `portfolio-site/index.html`.
4. Run it with the Live Server extension.

### Option 2 — Python

From the repository root:

```bash
cd portfolio-site
python -m http.server 5500
```

Open http://localhost:5500.

## Contents

- Academic and research profile.
- Education and areas of interest.
- Selected projects with links to their standalone repositories.
- Current deep learning research direction.
- Teaching interests and education technology focus.

## Technology

Semantic HTML · CSS · Vanilla JavaScript

## Validation

Run the static integrity checks:

```bash
python portfolio-site/validate.py
node --check portfolio-site/script.js
```

## Content notes

- The colorectal segmentation page distinguishes three-case anatomical-prior coverage from model performance. It does not claim end-to-end model validation.
- The Medical AI Research Workbench is described as a viewer and evaluation application; it does not claim model training or CBAM-RSU2Net inference.
- The NIST AI Audit Assistant is a research prototype. Its findings require human verification.
- The portfolio does not publish patient data, credentials, or unverified contact details.
