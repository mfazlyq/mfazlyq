# M. Fazly Qusyairy — Personal Portfolio

A responsive personal portfolio website built with **HTML, CSS, and vanilla JavaScript**.

## Run locally in VS Code

No Node.js packages, build step, or framework installation is required.

### Option 1 — VS Code Live Server

1. Clone the repository and checkout the portfolio branch:

   ```bash
   git clone --branch feature/portfolio-website https://github.com/mfazlyq/mfazlyq.git
   ```

2. Open the repository folder in VS Code.
3. Open `portfolio-site/index.html`.
4. Install the **Live Server** extension if it is not already installed.
5. Right-click `index.html` and select **Open with Live Server**.

### Option 2 — Python local server

From the repository root:

```bash
cd portfolio-site
python -m http.server 5500
```

Open [http://localhost:5500](http://localhost:5500) in your browser.

On Windows, `py -m http.server 5500` can be used when the `python` command is unavailable.

## Project files

```text
portfolio-site/
├── index.html     # Semantic page content and sections
├── styles.css     # Responsive visual system and layout
├── script.js      # Mobile navigation, reveal effects, project filters
├── favicon.svg    # FQ monogram
└── README.md      # Local setup instructions
```

## Features

- Responsive layout for desktop, tablet, and mobile.
- Dark navy visual theme with cyan/blue accents.
- About, education, expertise, selected projects, research, and contact sections.
- Project category filters.
- Mobile navigation.
- Scroll reveal effects with reduced-motion support.
- No runtime dependencies or build toolchain.

## Content and validation notes

- The 3D colorectal segmentation project is described as a research prototype.
- The reported 98.42% macro figure refers to tumor-label coverage by the anatomical prior in three selected samples. It is not model Dice, clinical validation, or full-cohort performance.
- The NIST AI Audit Assistant is described as a prototype whose generated findings require human verification.
- Update project details and external profile links as new work is completed.

## Customize

Edit the content in `index.html` and visual tokens at the top of `styles.css`. Project cards can be added by copying an existing `article.project-card` and setting its `data-category` attribute.
