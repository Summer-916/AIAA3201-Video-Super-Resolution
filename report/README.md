# Report Draft

This folder contains the report draft written from the project PDF requirements.

Files:

```text
report/main.tex
report/cvpr_main.tex
report/preamble.tex
report/references.bib
report/main.bib
report/sec/*.tex
report/figures/*.png
```

The project handout asks for the CVPR LaTeX template and a 6--8 page report excluding references. Two entry points are provided:

- `main.tex`: standalone CVPR-like article draft that does not require the official `cvpr.sty`.
- `cvpr_main.tex`: official CVPR author-kit style entry file matching the provided template structure.

To submit with the official template:

1. Download the CVPR LaTeX template.
2. Copy `cvpr_main.tex` to the official template's `main.tex`.
3. Copy `preamble.tex`, `main.bib`, the `sec/` folder, and the `figures/` folder into the official template directory.
4. Replace `Student 1` and `Student 2` with the real group member names.
5. Compile from the `report/` directory or update figure paths if the report folder is moved.

Key requirements already included:

- GitHub repository link at the end of the abstract.
- Related work cites all 18 papers listed in the project PDF.
- Method section covers Part 1, Part 2, and Part 3.
- Experiments include quantitative tables and qualitative figures.
- Full REDS validation benchmark is included as additional standard data.
- Conclusion, limitations, and future work are included.

The official-template version references compact report-specific figures under `figures/`. These figures are tracked with the report source so Overleaf does not need the full `results/` folder. The standalone `main.tex` may still reference local `results/` paths; prefer `cvpr_main.tex` for final submission.
