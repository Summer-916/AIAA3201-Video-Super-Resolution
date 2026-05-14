# Report Draft

This folder contains the report draft written from the project PDF requirements.

Files:

```text
report/main.tex
report/references.bib
```

The project handout asks for the CVPR LaTeX template and a 6--8 page report excluding references. This draft uses a CVPR-like two-column article layout because the official CVPR template files are not included in the repository. To submit with the official template:

1. Download the CVPR LaTeX template.
2. Copy the body of `main.tex` into the official `main.tex`.
3. Keep `references.bib` as the bibliography file.
4. Replace `Student 1` and `Student 2` with the real group member names.
5. Compile from the repository root or update figure paths if the report folder is moved.

Key requirements already included:

- GitHub repository link at the end of the abstract.
- Related work cites all 18 papers listed in the project PDF.
- Method section covers Part 1, Part 2, and Part 3.
- Experiments include quantitative tables and qualitative figures.
- Full REDS validation benchmark is included as additional standard data.
- Conclusion, limitations, and future work are included.

The figures referenced by the draft are local result files under `results/`. They are ignored by git, so make sure the downloaded `results/` folder is present before compiling.
