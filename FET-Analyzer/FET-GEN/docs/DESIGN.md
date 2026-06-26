# Workbench design reference

The implementation follows the generated concept at `docs/workbench-concept.png`.

Design tokens:

- Background: true cool white `#f8fafc`
- Surface: `#ffffff`
- Text: deep navy `#13213a`
- Accent: precision blue `#1769ff`
- Valid: mint green `#079669`
- Warning: amber `#e78a16`
- Borders: cool gray `#d8e0eb`, 1 px
- Radius: 6–9 px
- Typography: Inter/system sans for UI; tabular numerals for measurements

Layout:

- 58 px application header
- Three-column workbench: 300 px controls, fluid plot/table, 286 px results
- Bottom status rail
- Open panels and rows; the plot is the dominant visual surface
- On narrow screens, inspectors stack below the plot

