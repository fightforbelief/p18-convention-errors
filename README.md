# How Calendar and History Filters Change Panel Forecast Comparisons

Yunjie Guo and Hanqing Yang · Independent Researcher, Suzhou, China

[Read the manuscript (PDF)](paper/main.pdf) · [LaTeX source](paper/main.tex) · [Primary estimates and intervals](results_grid/paired_shift.csv)

A quarterly grid anchored to the first observation excludes series starting on 30 June or 30 September. On four US banking panels the absolute change in the forecast gap ranges from 0.8% to 2.6% of the reference gap. For electricity generation it reaches 17.3%, with 9,465 of 11,583 evaluated plants retained. The gap is mean per-entity MASE of the last-observation forecast minus MASE of the historical mean. Table 1 reports the signs, reference values and paired uncertainty. The electricity and bank-leverage shifts are similar in MASE units.

## Reproduce the primary result from the included data

Use the project root as the working directory. The study environment was Python 3.12.10 on Windows. The five compressed CSV files in [results_grid](results_grid) contain one row per reference entity. The columns are the forecast loss difference and a binary retention indicator. The files omit identifiers. Both membership and loss travel together in each of 4,000 bootstrap resamples. Fixed seed 20261120 is split into five panel-specific streams.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe scripts/grid_sensitivity.py
.\.venv\Scripts\python.exe scripts/make_grid_table.py
```

An existing project environment can be reused. The first analysis command checks both means and both sample sizes against the independently evaluated pipelines and writes the signed shifts and percentile intervals. The next command writes the primary LaTeX table. The paired sensitivity analysis was added after the original runs.

## Reconstruct the raw panels

**Banking.** Download the basic bank-level 1976–2020 ZIP from [NYU's Call Report data page](https://pages.stern.nyu.edu/~pschnabl/data/data_callreport.htm). Extract `callreports_1976_2020_WRDS.dta`. The authors make the source available for noncommercial use. From a fresh checkout build the four Parquet inputs with the command below. It validates the source SHA-256, applies the eligibility filters and float precision used in the original study, and preserves existing output files. In the research environment the verification mode compared all entity keys, dates and values to the four archived panels.

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py scripts/build_banks.py 'path/to/callreports_1976_2020_WRDS.dta'
```

**Electricity.** Download the 24 ZIP files named in [the EIA manifest](data/raw/eia923/MANIFEST.csv) into `data/raw/eia923/`. The manifest records the original URLs, sizes and SHA-256 values. The following commands download missing archives, validate them, and build the plant panel. The generation builder uses the Page 1 generation and fuel sheets, excludes EIA's imputation-only plant ID 99999, and keeps interior zero generation. The technology analysis additionally needs the [2024 EIA-860 archive](https://www.eia.gov/electricity/data/eia860/xls/eia8602024.zip) in `data/raw/eia860/`.

```powershell
Import-Csv data/raw/eia923/MANIFEST.csv | ForEach-Object {
    $file = Join-Path 'data/raw/eia923' $_.file
    if (!(Test-Path $file)) { Invoke-WebRequest $_.url -OutFile $file }
    if ((Get-FileHash $file).Hash -ne $_.sha256) { throw "Hash mismatch for $file" }
}
& $py scripts/build_eia.py
& $py scripts/grid_sensitivity.py --extract
```

The extraction recomputes reference scores from the raw histories. For an original snapshot it checks its file hash, then verifies the calendar-based selection against the original candidate counts and losses. A re-encoded Parquet is checked against the released multiset of loss/retention pairs before using the archived row order for resampling.

## Other results and manuscript

- The [bank](results_win/estimands) and [electricity](results_win_eia/estimands) CSVs report the candidate and reference estimates, decomposition and rankings. The full protocol records and outcomes are in [the freeze ledger](FREEZE_LEDGER.txt). Early hypotheses and the original wider study are retained in [the research log](RESEARCH_LOG.md).
- To rerun comparisons on the four bank panels, use `& $py scripts/run_all.py estimands`. For power plants set `$env:P18_PANELS='eia_plants'`, `$env:P18_OUT='results_win_eia'` and `$env:P18_WORK='work_eia'`, then run the same command. In a fresh reconstruction the length sweep uses the cached task-specific inputs produced by the runner's `llm` stage for each panel set. The original run manifests in both results folders give the code hashes and stage arguments.
- `scripts/mechanism_eia.py` joins generation to the 2024 technology inventory. `scripts/verify_props.py` checks the composition and selection bounds. `scripts/verify_prop4.py` checks the history-length identity against the same reference used for that sweep. The original numerical checks and mechanism tables are included in [proof checks](results_win/proofs) and [technology results](results_win_eia/mechanism).
- To regenerate the historical tables and figure data on the full research machine, run `& $py scripts/make_paper_values.py`. Bank Parquet inputs are required to regenerate the original weight-concentration statistics. All generated files needed to compile the manuscript are included in [paper/generated](paper/generated).
- From `paper/` run `pdflatex main.tex`, `bibtex main`, and `pdflatex main.tex` twice. The unchanged class and bibliography style are from the [Journal of Data Science template](https://github.com/vtex-soft/texsupport.ruc-jds/tree/d38a24d2411651136e1ad2a86243d158b3171a78).
