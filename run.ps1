<#
.SYNOPSIS
  Windows PowerShell equivalent of the Makefile targets for sdg-leakage-audit.
.USAGE
  .\run.ps1 setup
  .\run.ps1 data
  .\run.ps1 test
  .\run.ps1 notebooks
  .\run.ps1 all
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet("setup", "data", "test", "notebooks", "all")]
    [string]$Target = "all"
)

$PythonExe = "C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "py"
}

function Invoke-Setup {
    & $PythonExe -3.10 -m venv .venv 2>$null
    if (-not (Test-Path ".venv")) {
        & $PythonExe -m venv .venv
    }
    & .\.venv\Scripts\python.exe -m pip install --upgrade pip
    & .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    & .\.venv\Scripts\python.exe -m pip install -e .
}

function Invoke-Data {
    & .\.venv\Scripts\python.exe -m sdg_leakage_audit.data.download --help
}

function Invoke-Test {
    & .\.venv\Scripts\python.exe -m pytest -v
}

function Invoke-Notebooks {
    & .\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace `
        notebooks\00_project_setup_and_environment_check.ipynb
}

switch ($Target) {
    "setup"     { Invoke-Setup }
    "data"      { Invoke-Data }
    "test"      { Invoke-Test }
    "notebooks" { Invoke-Notebooks }
    "all"       { Invoke-Setup; Invoke-Test }
}
