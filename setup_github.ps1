# ============================================================
# GitHub Upload Script - AI-Assisted IT Support Concept
# ============================================================
# This script safely initializes a git repo, stages only safe
# files (excluding .md, secrets, large models, data), and
# pushes to the remote GitHub repository.
#
# SAFETY CHECKS:
#   1. All *.md files are excluded via .gitignore
#   2. .env is excluded (only .env.example is tracked)
#   3. models/ directory (multi-GB GGUF files) is excluded
#   4. data/raw, data/processed, data/labelled, data/eval, data/attribution are excluded
#   5. Virtual environments (.it_support, .venv, venv) are excluded
#   6. Logs, indexes, and results artifacts are excluded
#   7. __pycache__ and IDE files are excluded
# ============================================================

$ErrorActionPreference = "Stop"
Set-Location "c:\Users\nwagb\Desktop\it_support_ai"

Write-Host "`n=== Step 1: Initializing Git Repository ===" -ForegroundColor Cyan
git init -b main

Write-Host "`n=== Step 2: Configuring Remote ===" -ForegroundColor Cyan
git remote add origin https://github.com/Nduka99/ai_assisted_it_support_concept.git
Write-Host "Remote 'origin' set to: https://github.com/Nduka99/ai_assisted_it_support_concept.git"

Write-Host "`n=== Step 3: Staging Files (respecting .gitignore) ===" -ForegroundColor Cyan
git add .

Write-Host "`n=== Step 4: Pre-Push Safety Audit ===" -ForegroundColor Yellow
Write-Host "--- Checking for .md files in staging area ---"
$mdFiles = git diff --cached --name-only | Select-String "\.md$"
if ($mdFiles) {
    Write-Host "WARNING: The following .md files are staged:" -ForegroundColor Red
    $mdFiles | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    Write-Host "Unstaging .md files now..." -ForegroundColor Yellow
    git diff --cached --name-only | Select-String "\.md$" | ForEach-Object { git reset HEAD $_.ToString().Trim() }
} else {
    Write-Host "  PASS: No .md files in staging area" -ForegroundColor Green
}

Write-Host "--- Checking for .env files in staging area ---"
$envFiles = git diff --cached --name-only | Select-String "^\.env$"
if ($envFiles) {
    Write-Host "WARNING: .env file is staged! Removing..." -ForegroundColor Red
    git reset HEAD .env
} else {
    Write-Host "  PASS: No .env secrets in staging area" -ForegroundColor Green
}

Write-Host "--- Checking for large model files ---"
$modelFiles = git diff --cached --name-only | Select-String "\.gguf$|\.bin$|\.safetensors$"
if ($modelFiles) {
    Write-Host "WARNING: Large model files found! Removing..." -ForegroundColor Red
    $modelFiles | ForEach-Object { git reset HEAD $_.ToString().Trim() }
} else {
    Write-Host "  PASS: No large model files in staging area" -ForegroundColor Green
}

Write-Host "`n=== Step 5: Staged Files Summary ===" -ForegroundColor Cyan
git diff --cached --name-only --stat

Write-Host "`n=== Step 6: Review staged files before commit ===" -ForegroundColor Yellow
Write-Host "The files listed above will be committed." -ForegroundColor Yellow
$confirmation = Read-Host "Proceed with commit and push? (y/n)"

if ($confirmation -eq 'y' -or $confirmation -eq 'Y') {
    Write-Host "`n=== Step 7: Committing ===" -ForegroundColor Cyan
    git commit -m "Initial commit: AI-Assisted IT Support Concept

- Project structure with src/, scripts/, notebooks/, configs/, tests/
- Source package: it_support (triage, retrieval, generation, multimodal, data, eval, benchmark, demo)
- Data acquisition and preprocessing pipeline scripts
- Jupyter notebooks for data exploration and evaluation
- Configuration files and environment template
- .gitignore configured to exclude sensitive .md files, secrets, large models, and generated data"

    Write-Host "`n=== Step 8: Pushing to GitHub ===" -ForegroundColor Cyan
    git push -u origin main

    Write-Host "`n=== DONE! Repository pushed successfully ===" -ForegroundColor Green
    Write-Host "URL: https://github.com/Nduka99/ai_assisted_it_support_concept" -ForegroundColor Green
} else {
    Write-Host "Aborted. No changes were pushed." -ForegroundColor Yellow
    Write-Host "You can review and push manually with:" -ForegroundColor Yellow
    Write-Host '  git commit -m "Initial commit: AI-Assisted IT Support Concept"'
    Write-Host '  git push -u origin main'
}
