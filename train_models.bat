@echo off
REM ============================================================
REM  train_models.bat -- Juste des Ventilateurs
REM  Entraine les modeles de prediction de pannes
REM  et lance l'evaluation comparative.
REM
REM  Usage :
REM    train_models.bat              -> tous les modeles, label failure_60s
REM    train_models.bat failure_30s  -> label specifique
REM
REM  Prerequis :
REM    - conda activate juste-des-ventilateurs
REM    - data/processed/episode=* disponibles (lancer ingest_gen_features.bat)
REM ============================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

set LABEL=%1
if "%LABEL%"=="" set LABEL=failure_60s

echo.
echo ======================================================
echo   Juste des Ventilateurs -- Entrainement modeles
echo ======================================================
echo   Label cible : %LABEL%
echo   Modeles     : baseline, logistic, random_forest, gradient_boosting
echo   Split       : 70 / 15 / 15 par episode (Option A)
echo ======================================================
echo.

REM -- Verifier que les donnees processed existent
set FOUND_EP=0
for /d %%D in (data\processed\episode=*) do set FOUND_EP=1
if %FOUND_EP%==0 (
    echo ERREUR : Aucun episode dans data\processed\
    echo Lancer d'abord ingest_gen_features.bat
    pause
    exit /b 1
)

REM -- Verifier que Python fonctionne
python --version > nul 2>&1
if errorlevel 1 (
    echo ERREUR : Python non disponible dans le PATH.
    echo Activer l'environnement conda : conda activate juste-des-ventilateurs
    pause
    exit /b 1
)

REM -- Creer les dossiers de sortie si necessaire
if not exist "evaluation\results" mkdir "evaluation\results"
if not exist "models\failure_prediction\saved" mkdir "models\failure_prediction\saved"

echo [1/5] EDA rapide (verification du split)...
python ingest_quick_EDA.py --processed-only > evaluation\results\eda_summary.txt 2>&1
if errorlevel 1 (
    echo       ATTENTION : EDA echouee, on continue quand meme.
) else (
    echo       OK -- resume dans evaluation\results\eda_summary.txt
)
echo.

echo [2/5] Baseline heuristique (seuils T_warn + hot_zone)...
python -m evaluation.failure_prediction_eval ^
    --label %LABEL% ^
    --models baseline ^
    --output evaluation\results\results_baseline_%LABEL%.json
call :check_step "baseline"
echo.

echo [3/5] Regression logistique...
python -m evaluation.failure_prediction_eval ^
    --label %LABEL% ^
    --models logistic ^
    --output evaluation\results\results_logistic_%LABEL%.json
call :check_step "logistic"
echo.

echo [4/5] Random Forest...
python -m evaluation.failure_prediction_eval ^
    --label %LABEL% ^
    --models random_forest ^
    --output evaluation\results\results_random_forest_%LABEL%.json
call :check_step "random_forest"
echo.

echo [5/5] Gradient Boosting (XGBoost / LightGBM / sklearn)...
python -m evaluation.failure_prediction_eval ^
    --label %LABEL% ^
    --models gradient_boosting ^
    --output evaluation\results\results_gradient_boosting_%LABEL%.json
call :check_step "gradient_boosting"
echo.

echo ======================================================
echo   Evaluation comparative -- tous les modeles
echo ======================================================
python -m evaluation.failure_prediction_eval ^
    --label %LABEL% ^
    --models baseline logistic random_forest gradient_boosting ^
    --output evaluation\results\failure_prediction_results_%LABEL%.json

if errorlevel 1 (
    echo ERREUR lors de l'evaluation comparative.
    pause
    exit /b 1
)

echo.
echo ======================================================
echo   Entrainement et evaluation termines.
echo   Resultats : evaluation\results\failure_prediction_results_%LABEL%.json
echo   Modeles   : models\failure_prediction\saved\
echo ======================================================
echo.
pause
exit /b 0

REM ============================================================
:check_step
REM %1 = nom du modele (sans guillemets)
REM ============================================================
if errorlevel 1 (
    echo       ERREUR lors de l'etape %~1.
) else (
    echo       OK -- %~1 termine.
)
goto :eof
