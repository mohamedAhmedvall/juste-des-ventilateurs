@echo off
REM ============================================================
REM  ingest_gen_features.bat — Juste des Ventilateurs
REM  Genere les features pour chaque episode ingere dans data/raw/
REM
REM  Usage :
REM    ingest_gen_features.bat            -> traite tous les episodes trouves
REM    ingest_gen_features.bat 003        -> traite uniquement l'episode 003
REM
REM  Prerequis :
REM    - conda activate juste-des-ventilateurs
REM    - episodes ingestees dans data/raw/episode=*/
REM ============================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

set RAW_DIR=data\raw
set PROCESSED_DIR=data\processed
set EPISODE_FILTER=%1
set TOTAL=0
set SUCCESS=0
set FAILED=0

echo.
echo ======================================================
echo   Juste des Ventilateurs - Generation des features
echo ======================================================

if "%EPISODE_FILTER%"=="" (
    echo   Mode : tous les episodes dans %RAW_DIR%\
) else (
    echo   Mode : episode specifique = %EPISODE_FILTER%
)
echo ======================================================
echo.

REM -- Verifier que le repertoire raw existe
if not exist "%RAW_DIR%" (
    echo ERREUR : Repertoire '%RAW_DIR%' introuvable.
    echo Lance d'abord ingest_mqtt_simulations.bat pour collecter les donnees.
    pause
    exit /b 1
)

REM -- Lister uniquement les dossiers commencant par "episode=" via dir /b
REM    (le glob CMD avec = est bugge, on filtre avec findstr)
for /f "delims=" %%D in ('dir /b /ad "%RAW_DIR%" 2^>nul ^| findstr /B "episode="') do (
    call :process_episode "%%D"
)

goto :summary

REM ============================================================
:process_episode
REM %1 = nom du dossier (ex: episode=001), sans chemin
REM ============================================================
set FOLDER_NAME=%~1
set FOLDER_PATH=%RAW_DIR%\%FOLDER_NAME%

REM -- Extraire l'ID de l'episode (tout ce qui suit "episode=")
set EP_ID=%FOLDER_NAME:~8%

REM -- Filtrer si un episode specifique est demande
if not "%EPISODE_FILTER%"=="" (
    if not "%EP_ID%"=="%EPISODE_FILTER%" goto :eof
)

set /a TOTAL+=1

echo --------------------------------------------------
echo [Episode %EP_ID%] Dossier : %FOLDER_PATH%
echo --------------------------------------------------

REM -- Verifier si des donnees parquet ou csv existent (y compris sous-dossiers)
set FOUND_DATA=0
for /r "%FOLDER_PATH%" %%F in (*.parquet *.csv) do (
    set FOUND_DATA=1
)

if %FOUND_DATA%==0 (
    echo [%EP_ID%] ATTENTION : Aucun fichier parquet/csv trouve, episode ignore.
    set /a FAILED+=1
    goto :eof
)

REM -- Construire le chemin de sortie
set OUTPUT=%PROCESSED_DIR%\%FOLDER_NAME%

REM -- Passer metadata.json si present
set CONFIG_ARG=
if exist "%FOLDER_PATH%\metadata.json" (
    set CONFIG_ARG=--config %FOLDER_PATH%\metadata.json
    echo [%EP_ID%] Config : %FOLDER_PATH%\metadata.json
) else (
    echo [%EP_ID%] ATTENTION : Pas de metadata.json, specs machines par defaut.
)

echo [%EP_ID%] Sortie : %OUTPUT%
echo [%EP_ID%] Generation des features...

python -m features.pipeline ^
    --input %FOLDER_PATH% ^
    --output %OUTPUT% ^
    %CONFIG_ARG%

if errorlevel 1 (
    echo [%EP_ID%] ERREUR lors de la generation des features.
    set /a FAILED+=1
) else (
    echo [%EP_ID%] Features generees avec succes -^> %OUTPUT%
    set /a SUCCESS+=1
)
echo.
goto :eof

REM ============================================================
:check_failures
if %FAILED% gtr 0 echo ATTENTION : %FAILED% episode(s) en erreur. Verifier les logs ci-dessus.
goto :eof

REM ============================================================
:summary
REM ============================================================
if %TOTAL%==0 (
    if "%EPISODE_FILTER%"=="" (
        echo Aucun episode trouve dans %RAW_DIR%\.
        echo Lance d'abord ingest_mqtt_simulations.bat.
    ) else (
        echo Episode '%EPISODE_FILTER%' introuvable dans %RAW_DIR%\.
    )
    pause
    exit /b 1
)

echo ======================================================
echo   Feature engineering termine.
echo   Episodes traites : %TOTAL%
echo   Succes           : %SUCCESS%
echo   Echecs           : %FAILED%
echo   Donnees dans     : %PROCESSED_DIR%\
echo ======================================================
echo.

call :check_failures
pause
