@echo off
REM ============================================================
REM  ingest_mqtt_simulations.bat — Juste des Ventilateurs
REM  Pour chaque scenario jumeaux-chauds :
REM    1. Passe la simulation a x60
REM    2. Verifie que la simulation est demarree et non en pause
REM    3. Change le scenario
REM    4. Lance le subscriber MQTT pendant 180s
REM    5. Exporte les donnees en Parquet
REM
REM  Prerequis :
REM    - jumeaux-chauds doit etre lance (docker compose up -d)
REM    - conda activate juste-des-ventilateurs
REM    - le fichier .env doit etre present
REM ============================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

set API=http://localhost:8000
set DURATION=180

echo.
echo ======================================================
echo   Juste des Ventilateurs — Ingestion multi-scenarios
echo ======================================================
echo   API         : %API%
echo   Duree/ep    : %DURATION%s reelles (x60 = ~3h simulees)
echo   Scenarios   : basic, busy_weeks, heatwave, nominal, stress, trace_replay
echo ======================================================
echo.

REM -- Verifier que curl est disponible
where curl >nul 2>&1
if errorlevel 1 (
    echo ERREUR : curl n'est pas disponible dans le PATH.
    echo Installe curl ou utilise Git Bash.
    pause
    exit /b 1
)

REM -- Verifier que jumeaux-chauds repond
echo [Init] Verification de l'API jumeaux-chauds...
curl -s -o nul -w "%%{http_code}" %API%/ > %TEMP%\jdv_http_code.txt 2>nul
set /p HTTP_CODE=<%TEMP%\jdv_http_code.txt
if not "%HTTP_CODE%"=="200" (
    echo ERREUR : L'API jumeaux-chauds ne repond pas sur %API%
    echo Verifie que docker compose est lance dans jumeaux-chauds.
    pause
    exit /b 1
)
echo [Init] API OK.

REM -- Passer la simulation a x60
echo [Init] Passage de la simulation a x60...
curl -s -X PUT %API%/simulation/speed ^
     -H "Content-Type: application/json" ^
     -d "{\"speed_multiplier\": 60.0}" > nul
echo [Init] Vitesse x60 activee.
echo.

REM ============================================================
REM  Boucle sur les scenarios
REM ============================================================

set EPISODE=1
set SCENARIOS=basic busy_weeks heatwave nominal stress trace_replay

REM -- Boucle manuelle sur les scenarios (evite les problemes de globbing CMD)
call :run_scenario basic
call :run_scenario busy_weeks
call :run_scenario heatwave
call :run_scenario nominal
call :run_scenario stress
call :run_scenario trace_replay
goto :end_loop

:run_scenario
set SCENARIO=%1
set EP=00%EPISODE%
set EP=%EP:~-3%

echo --------------------------------------------------
echo [Episode %EP%] Scenario : %SCENARIO%
echo --------------------------------------------------

REM -- Changer le scenario
echo [%EP%] Changement du scenario vers %SCENARIO%...
curl -s -X PUT %API%/simulation/scenario ^
     -H "Content-Type: application/json" ^
     -d "{\"scenario\": \"%SCENARIO%\"}" > nul

REM -- Verifier l'etat de la simulation et corriger si necessaire
echo [%EP%] Verification de l'etat de la simulation...
curl -s %API%/simulation/status > %TEMP%\jdv_sim_status.txt 2>nul
set /p SIM_STATUS=<%TEMP%\jdv_sim_status.txt

echo %SIM_STATUS% | findstr /C:"\"paused\": true" > nul
if not errorlevel 1 (
    echo [%EP%] Simulation en pause - reprise...
    curl -s -X POST %API%/simulation/resume > nul
    timeout /t 2 /nobreak > nul
    goto :sim_ready
)
echo %SIM_STATUS% | findstr /C:"\"running\": false" > nul
if not errorlevel 1 (
    echo [%EP%] Simulation arretee - demarrage...
    curl -s -X POST %API%/simulation/start > nul
    timeout /t 3 /nobreak > nul
    goto :sim_ready
)
echo [%EP%] Simulation deja en cours.

:sim_ready
REM -- Laisser le scenario se stabiliser (5s)
echo [%EP%] Stabilisation du scenario (5s)...
timeout /t 5 /nobreak > nul

REM -- Lancer l'ingestion
echo [%EP%] Ingestion MQTT pendant %DURATION%s...
python -m ingest.mqtt_subscriber ^
    --duration %DURATION% ^
    --episode %EP% ^
    --scenario %SCENARIO%

if errorlevel 1 (
    echo [%EP%] ERREUR lors de l'ingestion de l'episode %EP% ^(%SCENARIO%^).
) else (
    echo [%EP%] Episode %EP% ^(%SCENARIO%^) ingere avec succes.
)

echo.
set /a EPISODE+=1
goto :eof

:end_loop

REM -- Repasser a vitesse normale a la fin
echo [Fin] Remise a vitesse x1...
curl -s -X PUT %API%/simulation/speed ^
     -H "Content-Type: application/json" ^
     -d "{\"speed_multiplier\": 1.0}" > nul

echo.
echo ======================================================
echo   Ingestion terminee pour tous les scenarios.
echo   Donnees dans : data\raw\episode=001 a episode=006
echo ======================================================
echo.
echo Lancer maintenant le feature engineering :
echo   ingest_gen_features.bat
echo.
pause
