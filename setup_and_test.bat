@echo off
REM Setup and Test Script for K-Means Clustering Project
REM Run this script to build everything and run initial tests

echo ============================================
echo K-Means Clustering Project Setup
echo ============================================
echo.

REM Create results directory if it doesn't exist
if not exist "results" mkdir results
echo [OK] Results directory ready

echo.
echo Step 1: Building Dataset Generator...
echo ----------------------------------------
cd datasets
gcc -O2 -o data_generator.exe data_generator.c -lm
if %errorlevel% neq 0 (
    echo [ERROR] Failed to compile dataset generator
    pause
    exit /b 1
)
echo [OK] Dataset generator compiled successfully

echo.
echo Step 2: Generating Sample Datasets...
echo ----------------------------------------
data_generator.exe 10000 5 2 data_10k.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to generate data_10k.txt
    pause
    exit /b 1
)
echo [OK] Generated data_10k.txt (10K points, 5 clusters, 2D)

data_generator.exe 100000 8 2 data_100k.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to generate data_100k.txt
    pause
    exit /b 1
)
echo [OK] Generated data_100k.txt (100K points, 8 clusters, 2D)

echo.
echo [INFO] You can generate more datasets later with:
echo        data_generator.exe ^<points^> ^<clusters^> ^<dimensions^> ^<output_file^>
echo.

cd ..

echo.
echo Step 3: Building Serial K-Means...
echo ----------------------------------------
cd 01_serial
gcc -O3 -o kmeans_serial.exe kmeans_serial.c -lm
if %errorlevel% neq 0 (
    echo [ERROR] Failed to compile serial K-Means
    cd ..
    pause
    exit /b 1
)
echo [OK] Serial K-Means compiled successfully

echo.
echo Step 4: Running Test with 10K Dataset...
echo ----------------------------------------
kmeans_serial.exe ..\datasets\data_10k.txt 5 100
if %errorlevel% neq 0 (
    echo [ERROR] Test run failed
    cd ..
    pause
    exit /b 1
)

echo.
echo Step 5: Running Test with 100K Dataset...
echo ----------------------------------------
kmeans_serial.exe ..\datasets\data_100k.txt 8 100
if %errorlevel% neq 0 (
    echo [ERROR] Test run failed
    cd ..
    pause
    exit /b 1
)

cd ..

echo.
echo ============================================
echo Setup Complete!
echo ============================================
echo.
echo Next Steps:
echo 1. Check results in the 'results' folder
echo 2. Review timing data in 'results\serial_timing.txt'
echo 3. Start implementing OpenMP version in '02_openmp' folder
echo.
echo Quick commands:
echo   - Generate more data: cd datasets ^&^& data_generator.exe ^<args^>
echo   - Run serial version: cd 01_serial ^&^& kmeans_serial.exe ^<args^>
echo   - View results: type results\serial_results.txt
echo.
echo See README.md and PROJECT_IMPLEMENTATION_FLOW.md for details
echo.

pause
