@echo off
setlocal

REM === Check for Python ===
where python >nul 2>&1
if errorlevel 1 (
    echo Python is not installed. Please install Python 3.x from https://www.python.org/downloads/ and re-run this script.
    pause
    exit /b 1
)

REM === Check for pip ===
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo pip not found. Attempting to install pip...
    python -m ensurepip
    if errorlevel 1 (
        echo Failed to install pip. Please install pip manually.
        pause
        exit /b 1
    )
)

REM === Install aqtinstall if not present ===
python -m pip show aqtinstall >nul 2>&1
if errorlevel 1 (
    echo Installing aqtinstall...
    python -m pip install aqtinstall
    if errorlevel 1 (
        echo Failed to install aqtinstall. Please check your Python/pip installation.
        pause
        exit /b 1
    )
)

REM === Download and install Qt ===
set QT_VERSION=5.15.2
set QT_DIR=C:\Qt\%QT_VERSION%\msvc2019_64
if not exist "%QT_DIR%" (
    echo Downloading and installing Qt %QT_VERSION% for MSVC2019_64...
    python -m aqt install-qt windows desktop %QT_VERSION% win64_msvc2019_64 --outputdir C:\Qt
    if errorlevel 1 (
        echo Failed to install Qt with aqtinstall.
        pause
        exit /b 1
    )
) else (
    echo Qt already installed at %QT_DIR%
)

REM === Download and extract OpenCV ===
set OPENCV_URL=https://github.com/opencv/opencv/releases/download/4.5.5/opencv-4.5.5-vc14_vc15.exe
set OPENCV_EXTRACT=C:\opencv
if not exist "%OPENCV_EXTRACT%" (
    echo Downloading OpenCV...
    powershell -Command "Invoke-WebRequest -Uri %OPENCV_URL% -OutFile opencv.exe"
    echo Extracting OpenCV...
    mkdir "%OPENCV_EXTRACT%"
    opencv.exe /S /D=%OPENCV_EXTRACT%
    del opencv.exe
) else (
    echo OpenCV already exists at %OPENCV_EXTRACT%
)

REM === Download and extract Assimp ===
set ASSIMP_URL=https://github.com/assimp/assimp/releases/download/v5.2.5/assimp-5.2.5-win64.zip
set ASSIMP_EXTRACT=C:\assimp
if not exist "%ASSIMP_EXTRACT%" (
    echo Downloading Assimp...
    powershell -Command "Invoke-WebRequest -Uri %ASSIMP_URL% -OutFile assimp.zip"
    echo Extracting Assimp...
    powershell -Command "Expand-Archive -Path assimp.zip -DestinationPath %ASSIMP_EXTRACT%"
    del assimp.zip
) else (
    echo Assimp already exists at %ASSIMP_EXTRACT%
)

REM === SET PATHS FOR CMAKE ===
set QT_PATH=%QT_DIR%\lib\cmake
set OPENCV_DIR=%OPENCV_EXTRACT%\build\x64\vc15\lib
set ASSIMP_PATH=%ASSIMP_EXTRACT%\assimp-5.2.5-win64\lib\cmake\assimp-5.2

REM === Create and enter build directory ===
cd /d "%~dp0"
if not exist build mkdir build
cd build

REM === Run CMake configuration ===
cmake .. ^
  -DOpenCV_DIR="%OPENCV_DIR%" ^
  -DCMAKE_PREFIX_PATH="%QT_PATH%;%ASSIMP_PATH%"

REM === Build the project (Release mode) ===
cmake --build . --config Release

REM === Copy the executable to app folder ===
if exist .\bin\RBOT.exe (
    copy .\bin\RBOT.exe ..\app\
    echo Executable copied to app folder.
) else (
    echo Build failed or RBOT.exe not found!
)

cd ..
echo Done!
pause 