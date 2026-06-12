@echo off
echo ====================================================
echo  Building FaceRec AI Worker for ALL Providers
echo ====================================================
echo.

for %%t in (cpu gpu directml openvino) do (
    echo Starting build for %%t...
    echo. | call build_win.bat %%t
    echo Finished build for %%t.
    echo.
)

echo ====================================================
echo  All builds completed successfully!
echo  Check the 'dist' folder for the executable files.
echo ====================================================
pause
