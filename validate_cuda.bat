@echo off
if "%CUDA_HOME%" == "" (
    echo CUDA_HOME is not set.
) else (
    echo CUDA_HOME is set to %CUDA_HOME%.
)
