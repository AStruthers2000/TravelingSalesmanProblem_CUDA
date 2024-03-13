@echo OFF

rmdir /s /q Build
mkdir Build
mkdir Build\BuildFiles

cd Build\BuildFiles
cmake ..\..
cmake --build .
cd ..
move BuildFiles\Datasets
move BuildFiles\Debug\VRP.exe

start

@echo ON