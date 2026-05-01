@echo off
chcp 936 >nul
echo ================================
echo  批量TXT转XML格式化工具
echo ================================

cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo 错误：未找到venv虚拟环境
    pause
    exit /b 1
)

echo 正在激活虚拟环境...
call venv\Scripts\activate.bat

python -c "import flask" 2>nul
if not errorlevel 1 goto deps_ok

echo ================================
echo  首次启动，正在安装依赖...
echo ================================
echo.
echo 安装 PyTorch CUDA 12.8 ...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
echo.
echo 安装其他依赖...
pip install -r requirements.txt
echo.
echo ================================
echo  依赖安装完成！
echo ================================

:deps_ok
echo.
echo 正在启动服务...
python app.py
pause
