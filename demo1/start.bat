@echo off
chcp 65001 >nul
cd /d "%~dp0"

where py >nul 2>&1
if errorlevel 1 (
    where python >nul 2>&1
    if errorlevel 1 (
        echo 未检测到 Python，请先安装 Python 3.10-3.12，并勾选 Add Python to PATH。
        pause
        exit /b 1
    )
    set "PYTHON_CMD=python"
) else (
    set "PYTHON_CMD=py -3"
)

if not exist ".venv\Scripts\python.exe" (
    echo 正在创建项目虚拟环境...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 goto :failed
)

echo 正在检查并安装项目依赖...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :failed

if not exist ".env" (
    copy ".env.example" ".env" >nul
    echo.
    echo 已生成 .env，请填写模型密钥和本机 MySQL 配置后再次双击 start.bat。
    start "" notepad ".env"
    pause
    exit /b 1
)

".venv\Scripts\python.exe" main.py
goto :eof

:failed
echo.
echo 初始化失败，请查看上方错误信息。
pause
exit /b 1
