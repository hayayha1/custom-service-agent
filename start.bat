@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: ============================================================
:: 垂直客服 Agent 一键启动脚本 (同步随项目迭代更新)
:: 当前覆盖阶段: Phase 1 + Phase 2 (知识库清洗分块 + 四阶段混合检索 RRF/Rerank + Agent 全链路监测)
:: ============================================================

cd /d "%~dp0"

echo.
echo ============================================================
echo   垂直客服 Agent (Phase 1 + Phase 2) 一键启动
echo   工作目录: %cd%
echo ============================================================
echo.

:: 1. 检查 .env 配置文件
if not exist ".env" (
    if exist ".env.example" (
        echo [1/4] 未检测到 .env，已自动从 .env.example 复制创建...
        copy /Y ".env.example" ".env" >nul
        echo       提示: 可在 .env 或启动后前往 Web UI「设置」页面配置模型与 Langfuse Key
    ) else (
        echo [WARN] 未找到 .env 与 .env.example，将使用默认配置启动
    )
) else (
    echo [1/4] 已检测到 .env 配置文件
)

:: 2. 启动基础设施容器 (PostgreSQL pgvector:5433 + Redis:6379)
echo [2/4] 检查并启动 Docker 容器 (PostgreSQL pgvector + Redis)...
docker info >nul 2>&1
if %errorlevel%==0 (
    docker compose up -d
    if !errorlevel! neq 0 (
        echo [WARN] docker compose up -d 执行异常，后端将自动使用本地内存降级模式
    ) else (
        echo       Docker 容器已就绪
    )
) else (
    echo [WARN] 未检测到运行中的 Docker Desktop，后端将自动使用本地内存降级模式
)

:: 3. 检查后端与前端依赖
echo [3/4] 检查后端 (.venv) 与前端 (node_modules) 依赖...
if not exist ".venv\Scripts\python.exe" (
    echo       未检测到 Python 虚拟环境，正在执行 uv sync...
    uv sync --default-index https://pypi.tuna.tsinghua.edu.cn/simple
    if !errorlevel! neq 0 (
        echo [ERROR] 后端依赖安装失败，请检查 uv 与网络环境
        pause
        exit /b 1
    )
)

if not exist "frontend\node_modules" (
    echo       未检测到前端依赖，正在执行 pnpm install...
    pushd frontend
    call pnpm install --registry=https://registry.npmmirror.com
    if !errorlevel! neq 0 (
        popd
        echo [ERROR] 前端依赖安装失败，请检查 pnpm 与网络环境
        pause
        exit /b 1
    )
    popd
)

:: 4. 清理上次生产构建残留的 .next 缓存（防止 next dev 路由 404）并分别启动前后端
if exist "frontend\.next\BUILD_ID" (
    echo       检测到旧版生产构建缓存 frontend\.next，正在自动清理...
    rmdir /s /q "frontend\.next"
)

echo [4/4] 正在启动后端服务 (8000) 与前端 Web UI (3000)...
start "CS-Agent Backend (8000)" /D "%~dp0" cmd /k "chcp 65001 >nul && .venv\Scripts\python.exe -m scripts.run_server"
start "CS-Agent Frontend (3000)" /D "%~dp0frontend" cmd /k "chcp 65001 >nul && pnpm dev"

echo.
echo ============================================================
echo   服务已在新窗口启动！
echo   - Web UI 控制台 : http://localhost:3000
echo   - 知识库构建检索: http://localhost:3000/dashboard/knowledge
echo   - Agent 全链监测: http://localhost:3000/dashboard/monitoring
echo   - 模型与监测设置: http://localhost:3000/dashboard/settings
echo   - 后端 API 文档 : http://localhost:8000/docs
echo ============================================================
echo.

timeout /t 4 /nobreak >nul
start "" "http://localhost:3000"

endlocal
