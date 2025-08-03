@echo off
REM Script para executar todos os testes do S&P 500 Predictor
REM Compatível com Windows

echo.
echo =====================================
echo    S&P 500 PREDICTOR - TEST SUITE
echo =====================================
echo.

REM Verificar se Python está disponível
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ ERRO: Python não encontrado no PATH
    echo    Instale Python ou adicione ao PATH
    pause
    exit /b 1
)

REM Verificar se o arquivo principal existe
if not exist "run_all_tests.py" (
    echo ❌ ERRO: run_all_tests.py não encontrado
    echo    Certifique-se de que está no diretório correto
    pause
    exit /b 1
)

REM Mostrar menu de opções
echo Escolha uma opção:
echo.
echo 1. Executar todos os testes (padrão)
echo 2. Testes rápidos apenas
echo 3. Apenas tester1 (testes principais)
echo 4. Apenas tester2 (config/benchmark)
echo 5. Apenas benchmarks
echo 6. Apenas testes de configuração
echo 7. Validar ambiente
echo 8. Sair
echo.

set /p choice="Digite sua escolha (1-8): "

if "%choice%"=="1" goto run_all
if "%choice%"=="2" goto run_quick
if "%choice%"=="3" goto run_tester1
if "%choice%"=="4" goto run_tester2
if "%choice%"=="5" goto run_benchmark
if "%choice%"=="6" goto run_config
if "%choice%"=="7" goto validate
if "%choice%"=="8" goto exit
goto invalid_choice

:run_all
echo.
echo 🚀 Executando todos os testes...
python run_all_tests.py --verbose
goto end

:run_quick
echo.
echo ⚡ Executando testes rápidos...
python run_all_tests.py --quick --verbose
goto end

:run_tester1
echo.
echo 🧪 Executando apenas tester1 (testes principais)...
python run_all_tests.py --tester1-only --verbose
goto end

:run_tester2
echo.
echo 🔧 Executando apenas tester2 (config/benchmark)...
python run_all_tests.py --tester2-only --verbose
goto end

:run_benchmark
echo.
echo 📊 Executando apenas benchmarks...
python run_all_tests.py --benchmark-only --verbose
goto end

:run_config
echo.
echo ⚙️ Executando apenas testes de configuração...
python run_all_tests.py --config-only --verbose
goto end

:validate
echo.
echo 🔍 Validando ambiente...
python test_config.py
goto end

:invalid_choice
echo.
echo ❌ Opção inválida. Tente novamente.
pause
goto menu

:end
echo.
echo ✅ Execução concluída!
echo.
echo 📁 Verifique os relatórios gerados:
dir /b *test_report*.* 2>nul
dir /b *benchmark_results*.* 2>nul
echo.
pause
goto exit

:exit
echo.
echo 👋 Até logo!
exit /b 0