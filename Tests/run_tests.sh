#!/bin/bash
# Script para executar todos os testes do S&P 500 Predictor
# Compatível com Linux/Mac

set -e  # Sair em caso de erro

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "\n====================================="
echo -e "   ${BLUE}S&P 500 PREDICTOR - TEST SUITE${NC}"
echo -e "====================================="

# Função para mostrar erros
show_error() {
    echo -e "${RED}❌ ERRO: $1${NC}"
}

# Função para mostrar sucesso
show_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# Função para mostrar info
show_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Verificar se Python está disponível
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        show_error "Python não encontrado no PATH"
        echo "   Instale Python 3.7+ ou adicione ao PATH"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

# Verificar versão do Python
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
show_info "Usando Python $PYTHON_VERSION"

# Verificar se o arquivo principal existe
if [ ! -f "run_all_tests.py" ]; then
    show_error "run_all_tests.py não encontrado"
    echo "   Certifique-se de que está no diretório correto"
    exit 1
fi

# Função para mostrar menu
show_menu() {
    echo -e "\n${YELLOW}Escolha uma opção:${NC}"
    echo ""
    echo "1. Executar todos os testes (padrão)"
    echo "2. Testes rápidos apenas"
    echo "3. Apenas tester1 (testes principais)"
    echo "4. Apenas tester2 (config/benchmark)"
    echo "5. Apenas benchmarks"
    echo "6. Apenas testes de configuração"
    echo "7. Validar ambiente"
    echo "8. Instalar dependências"
    echo "9. Sair"
    echo ""
}

# Função para instalar dependências
install_dependencies() {
    echo -e "\n${YELLOW}🔧 Instalando dependências...${NC}"
    
    # Verificar se pip está disponível
    if command -v pip3 &> /dev/null; then
        PIP_CMD="pip3"
    elif command -v pip &> /dev/null; then
        PIP_CMD="pip"
    else
        show_error "pip não encontrado"
        return 1
    fi
    
    # Lista de dependências
    DEPENDENCIES="pandas numpy scikit-learn yfinance joblib psutil matplotlib seaborn"
    
    echo "Instalando: $DEPENDENCIES"
    $PIP_CMD install $DEPENDENCIES
    
    if [ $? -eq 0 ]; then
        show_success "Dependências instaladas com sucesso!"
    else
        show_error "Falha na instalação das dependências"
        return 1
    fi
}

# Função para executar testes
run_tests() {
    local cmd="$PYTHON_CMD run_all_tests.py $1"
    echo -e "\n${BLUE}Executando: $cmd${NC}"
    echo ""
    
    # Executar comando
    if $cmd; then
        show_success "Execução concluída!"
        
        # Mostrar arquivos de relatório gerados
        echo -e "\n${YELLOW}📁 Relatórios gerados:${NC}"
        ls -la *test_report*.* 2>/dev/null || echo "   Nenhum relatório encontrado"
        ls -la *benchmark_results*.* 2>/dev/null || echo "   Nenhum benchmark encontrado"
        
    else
        show_error "Falha na execução dos testes"
        return 1
    fi
}

# Loop principal
while true; do
    show_menu
    read -p "Digite sua escolha (1-9): " choice
    
    case $choice in
        1)
            echo -e "\n${BLUE}🚀 Executando todos os testes...${NC}"
            run_tests "--verbose"
            ;;
        2)
            echo -e "\n${BLUE}⚡ Executando testes rápidos...${NC}"
            run_tests "--quick --verbose"
            ;;
        3)
            echo -e "\n${BLUE}🧪 Executando apenas tester1 (testes principais)...${NC}"
            run_tests "--tester1-only --verbose"
            ;;
        4)
            echo -e "\n${BLUE}🔧 Executando apenas tester2 (config/benchmark)...${NC}"
            run_tests "--tester2-only --verbose"
            ;;
        5)
            echo -e "\n${BLUE}📊 Executando apenas benchmarks...${NC}"
            run_tests "--benchmark-only --verbose"
            ;;
        6)
            echo -e "\n${BLUE}⚙️ Executando apenas testes de configuração...${NC}"
            run_tests "--config-only --verbose"
            ;;
        7)
            echo -e "\n${BLUE}🔍 Validando ambiente...${NC}"
            $PYTHON_CMD test_config.py
            ;;
        8)
            install_dependencies
            ;;
        9)
            echo -e "\n${GREEN}👋 Até logo!${NC}"
            exit 0
            ;;
        *)
            show_error "Opção inválida. Tente novamente."
            ;;
    esac
    
    echo ""
    read -p "Pressione Enter para continuar..."
done