#!/usr/bin/env python3
"""
S&P 500 Predictor Server Startup Script - Melhorado
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path
import signal
import atexit

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print('\n\n⏹️  Servidor interrompido pelo usuário')
    sys.exit(0)

def check_python_version():
    """Check Python version"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ é necessário")
        print(f"   Versão atual: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} detectado")
    return True

def check_requirements():
    """Check if all requirements are installed"""
    required_packages = [
        ('flask', 'Flask'),
        ('flask_cors', 'Flask-CORS'),
        ('yfinance', 'yfinance'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('sklearn', 'scikit-learn'),
        ('joblib', 'joblib')
    ]
    
    missing = []
    for package, pip_name in required_packages:
        try:
            __import__(package)
            print(f"✅ {pip_name}")
        except ImportError:
            print(f"❌ {pip_name} - AUSENTE")
            missing.append(pip_name)
    
    return missing

def install_requirements(missing_packages):
    """Install missing requirements"""
    if not missing_packages:
        return True
    
    print(f"\n📦 Instalando {len(missing_packages)} pacotes ausentes...")
    
    try:
        cmd = [sys.executable, "-m", "pip", "install"] + missing_packages
        subprocess.check_call(cmd)
        print("✅ Pacotes instalados com sucesso")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Falha na instalação: {e}")
        print("\n💡 Tente instalar manualmente:")
        for package in missing_packages:
            print(f"   pip install {package}")
        return False

def setup_templates():
    """Setup templates folder"""
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    
    index_html = templates_dir / "index.html"
    
    if not index_html.exists():
        # Check if index.html exists in current directory
        if Path("index.html").exists():
            print("📁 Movendo index.html para templates/")
            Path("index.html").rename(index_html)
        else:
            print("⚠️  templates/index.html não encontrado")
            print("   Certifique-se de que o arquivo HTML está na pasta templates/")
            return False
    
    print("✅ Templates configurados")
    return True

def check_files():
    """Check required files"""
    required_files = [
        ("sp500_predictor.py", "Módulo principal do predictor"),
        ("app.py", "Servidor Flask"),
        ("templates/index.html", "Interface web")
    ]
    
    all_exist = True
    for file_path, description in required_files:
        if Path(file_path).exists():
            print(f"✅ {description}")
        else:
            print(f"❌ {description} - {file_path} não encontrado")
            all_exist = False
    
    return all_exist

def test_predictor():
    """Test if predictor works"""
    print("\n🧪 Testando predictor básico...")
    
    try:
        from sp500_predictor import EnhancedSP500Predictor
        predictor = EnhancedSP500Predictor()
        
        # Test data download
        data = predictor.download_data(period="5d")
        if data is None or data.empty:
            print("❌ Falha no download de dados de teste")
            return False
        
        print(f"✅ Download de dados funcionando (últimos 5 dias)")
        print(f"   Último preço: ${data['Close'].iloc[-1]:.2f}")
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste do predictor: {e}")
        return False

def check_port(port=5000):
    """Check if port is available"""
    import socket
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        
        if result == 0:
            print(f"⚠️  Porta {port} já está em uso")
            return False
        else:
            print(f"✅ Porta {port} disponível")
            return True
    except Exception as e:
        print(f"⚠️  Erro verificando porta: {e}")
        return True  # Assume available if can't check

def open_browser_delayed(url, delay=3):
    """Open browser after delay"""
    def open_browser():
        time.sleep(delay)
        try:
            webbrowser.open(url)
            print(f"🌐 Abrindo navegador: {url}")
        except Exception as e:
            print(f"⚠️  Não foi possível abrir o navegador: {e}")
            print(f"   Acesse manualmente: {url}")
    
    import threading
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()

def start_server(port=5000):
    """Start the Flask server"""
    print(f"\n🚀 Iniciando S&P 500 Predictor Server na porta {port}...")
    print("=" * 60)
    print("📊 Dashboard: http://localhost:5000")
    print("🔧 API Status: http://localhost:5000/api/status")
    print("⏹️  Para parar: Ctrl+C")
    print("=" * 60)
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Open browser after delay
        open_browser_delayed(f'http://localhost:{port}')
        
        # Import and start Flask app
        from app import app
        app.run(
            debug=False,
            host='0.0.0.0',
            port=port,
            threaded=True,
            use_reloader=False  # Avoid double startup
        )
        
    except ImportError as e:
        print(f"❌ Erro importando app.py: {e}")
        return False
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Porta {port} já está em uso")
            print("   Tente uma porta diferente ou pare o processo existente")
        else:
            print(f"❌ Erro do sistema: {e}")
        return False
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        return False

def cleanup_old_files():
    """Clean up old or corrupted files"""
    print("🧹 Limpando arquivos antigos...")
    
    # Check for corrupted model
    model_file = Path("enhanced_sp500_model.pkl")
    if model_file.exists():
        try:
            import joblib
            joblib.load(model_file)
            print("✅ Modelo existente válido")
        except Exception:
            print("⚠️  Modelo corrompido, removendo...")
            model_file.unlink()
            print("✅ Modelo removido - será retreinado automaticamente")
    
    # Clean up old log files if too large
    log_files = ["sp500_predictions.log"]
    for log_file in log_files:
        log_path = Path(log_file)
        if log_path.exists() and log_path.stat().st_size > 10_000_000:  # 10MB
            print(f"🗑️  Removendo log grande: {log_file}")
            log_path.unlink()

def show_system_info():
    """Show system information"""
    print("💻 INFORMAÇÕES DO SISTEMA")
    print("-" * 30)
    print(f"Python: {sys.version.split()[0]}")
    print(f"Plataforma: {sys.platform}")
    print(f"Diretório: {os.getcwd()}")
    
    # Check virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("🔒 Ambiente virtual: Ativo")
    else:
        print("⚠️  Ambiente virtual: Inativo (recomendado usar venv)")

def interactive_setup():
    """Interactive setup process"""
    print("🔧 CONFIGURAÇÃO INTERATIVA")
    print("-" * 30)
    
    # Ask about missing dependencies
    missing = check_requirements()
    if missing:
        response = input(f"\n❓ Instalar {len(missing)} pacotes ausentes? (y/n): ").lower()
        if response in ['y', 'yes', 's', 'sim']:
            if not install_requirements(missing):
                return False
        else:
            print("⚠️  Instalação cancelada. O sistema pode não funcionar corretamente.")
    
    # Ask about port
    try:
        port_input = input("\n❓ Porta do servidor (padrão: 5000): ").strip()
        port = int(port_input) if port_input else 5000
    except ValueError:
        port = 5000
        print("⚠️  Porta inválida, usando padrão: 5000")
    
    return port

def quick_start():
    """Quick start without interaction"""
    missing = check_requirements()
    if missing:
        print("📦 Instalando dependências automaticamente...")
        if not install_requirements(missing):
            return False
    
    return 5000

def main():
    """Main function"""
    print("🚀 S&P 500 AI Predictor - Inicialização")
    print("=" * 50)
    
    # Check Python version first
    if not check_python_version():
        return
    
    # Show system info
    show_system_info()
    
    # Clean up old files
    cleanup_old_files()
    
    # Check required files
    print(f"\n📁 VERIFICANDO ARQUIVOS")
    print("-" * 30)
    if not check_files():
        print("\n❌ Arquivos essenciais ausentes!")
        print("   Certifique-se de que todos os arquivos estão no diretório correto.")
        return
    
    # Setup templates
    if not setup_templates():
        return
    
    # Check dependencies and get port
    print(f"\n📦 VERIFICANDO DEPENDÊNCIAS")
    print("-" * 30)
    
    # Check if running interactively
    if len(sys.argv) > 1 and '--auto' in sys.argv:
        port = quick_start()
    else:
        port = interactive_setup()
    
    if not port:
        return
    
    # Check port availability
    if not check_port(port):
        alt_port = port + 1
        print(f"🔄 Tentando porta alternativa: {alt_port}")
        if check_port(alt_port):
            port = alt_port
        else:
            print("❌ Nenhuma porta disponível encontrada")
            return
    
    # Test predictor
    print(f"\n🧪 TESTE DO SISTEMA")
    print("-" * 30)
    if not test_predictor():
        response = input("\n❓ Predictor com problemas. Continuar mesmo assim? (y/n): ").lower()
        if response not in ['y', 'yes', 's', 'sim']:
            print("⏹️  Inicialização cancelada")
            return
    
    # Final confirmation
    print(f"\n✅ SISTEMA PRONTO")
    print("-" * 30)
    print("🎯 Tudo configurado e funcionando!")
    print(f"🌐 Servidor será iniciado em: http://localhost:{port}")
    
    if '--auto' not in sys.argv:
        input("\n▶️  Pressione Enter para iniciar o servidor...")
    
    # Start server
    start_server(port)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Inicialização interrompida pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro durante inicialização: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Dicas:")
        print("1. Verifique se todos os arquivos estão presentes")
        print("2. Execute: python debug_server.py para diagnósticos")
        print("3. Verifique conexão com internet")
        print("4. Tente executar: python app.py diretamente")