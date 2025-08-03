#!/usr/bin/env python3
"""
S&P 500 Predictor Server Startup Script - Enhanced
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
    print('\n\n  Server interrupted by user')
    sys.exit(0)

def check_python_version():
    """Check Python version"""
    if sys.version_info < (3, 8):
        print(" Python 3.8+ is required")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f" Python {sys.version.split()[0]} detected")
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
            print(f" {pip_name}")
        except ImportError:
            print(f" {pip_name} - MISSING")
            missing.append(pip_name)
    
    return missing

def install_requirements(missing_packages):
    """Install missing requirements"""
    if not missing_packages:
        return True
    
    print(f"\n Installing {len(missing_packages)} missing packages...")
    
    try:
        cmd = [sys.executable, "-m", "pip", "install"] + missing_packages
        subprocess.check_call(cmd)
        print(" Packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f" Installation failed: {e}")
        print("\n Try installing manually:")
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
            print(" Moving index.html to templates/")
            Path("index.html").rename(index_html)
        else:
            print(" templates/index.html not found")
            print("   Make sure the HTML file is in the templates/ folder")
            return False
    
    print(" Templates configured")
    return True

def check_files():
    """Check required files"""
    required_files = [
        ("sp500_predictor.py", "Main predictor module"),
        ("app.py", "Flask server"),
        ("templates/index.html", "Web interface")
    ]
    
    all_exist = True
    for file_path, description in required_files:
        if Path(file_path).exists():
            print(f" {description}")
        else:
            print(f" {description} - {file_path} not found")
            all_exist = False
    
    return all_exist

def test_predictor():
    """Test if predictor works"""
    print("\n Testing basic predictor...")
    
    try:
        from sp500_predictor import EnhancedSP500Predictor
        predictor = EnhancedSP500Predictor()
        
        # Test data download
        data = predictor.download_data(period="5d")
        if data is None or data.empty:
            print(" Test data download failed")
            return False
        
        print(f" Data download working (last 5 days)")
        print(f"   Last price: ${data['Close'].iloc[-1]:.2f}")
        return True
        
    except Exception as e:
        print(f" Predictor test error: {e}")
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
            print(f"  Port {port} is already in use")
            return False
        else:
            print(f" Port {port} available")
            return True
    except Exception as e:
        print(f"  Error checking port: {e}")
        return True  # Assume available if can't check

def open_browser_delayed(url, delay=3):
    """Open browser after delay"""
    def open_browser():
        time.sleep(delay)
        try:
            webbrowser.open(url)
            print(f" Opening browser: {url}")
        except Exception as e:
            print(f"  Could not open browser: {e}")
            print(f"   Access manually: {url}")
    
    import threading
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()

def start_server(port=5000):
    """Start the Flask server"""
    print(f"\n Starting S&P 500 Predictor Server on port {port}...")
    print("=" * 60)
    print(" Dashboard: http://localhost:5000")
    print(" API Status: http://localhost:5000/api/status")
    print("  To stop: Ctrl+C")
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
        print(f" Error importing app.py: {e}")
        return False
    except OSError as e:
        if "Address already in use" in str(e):
            print(f" Port {port} is already in use")
            print("   Try a different port or stop the existing process")
        else:
            print(f" System error: {e}")
        return False
    except Exception as e:
        print(f" Unexpected error: {e}")
        return False

def cleanup_old_files():
    """Clean up old or corrupted files"""
    print(" Cleaning up old files...")
    
    # Check for corrupted model
    model_file = Path("enhanced_sp500_model.pkl")
    if model_file.exists():
        try:
            import joblib
            joblib.load(model_file)
            print(" Existing model is valid")
        except Exception:
            print("  Corrupted model, removing...")
            model_file.unlink()
            print(" Model removed - will be retrained automatically")
    
    # Clean up old log files if too large
    log_files = ["sp500_predictions.log"]
    for log_file in log_files:
        log_path = Path(log_file)
        if log_path.exists() and log_path.stat().st_size > 10_000_000:  # 10MB
            print(f"Removing large log: {log_file}")
            log_path.unlink()

def show_system_info():
    """Show system information"""
    print(" SYSTEM INFORMATION")
    print("-" * 30)
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {sys.platform}")
    print(f"Directory: {os.getcwd()}")
    
    # Check virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print(" Virtual environment: Active")
    else:
        print("  Virtual environment: Inactive (recommended to use venv)")

def interactive_setup():
    """Interactive setup process"""
    print(" INTERACTIVE SETUP")
    print("-" * 30)
    
    # Ask about missing dependencies
    missing = check_requirements()
    if missing:
        response = input(f"\n❓ Install {len(missing)} missing packages? (y/n): ").lower()
        if response in ['y', 'yes', 's', 'sim']:
            if not install_requirements(missing):
                return False
        else:
            print("  Installation cancelled. System may not work properly.")
    
    # Ask about port
    try:
        port_input = input("\n❓ Server port (default: 5000): ").strip()
        port = int(port_input) if port_input else 5000
    except ValueError:
        port = 5000
        print("  Invalid port, using default: 5000")
    
    return port

def quick_start():
    """Quick start without interaction"""
    missing = check_requirements()
    if missing:
        print(" Installing dependencies automatically...")
        if not install_requirements(missing):
            return False
    
    return 5000

def main():
    """Main function"""
    print(" S&P 500 AI Predictor - Initialization")
    print("=" * 50)
    
    # Check Python version first
    if not check_python_version():
        return
    
    # Show system info
    show_system_info()
    
    # Clean up old files
    cleanup_old_files()
    
    # Check required files
    print(f"\n CHECKING FILES")
    print("-" * 30)
    if not check_files():
        print("\n Essential files missing!")
        print("   Make sure all files are in the correct directory.")
        return
    
    # Setup templates
    if not setup_templates():
        return
    
    # Check dependencies and get port
    print(f"\n CHECKING DEPENDENCIES")
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
        print(f" Trying alternative port: {alt_port}")
        if check_port(alt_port):
            port = alt_port
        else:
            print(" No available port found")
            return
    
    # Test predictor
    print(f"\n SYSTEM TEST")
    print("-" * 30)
    if not test_predictor():
        response = input("\n❓ Predictor has issues. Continue anyway? (y/n): ").lower()
        if response not in ['y', 'yes', 's', 'sim']:
            print("  Initialization cancelled")
            return
    
    # Final confirmation
    print(f"\n SYSTEM READY")
    print("-" * 30)
    print(" Everything configured and working!")
    print(f" Server will start at: http://localhost:{port}")
    
    if '--auto' not in sys.argv:
        input("\n  Press Enter to start server...")
    
    # Start server
    start_server(port)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Initialization interrupted by user")
    except Exception as e:
        print(f"\n Error during initialization: {e}")
        import traceback
        traceback.print_exc()
        print("\n Tips:")
        print("1. Check if all files are present")
        print("2. Run: python debug_server.py for diagnostics")
        print("3. Check internet connection")
        print("4. Try running: python app.py directly")