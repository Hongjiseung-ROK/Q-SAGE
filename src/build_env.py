import sys
import shutil
import importlib.util
import os

def check_package(package_name):
    """Checks if a python package is installed."""
    if package_name in sys.modules:
        print(f"[OK] {package_name} is already imported")
        return True
    elif (spec := importlib.util.find_spec(package_name)) is not None:
        print(f"[OK] {package_name} found")
        return True
    else:
        print(f"[FAIL] {package_name} NOT found")
        return False

def check_command(command):
    """Checks if a system command is available."""
    if shutil.which(command):
        print(f"[OK] Command '{command}' found")
        return True
    else:
        print(f"[FAIL] Command '{command}' NOT found. Please ensure it is in your PATH.")
        return False

def main():
    print("Checking Environment for Project ALCHEMIST...")
    print("-" * 50)
    
    all_passed = True
    
    # Check Python Packages
    packages = ["rdkit", "torch", "torch_geometric", "cclib", "numpy", "pandas"]
    for pkg in packages:
        if not check_package(pkg):
            all_passed = False
            
    print("-" * 50)
    
    # Check ORCA
    local_orca = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "orca_bin", "orca.exe"))
    if shutil.which("orca"):
        print("[OK] Command 'orca' found in PATH")
    elif os.path.exists(local_orca):
         print(f"[OK] Local ORCA found at {local_orca}")
    else:
        print("Warning: 'orca' command not found. You may need to add ORCA directory to your PATH.")
        # We don't fail immediately, but warn heavily
        
    print("-" * 50)
    if all_passed:
        print("Environment Check Passed. You are ready to synthesize!")
    else:
        print("Environment Check Failed. Please install missing dependencies.")
        sys.exit(1)

if __name__ == "__main__":
    main()
