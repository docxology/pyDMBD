import sys
import pkg_resources

def check_versions():
    """Print versions of all installed packages."""
    print(f"Python version: {sys.version}")
    print("\nInstalled packages:")
    for pkg in sorted(pkg_resources.working_set):
        print(f"{pkg.key} version: {pkg.version}")

if __name__ == "__main__":
    check_versions() 