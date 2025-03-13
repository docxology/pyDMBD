import sys
print('Python version:', sys.version)
try:
    import dmbd
    print('DMBD is available')
    if hasattr(dmbd, '__version__'):
        print('DMBD version:', dmbd.__version__)
    else:
        print('DMBD version: Unknown')
except ImportError as e:
    print('Could not import DMBD:', e)
    
# Also check if there's a local dmbd module in parent dir
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"\nChecking for DMBD in parent directory: {parent_dir}")
if os.path.exists(os.path.join(parent_dir, 'dmbd')):
    print("There is a 'dmbd' directory in the parent folder")
else:
    print("No 'dmbd' directory found in parent folder") 