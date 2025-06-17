#!/usr/bin/env python3
"""
Test runner for gym-nim.

This file serves as the main entry point for running tests.
It runs the comprehensive pytest suite located in the tests/ directory.
"""

import sys
import subprocess


def main():
    """Run the full test suite."""
    try:
        # Run pytest with coverage
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            'tests/', 
            '-v',
            '--cov=gym_nim',
            '--cov-report=term-missing',
            '--cov-report=html'
        ], check=True)
        
        print("\n✅ All tests passed!")
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with return code {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
