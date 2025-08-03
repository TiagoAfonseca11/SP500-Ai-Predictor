#!/usr/bin/env python3
"""
Comprehensive Test Runner for S&P 500 Predictor System
Runs all test suites and generates a detailed report
"""

import unittest
import sys
import os
import time
import io
import traceback
from datetime import datetime
from pathlib import Path
import json

class TestResult:
    """Container for test results"""
    def __init__(self, name):
        self.name = name
        self.passed = 0
        self.failed = 0
        self.errors = 0
        self.skipped = 0
        self.total = 0
        self.duration = 0
        self.failures = []
        self.errors_list = []
        self.skipped_list = []
        
    def add_result(self, result):
        """Add unittest result to this container"""
        self.passed = result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)
        self.failed = len(result.failures)
        self.errors = len(result.errors)
        self.skipped = len(result.skipped)
        self.total = result.testsRun
        self.failures = [(test.id(), error) for test, error in result.failures]
        self.errors_list = [(test.id(), error) for test, error in result.errors]
        self.skipped_list = [(test.id(), reason) for test, reason in result.skipped]


class ColoredTextTestResult(unittest.TextTestResult):
    """Enhanced test result with colors and better formatting"""
    
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.successes = []
        self._verbosity = verbosity  # Store verbosity explicitly
        
    @property
    def verbosity(self):
        """Get verbosity level"""
        return getattr(self, '_verbosity', 1)
        
    def addSuccess(self, test):
        super().addSuccess(test)
        self.successes.append(test)
        if self.verbosity > 1:
            self.stream.write(f" {test.id()}\n")
    
    def addError(self, test, err):
        super().addError(test, err)
        if self.verbosity > 1:
            self.stream.write(f" {test.id()} - ERROR\n")
    
    def addFailure(self, test, err):
        super().addFailure(test, err)
        if self.verbosity > 1:
            self.stream.write(f" {test.id()} - FAILED\n")
    
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        if self.verbosity > 1:
            self.stream.write(f"  {test.id()} - SKIPPED: {reason}\n")


class TestRunner:
    """Main test runner class"""
    
    def __init__(self, test_directory="Tests"):
        self.test_directory = Path(test_directory)
        self.results = {}
        self.start_time = None
        self.end_time = None
        
        # If the specified directory doesn't exist, try current directory
        if not self.test_directory.exists():
            current_dir = Path('.')
            # Check if we're already in a test directory (has test_*.py files)
            if list(current_dir.glob("test_*.py")):
                self.test_directory = current_dir
            # Or check if there's a Tests subdirectory
            elif (current_dir / "Tests").exists():
                self.test_directory = current_dir / "Tests"
        
    def discover_tests(self):
        """Discover all test files in the test directory"""
        test_files = []
        
        if not self.test_directory.exists():
            print(f" Test directory '{self.test_directory}' not found!")
            return test_files
        
        # Look for test_*.py files
        for file_path in self.test_directory.glob("test_*.py"):
            test_files.append(file_path)
        
        benchmark_file = self.test_directory / "benchmark_tester.py"
        if benchmark_file.exists():
            test_files.append(benchmark_file)
        
        return sorted(test_files)
    
    def run_test_file(self, test_file):
        """Run tests from a specific file"""
        print(f"\n{'='*60}")
        print(f"🧪 Running tests from: {test_file.name}")
        print(f"{'='*60}")
        
        # Add test directory to Python path
        sys.path.insert(0, str(test_file.parent))
        sys.path.insert(0, str(test_file.parent.parent))  # Add parent for imports
        
        # Module name without .py extension
        module_name = test_file.stem
        
        try:
            # Clear any existing module
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            # Import the module directly
            import importlib.util
            spec = importlib.util.spec_from_file_location(module_name, test_file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # Load tests from the module
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(module)
            
            # Create a test runner with simple TextTestResult to avoid issues
            stream = io.StringIO()
            runner = unittest.TextTestRunner(
                stream=stream,
                verbosity=2,
                buffer=True
            )
            
            # Run the tests
            start_time = time.time()
            result = runner.run(suite)
            duration = time.time() - start_time
            
            # Store results
            test_result = TestResult(module_name)
            test_result.add_result(result)
            test_result.duration = duration
            
            # Print summary for this file
            self.print_file_summary(test_result)
            
            return test_result
            
        except Exception as e:
            print(f" Error running {test_file.name}: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            
            # Return error result
            error_result = TestResult(module_name)
            error_result.errors = 1
            error_result.total = 1
            error_result.errors_list = [(module_name, str(e))]
            return error_result
        
        finally:
            # Clean up path
            if str(test_file.parent) in sys.path:
                sys.path.remove(str(test_file.parent))
            if str(test_file.parent.parent) in sys.path:
                sys.path.remove(str(test_file.parent.parent))
    
    def print_file_summary(self, result):
        """Print summary for a single test file"""
        total = result.total
        passed = result.passed
        failed = result.failed
        errors = result.errors
        skipped = result.skipped
        duration = result.duration
        
        print(f"\n Summary for {result.name}:")
        print(f"   Total: {total} |  Passed: {passed} |  Failed: {failed} |  Errors: {errors} |   Skipped: {skipped}")
        print(f"   Duration: {duration:.2f}s")
        
        if failed > 0:
            print(f"\n Failures in {result.name}:")
            for test_name, error in result.failures[:3]:  # Show first 3 failures
                print(f"   • {test_name}")
        
        if errors > 0:
            print(f"\n Errors in {result.name}:")
            for test_name, error in result.errors_list[:3]:  # Show first 3 errors
                print(f"   • {test_name}")
    
    def run_all_tests(self):
        """Run all discovered tests"""
        self.start_time = datetime.now()
        
        print("🚀 S&P 500 Predictor - Test Suite Runner")
        print("=" * 60)
        print(f"Started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        test_files = self.discover_tests()
        
        if not test_files:
            print("❌ No test files found!")
            return
        
        print(f"📁 Found {len(test_files)} test files:")
        for file in test_files:
            print(f"   • {file.name}")
        
        # Run each test file
        for test_file in test_files:
            result = self.run_test_file(test_file)
            self.results[test_file.name] = result
        
        self.end_time = datetime.now()
        
        # Generate final report
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive test report"""
        total_duration = (self.end_time - self.start_time).total_seconds()
        
        print(f"\n{'='*80}")
        print("📋 COMPREHENSIVE TEST REPORT")
        print(f"{'='*80}")
        
        # Overall statistics
        total_tests = sum(r.total for r in self.results.values())
        total_passed = sum(r.passed for r in self.results.values())
        total_failed = sum(r.failed for r in self.results.values())
        total_errors = sum(r.errors for r in self.results.values())
        total_skipped = sum(r.skipped for r in self.results.values())
        
        print(f"🕐 Execution Time: {total_duration:.2f} seconds")
        print(f"📊 Overall Results:")
        print(f"   Total Tests: {total_tests}")
        print(f"    Passed: {total_passed} ({total_passed/total_tests*100:.1f}%)" if total_tests > 0 else "   Passed: 0")
        print(f"    Failed: {total_failed} ({total_failed/total_tests*100:.1f}%)" if total_tests > 0 else "   Failed: 0")
        print(f"    Errors: {total_errors} ({total_errors/total_tests*100:.1f}%)" if total_tests > 0 else "   Errors: 0")
        print(f"    Skipped: {total_skipped} ({total_skipped/total_tests*100:.1f}%)" if total_tests > 0 else "   Skipped: 0")
        
        # Success rate
        if total_tests > 0:
            success_rate = (total_passed / total_tests) * 100
            print(f"\n🎯 Success Rate: {success_rate:.1f}%")
            
            if success_rate >= 89.5:
                print("🟢 Excellent! High test success rate.")
            elif success_rate >= 75:
                print("🟡 Good test success rate, but room for improvement.")
            else:
                print("🔴 Low test success rate - attention needed.")
        
        # Per-file breakdown
        print(f"\n📁 Per-File Breakdown:")
        print("-" * 80)
        
        for file_name, result in self.results.items():
            status = "" if result.failed == 0 and result.errors == 0 else ""
            print(f"{status} {file_name:<25} | "
                  f"Tests: {result.total:>3} | "
                  f"Passed: {result.passed:>3} | "
                  f"Failed: {result.failed:>3} | "
                  f"Errors: {result.errors:>3} | "
                  f"Time: {result.duration:>6.2f}s")
        
        # Test coverage assessment
        has_unit_tests = any('test_sp500_predictor' in name for name in self.results.keys())
        has_api_tests = any('test_api' in name for name in self.results.keys())
        has_acceptance_tests = any('test_acceptance' in name for name in self.results.keys())
        
        print(f"\n🎯 Test Coverage Assessment:")
        print(f"   Unit Tests: {'✅' if has_unit_tests else '❌'}")
        print(f"   API Tests: {'✅' if has_api_tests else '❌'}")
        print(f"   Acceptance Tests: {'✅' if has_acceptance_tests else '❌'}")
        
        # Final verdict
        print(f"\n{'='*80}")
        if total_tests > 0:
            if total_failed == 0 and total_errors == 0:
                print(" ALL TESTS PASSED! System appears to be working correctly.")
            elif total_failed + total_errors < total_tests * 0.1:  # Less than 10% failures
                print(" Most tests passed. Minor issues to address.")
            else:
                print("  Significant test failures detected. Review needed.")
        else:
            print(" No tests executed. Check test discovery and setup.")
        
        print(f"{'='*80}")
        
        # Save report to file
        self.save_report_to_file()
    
    def save_report_to_file(self):
        """Save test report to a JSON file"""
        report_data = {
            'timestamp': self.start_time.isoformat(),
            'duration': (self.end_time - self.start_time).total_seconds(),
            'summary': {
                'total_tests': sum(r.total for r in self.results.values()),
                'total_passed': sum(r.passed for r in self.results.values()),
                'total_failed': sum(r.failed for r in self.results.values()),
                'total_errors': sum(r.errors for r in self.results.values()),
                'total_skipped': sum(r.skipped for r in self.results.values())
            },
            'files': {}
        }
        
        for file_name, result in self.results.items():
            report_data['files'][file_name] = {
                'total': result.total,
                'passed': result.passed,
                'failed': result.failed,
                'errors': result.errors,
                'skipped': result.skipped,
                'duration': result.duration,
                'failures': [name for name, _ in result.failures],
                'errors_list': [name for name, _ in result.errors_list]
            }
        
        # Save to file
        report_file = Path('Tests/test_report.json')
        try:
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2)
            print(f"\n📄 Detailed report saved to: {report_file}")
        except Exception as e:
            print(f"⚠️  Could not save report file: {e}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run S&P 500 Predictor test suite')
    parser.add_argument('--directory', '-d', default='Tests', 
                       help='Test directory (default: Tests)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Create and run test runner
    runner = TestRunner(args.directory)
    runner.run_all_tests()


if __name__ == '__main__':
    main()