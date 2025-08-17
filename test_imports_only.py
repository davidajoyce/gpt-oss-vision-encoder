#!/usr/bin/env python3
"""
Test that our code changes don't break the basic import structure.
This test validates syntax and basic structure without requiring torch.
"""

import sys
import ast
import os

def test_syntax_validity():
    """Test that all our Python files have valid syntax."""
    files_to_check = [
        "gpt_oss/torch/model.py",
        "gpt_oss/torch/vision_projector.py", 
        "gpt_oss/torch/vision_tower.py",
        "tests/test_vision_config.py",
        "tests/test_projector_builder.py",
        "tests/test_vision_integration.py"
    ]
    
    for filepath in files_to_check:
        print(f"Checking syntax of {filepath}...")
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Parse the AST to check syntax
            ast.parse(content)
            print(f"✅ {filepath} has valid syntax")
            
        except SyntaxError as e:
            print(f"❌ {filepath} has syntax error: {e}")
            return False
        except Exception as e:
            print(f"❌ Error checking {filepath}: {e}")
            return False
    
    return True

def test_modelconfig_structure():
    """Test that ModelConfig has the expected structure."""
    print("Checking ModelConfig structure...")
    
    try:
        with open("gpt_oss/torch/model.py", 'r') as f:
            content = f.read()
        
        # Check that vision parameters are present
        vision_params = [
            'mm_vision_tower', 'mm_projector_type', 'mm_hidden_size',
            'mm_vision_select_layer', 'use_mm_proj'
        ]
        
        for param in vision_params:
            if param not in content:
                print(f"❌ Missing vision parameter: {param}")
                return False
        
        print("✅ ModelConfig has all expected vision parameters")
        return True
        
    except Exception as e:
        print(f"❌ Error checking ModelConfig: {e}")
        return False

def test_new_files_exist():
    """Test that new files were created."""
    print("Checking new files exist...")
    
    new_files = [
        "gpt_oss/torch/vision_projector.py",
        "gpt_oss/torch/vision_tower.py"
    ]
    
    for filepath in new_files:
        if not os.path.exists(filepath):
            print(f"❌ Missing new file: {filepath}")
            return False
        print(f"✅ Found new file: {filepath}")
    
    return True

def main():
    print("Running Import-Only Tests for Phase 1")
    print("=" * 50)
    
    tests = [
        test_syntax_validity,
        test_modelconfig_structure, 
        test_new_files_exist
    ]
    
    all_passed = True
    for test in tests:
        try:
            if not test():
                all_passed = False
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with error: {e}")
            all_passed = False
        print()
    
    print("=" * 50)
    if all_passed:
        print("✅ All import-only tests passed!")
        print("Phase 1 implementation is syntactically correct and ready for commit.")
        return True
    else:
        print("❌ Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)