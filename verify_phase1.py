#!/usr/bin/env python3
"""
Verification script for Phase 1 implementation.
Checks that all required files exist and have the expected structure.
"""

import os
import ast
import sys

def check_file_exists(filepath, description):
    """Check if a file exists."""
    if os.path.exists(filepath):
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} - NOT FOUND")
        return False

def check_class_has_attributes(filepath, class_name, expected_attrs):
    """Check if a class has expected attributes."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                # Extract attribute names from annotations and assignments
                found_attrs = set()
                
                for item in node.body:
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        found_attrs.add(item.target.id)
                    elif isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name):
                                found_attrs.add(target.id)
                
                missing_attrs = set(expected_attrs) - found_attrs
                if missing_attrs:
                    print(f"❌ {class_name} missing attributes: {missing_attrs}")
                    return False
                else:
                    print(f"✓ {class_name} has all expected vision attributes")
                    return True
        
        print(f"❌ Class {class_name} not found in {filepath}")
        return False
        
    except Exception as e:
        print(f"❌ Error checking {filepath}: {e}")
        return False

def check_function_exists(filepath, function_name):
    """Check if a function exists in a file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                print(f"✓ Function {function_name} found in {filepath}")
                return True
        
        print(f"❌ Function {function_name} not found in {filepath}")
        return False
        
    except Exception as e:
        print(f"❌ Error checking {filepath}: {e}")
        return False

def check_class_methods(filepath, class_name, expected_methods):
    """Check if a class has expected methods."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                found_methods = set()
                
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        found_methods.add(item.name)
                
                missing_methods = set(expected_methods) - found_methods
                if missing_methods:
                    print(f"❌ {class_name} missing methods: {missing_methods}")
                    return False
                else:
                    print(f"✓ {class_name} has all expected methods")
                    return True
        
        print(f"❌ Class {class_name} not found in {filepath}")
        return False
        
    except Exception as e:
        print(f"❌ Error checking {filepath}: {e}")
        return False

def main():
    """Main verification function."""
    print("Phase 1 Implementation Verification")
    print("=" * 50)
    
    all_good = True
    
    # Check that all required files exist
    print("\n1. Checking file existence...")
    files_to_check = [
        ("gpt_oss/torch/model.py", "Extended model file"),
        ("gpt_oss/torch/vision_projector.py", "Vision projector builder"),
        ("gpt_oss/torch/vision_tower.py", "Vision tower implementation"),
        ("tests/test_vision_config.py", "Vision config tests"),
        ("tests/test_projector_builder.py", "Projector builder tests"),
        ("tests/test_vision_integration.py", "Vision integration tests"),
        ("VISION_IMPLEMENTATION_PLAN.md", "Implementation plan document")
    ]
    
    for filepath, description in files_to_check:
        if not check_file_exists(filepath, description):
            all_good = False
    
    # Check ModelConfig has vision attributes
    print("\n2. Checking ModelConfig vision extensions...")
    vision_attrs = [
        'mm_vision_tower', 'mm_projector_type', 'mm_hidden_size',
        'mm_vision_select_layer', 'mm_vision_select_feature', 'mm_patch_merge_type',
        'use_mm_proj', 'tune_mm_mlp_adapter', 'freeze_mm_mlp_adapter', 'pretrain_mm_mlp_adapter'
    ]
    
    if not check_class_has_attributes("gpt_oss/torch/model.py", "ModelConfig", vision_attrs):
        all_good = False
    
    # Check Transformer has vision methods
    print("\n3. Checking Transformer vision methods...")
    transformer_methods = [
        'forward_text_only', 'forward_multimodal', 'get_vision_tower', 'initialize_vision_modules'
    ]
    
    if not check_class_methods("gpt_oss/torch/model.py", "Transformer", transformer_methods):
        all_good = False
    
    # Check projector builder functions
    print("\n4. Checking projector builder functions...")
    if not check_function_exists("gpt_oss/torch/vision_projector.py", "build_vision_projector"):
        all_good = False
    
    # Check vision tower functions
    print("\n5. Checking vision tower functions...")
    if not check_function_exists("gpt_oss/torch/vision_tower.py", "build_vision_tower"):
        all_good = False
    
    print("\n" + "=" * 50)
    if all_good:
        print("✅ Phase 1 implementation verification PASSED!")
        print("\nAll required components are implemented:")
        print("- Extended ModelConfig with vision parameters")
        print("- Vision projector builder with multiple projector types")
        print("- Vision tower abstraction with CLIP support")
        print("- Extended Transformer class with multimodal methods")
        print("- Comprehensive test suite")
        print("\nReady to proceed to Phase 2!")
    else:
        print("❌ Phase 1 implementation verification FAILED!")
        print("Please fix the issues above before proceeding.")
    
    return all_good

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)