"""Tests for data_utils.py - specifically testing platform-agnostic line counting."""
import os
import tempfile
import pytest
from t5chem.data_utils import count_lines, LineByLineTextDataset, TaskPrefixDataset


def test_count_lines_basic():
    """Test count_lines with a basic file."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("line1\n")
        f.write("line2\n")
        f.write("line3\n")
        temp_path = f.name
    
    try:
        assert count_lines(temp_path) == 3
    finally:
        os.unlink(temp_path)


def test_count_lines_empty_file():
    """Test count_lines with an empty file."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        temp_path = f.name
    
    try:
        assert count_lines(temp_path) == 0
    finally:
        os.unlink(temp_path)


def test_count_lines_no_trailing_newline():
    """Test count_lines with a file without a trailing newline."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("line1\n")
        f.write("line2\n")
        f.write("line3")  # No trailing newline
        temp_path = f.name
    
    try:
        assert count_lines(temp_path) == 3
    finally:
        os.unlink(temp_path)


def test_count_lines_unicode():
    """Test count_lines with unicode content."""
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
        f.write("Hëllö\n")
        f.write("Wörld\n")
        f.write("😀🎉\n")
        temp_path = f.name
    
    try:
        assert count_lines(temp_path) == 3
    finally:
        os.unlink(temp_path)


def test_linebyline_dataset_line_counting():
    """Test that LineByLineTextDataset correctly counts lines."""
    from unittest.mock import Mock
    
    # Create a temporary file with test data
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O\n")
        f.write("CC1=CC=C(C=C1)C\n")
        f.write("CCO\n")
        temp_path = f.name
    
    try:
        # Use a mock tokenizer for testing (we just need to test line counting)
        tokenizer = Mock()
        dataset = LineByLineTextDataset(
            tokenizer=tokenizer,
            file_path=temp_path,
            block_size=128,
            prefix=""
        )
        
        # Check that the dataset length matches the line count
        assert len(dataset) == 3
    finally:
        os.unlink(temp_path)


def test_taskprefix_dataset_line_counting():
    """Test that TaskPrefixDataset correctly counts lines in source and target files."""
    from unittest.mock import Mock
    
    # Create temporary directory and files
    with tempfile.TemporaryDirectory() as temp_dir:
        source_path = os.path.join(temp_dir, "test.source")
        target_path = os.path.join(temp_dir, "test.target")
        
        # Create source file
        with open(source_path, 'w') as f:
            f.write("source1\n")
            f.write("source2\n")
        
        # Create target file
        with open(target_path, 'w') as f:
            f.write("target1\n")
            f.write("target2\n")
        
        # Use a mock tokenizer for testing (we just need to test line counting)
        tokenizer = Mock()
        dataset = TaskPrefixDataset(
            tokenizer=tokenizer,
            data_dir=temp_dir,
            prefix="",
            type_path="test",
            max_source_length=128,
            max_target_length=128
        )
        
        # Check that the dataset length matches the line count
        assert len(dataset) == 2


def test_taskprefix_dataset_mismatched_lengths():
    """Test that TaskPrefixDataset raises an error when source and target have different lengths."""
    from unittest.mock import Mock
    
    # Create temporary directory and files
    with tempfile.TemporaryDirectory() as temp_dir:
        source_path = os.path.join(temp_dir, "test.source")
        target_path = os.path.join(temp_dir, "test.target")
        
        # Create source file with 3 lines
        with open(source_path, 'w') as f:
            f.write("source1\n")
            f.write("source2\n")
            f.write("source3\n")
        
        # Create target file with 2 lines
        with open(target_path, 'w') as f:
            f.write("target1\n")
            f.write("target2\n")
        
        # Use a mock tokenizer for testing (we just need to test line counting)
        tokenizer = Mock()
        
        # This should raise an AssertionError
        with pytest.raises(AssertionError, match="Source file and target file don't match"):
            TaskPrefixDataset(
                tokenizer=tokenizer,
                data_dir=temp_dir,
                prefix="",
                type_path="test",
                max_source_length=128,
                max_target_length=128
            )
