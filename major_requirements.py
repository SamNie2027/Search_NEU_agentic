"""
Module for loading and parsing major requirement configurations.

This module handles loading course requirement data from JSON files and
providing utilities to extract course codes for specific majors.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# Default path for CS major requirements JSON
DEFAULT_REQUIREMENTS_PATH = Path(__file__).parent / "major_requirements.json"

# Try to import major data from extract_cs_courses.py
try:
    # Add parent directory to path to import extract_cs_courses
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from extract_cs_courses import cs_major_json_data, cyber_major_json_data, data_science_major_json_data
    MAJOR_DATA_AVAILABLE = True
except ImportError:
    MAJOR_DATA_AVAILABLE = False
    cs_major_json_data = None
    cyber_major_json_data = None
    data_science_major_json_data = None


def load_requirements(json_path: str | Path | None = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load major requirements from a JSON file.
    
    Args:
        json_path: Path to the JSON file. If None, uses default path.
        
    Returns:
        Dictionary mapping major names to lists of course code dictionaries.
        Format: {"CS": [{"subject": "CS", "number": 2500}, ...]}
        
    Raises:
        FileNotFoundError: If the JSON file doesn't exist.
        json.JSONDecodeError: If the JSON is invalid.
    """
    if json_path is None:
        json_path = DEFAULT_REQUIREMENTS_PATH
    else:
        json_path = Path(json_path)
    
    if not json_path.exists():
        raise FileNotFoundError(f"Requirements file not found: {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data


def parse_course_code(course_entry: Dict[str, Any] | str) -> Optional[Tuple[str, int]]:
    """
    Parse a course code from various formats.
    
    Handles formats like:
    - {"subject": "CS", "number": 2500}
    - {"subject": "CS", "classId": 2500}  # Used in extract_cs_courses.py format
    - "CS 2500"
    - {"subject": "CS", "courseNumber": 2500}
    
    Args:
        course_entry: Course code in various formats
        
    Returns:
        Tuple of (subject, number) or None if invalid
    """
    if isinstance(course_entry, str):
        # Handle "CS 2500" format
        parts = course_entry.strip().split()
        if len(parts) >= 2:
            try:
                subject = parts[0].upper()
                number = int(parts[1])
                return (subject, number)
            except (ValueError, IndexError):
                return None
        return None
    
    if isinstance(course_entry, dict):
        # Handle {"subject": "CS", "number": 2500} or {"subject": "CS", "courseNumber": 2500}
        # Also handle {"subject": "CS", "classId": 2500} format from extract_cs_courses.py
        subject = course_entry.get("subject") or course_entry.get("Subject")
        number = (course_entry.get("number") or 
                 course_entry.get("courseNumber") or 
                 course_entry.get("Number") or
                 course_entry.get("classId"))  # Support classId format
        
        if subject and number is not None:
            try:
                subject = str(subject).upper().strip()
                number = int(number)
                return (subject, number)
            except (ValueError, TypeError):
                return None
    
    return None


def extract_all_course_codes(requirements_data: Dict[str, Any]) -> List[Tuple[str, int]]:
    """
    Extract all unique course codes from a requirements data structure.
    
    Recursively traverses the JSON structure to find all course codes,
    ignoring concentrations and optional paths as specified.
    
    Args:
        requirements_data: The loaded JSON data (can be nested)
        
    Returns:
        List of unique (subject, number) tuples
    """
    course_codes = set()
    
    def traverse(obj: Any) -> None:
        """Recursively traverse the JSON structure."""
        if isinstance(obj, dict):
            # Check if this dict represents a course (has subject and classId/number)
            parsed = parse_course_code(obj)
            if parsed:
                course_codes.add(parsed)
            
            # Skip concentration/optional keys and metadata
            skip_keys = {"concentration", "concentrations", "optional", "optionals", 
                        "choose", "select", "elective", "electives", "metadata",
                        "name", "totalCreditsRequired", "yearVersion", "minOptions",
                        "concentrationOptions", "minRequirementCount"}
            
            for key, value in obj.items():
                key_lower = str(key).lower()
                # Skip concentration sections and metadata
                if any(skip in key_lower for skip in skip_keys):
                    continue
                traverse(value)
                
        elif isinstance(obj, list):
            for item in obj:
                traverse(item)
        elif isinstance(obj, str):
            parsed = parse_course_code(obj)
            if parsed:
                course_codes.add(parsed)
    
    traverse(requirements_data)
    return sorted(list(course_codes))


def _get_major_data_from_python(major: str) -> Dict[str, Any] | None:
    """
    Get major data from Python dictionaries in extract_cs_courses.py.
    
    Args:
        major: Major name (e.g., "CS", "Cyber", "Data Science")
        
    Returns:
        Major data dictionary or None if not found
    """
    if not MAJOR_DATA_AVAILABLE:
        return None
    
    major_upper = major.upper().strip()
    major_lower = major.lower().strip()
    
    # Map major names to their data
    major_mapping = {
        "CS": cs_major_json_data,
        "COMPUTER SCIENCE": cs_major_json_data,
        "CYBER": cyber_major_json_data,
        "CYBERSECURITY": cyber_major_json_data,
        "CY": cyber_major_json_data,
        "DATA SCIENCE": data_science_major_json_data,
        "DS": data_science_major_json_data,
    }
    
    # Try exact match first
    if major_upper in major_mapping:
        return major_mapping[major_upper]
    
    # Try case-insensitive match
    for key, value in major_mapping.items():
        if key.upper() == major_upper or key.lower() == major_lower:
            return value
    
    return None


def get_requirement_codes(major: str, requirements: Dict[str, Any] | None = None, 
                         json_path: str | Path | None = None) -> List[Tuple[str, int]]:
    """
    Get course codes for a specific major from requirements data.
    
    Args:
        major: Major name (e.g., "CS", "Cyber", "Data Science")
        requirements: Pre-loaded requirements dict. If None, loads from json_path or Python data.
        json_path: Path to requirements JSON. Only used if requirements is None.
        
    Returns:
        List of (subject, number) tuples for courses that fulfill any requirement
        for the specified major.
    """
    # First, try to get data from Python dictionaries
    major_data = _get_major_data_from_python(major)
    if major_data is not None:
        # Extract all course codes from the nested structure
        return extract_all_course_codes(major_data)
    
    # Fall back to JSON file loading
    if requirements is None:
        requirements = load_requirements(json_path)
    
    major_upper = major.upper().strip()
    
    # If major exists as a top-level key with a list of course codes
    if major_upper in requirements:
        major_data = requirements[major_upper]
        if isinstance(major_data, list):
            codes = []
            for entry in major_data:
                parsed = parse_course_code(entry)
                if parsed:
                    codes.append(parsed)
            return sorted(codes)
        else:
            # Nested structure - extract all course codes
            return extract_all_course_codes(major_data)
    
    # If the structure is different, try to extract all codes from the entire structure
    # This handles cases where the JSON structure is more complex
    all_codes = extract_all_course_codes(requirements)
    
    # Filter by major if possible (if subject matches)
    if major_upper:
        filtered = [code for code in all_codes if code[0].upper() == major_upper]
        if filtered:
            return sorted(filtered)
    
    # Return all codes if we can't filter by major
    return sorted(all_codes)


def clean_and_save_requirements(input_json_path: str | Path, 
                                output_json_path: str | Path | None = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Clean a requirements JSON file and save it in the standard format.
    
    This function extracts all course codes from a complex requirements structure,
    ignoring concentrations and optional paths, and saves a simplified version.
    
    Args:
        input_json_path: Path to the input JSON file (can be complex structure)
        output_json_path: Path to save cleaned JSON. If None, overwrites input.
        
    Returns:
        The cleaned requirements dictionary
    """
    with open(input_json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    all_codes = extract_all_course_codes(raw_data)
    
    by_subject: Dict[str, List[Dict[str, Any]]] = {}
    for subject, number in all_codes:
        if subject not in by_subject:
            by_subject[subject] = []
        by_subject[subject].append({"subject": subject, "number": number})
    
    # Sort each major's courses by number
    for subject in by_subject:
        by_subject[subject].sort(key=lambda x: x["number"])
    
    # Save cleaned version
    if output_json_path is None:
        output_json_path = input_json_path
    
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(by_subject, f, indent=2)
    
    return by_subject


__all__ = [
    "load_requirements",
    "get_requirement_codes",
    "parse_course_code",
    "extract_all_course_codes",
    "clean_and_save_requirements",
]

