#!/usr/bin/env python3
"""
Verify all control_flow_conditional outputs.
"""

from pathlib import Path
import sys

def verify_all_outputs():
    """Verify all output files are correct."""
    output_dir = Path('examples/outputs/control_flow_conditional')
    
    if not output_dir.exists():
        print(f"❌ Output directory does not exist: {output_dir}")
        return False
    
    # Expected files with their characteristics
    expected_files = {
        'processed_empty.txt': {'size': 0, 'processing': 'Empty file', 'check': 'empty message'},
        'processed_tiny.txt': {'size': 33, 'processing': 'Expanded', 'check': 'expanded content'},
        'processed_small.txt': {'size': 448, 'processing': 'Expanded', 'check': 'bytes not kilobytes'},
        'processed_exact_threshold.txt': {'size': 1000, 'processing': 'Expanded', 'check': 'explains X pattern'},
        'processed_repeated_x.txt': {'size': 1000, 'processing': 'Expanded', 'check': 'explains X pattern'},
        'processed_just_over.txt': {'size': 1001, 'processing': 'Compressed', 'check': 'bullet points'},
        'processed_large.txt': {'size': 4920, 'processing': 'Compressed', 'check': 'bullet points'},
        'processed_sample.txt': {'size': 150, 'processing': 'Expanded', 'check': 'expanded content'},
        'processed_special_chars.txt': {'size': 52, 'processing': 'Expanded', 'check': 'handles special chars'},
        'processed_multiline.txt': {'size': 481, 'processing': 'Expanded', 'check': 'handles multiline'},
        'processed_long_line.txt': {'size': 2000, 'processing': 'Compressed', 'check': 'bullet points with content'},
        'processed_repeated_a.txt': {'size': 2000, 'processing': 'Compressed', 'check': 'correct count 10000'},
        'processed_medium_repetitive.txt': {'size': 420, 'processing': 'Expanded', 'check': 'expanded content'}
    }
    
    all_good = True
    results = []
    
    for filename, expected in expected_files.items():
        file_path = output_dir / filename
        
        if not file_path.exists():
            results.append(f"❌ {filename}: FILE MISSING")
            all_good = False
            continue
        
        content = file_path.read_text()
        checks = []
        
        # Check processing type
        if f"Processing type: {expected['processing']}" in content:
            checks.append("✓ processing type")
        else:
            checks.append("✗ wrong processing type")
            all_good = False
        
        # Check original size
        if f"Original size: {expected['size']} bytes" in content:
            checks.append("✓ size")
        else:
            checks.append("✗ wrong size")
            all_good = False
        
        # Specific content checks
        if expected['check'] == 'empty message':
            if "The input file was empty. No content to process." in content:
                checks.append("✓ empty message")
            else:
                checks.append("✗ missing empty message")
                all_good = False
                
        elif expected['check'] == 'expanded content':
            if len(content) > 500 and "## Result" in content:
                checks.append("✓ expanded")
            else:
                checks.append("✗ not expanded")
                all_good = False
                
        elif expected['check'] == 'bytes not kilobytes':
            if "kilobytes" not in content.lower() and "448 bytes" in content:
                checks.append("✓ correct units")
            else:
                checks.append("✗ wrong units")
                all_good = False
                
        elif expected['check'] == 'explains X pattern':
            if "repetitive pattern" in content.lower() and "testing" in content.lower():
                checks.append("✓ explains pattern")
            else:
                checks.append("✗ just shows X's")
                all_good = False
                
        elif expected['check'] == 'bullet points':
            if "•" in content and content.count("•") >= 3:
                checks.append("✓ has bullets")
            else:
                checks.append("✗ missing bullets")
                all_good = False
                
        elif expected['check'] == 'handles special chars':
            if "émoji" in content.lower() or "special" in content.lower():
                checks.append("✓ handles special")
            else:
                checks.append("✗ rejected special")
                all_good = False
                
        elif expected['check'] == 'handles multiline':
            if "line" in content.lower() and len(content) > 500:
                checks.append("✓ handles multiline")
            else:
                checks.append("✗ multiline issue")
                all_good = False
                
        elif expected['check'] == 'bullet points with content':
            bullets = content.count("•")
            if bullets >= 3:
                # Check each bullet has content
                lines = content.split('\n')
                bullet_lines = [l for l in lines if l.strip().startswith("•")]
                if all(len(l.strip()) > 5 for l in bullet_lines):
                    checks.append("✓ bullets with content")
                else:
                    checks.append("✗ empty bullets")
                    all_good = False
            else:
                checks.append("✗ missing bullets")
                all_good = False
                
        elif expected['check'] == 'correct count 10000':
            if "10000" in content or "10,000" in content:
                checks.append("✓ correct count")
            else:
                checks.append("✗ wrong count")
                all_good = False
        
        status = "✅" if all("✓" in c for c in checks) else "⚠️"
        results.append(f"{status} {filename}: {', '.join(checks)}")
    
    print("\n" + "=" * 80)
    print("VERIFICATION RESULTS")
    print("=" * 80)
    
    for result in results:
        print(result)
    
    print("\n" + "-" * 80)
    if all_good:
        print("✅ ALL FILES VERIFIED SUCCESSFULLY!")
        return True
    else:
        print("⚠️ Some issues found - review above")
        return False

if __name__ == "__main__":
    success = verify_all_outputs()
    sys.exit(0 if success else 1)