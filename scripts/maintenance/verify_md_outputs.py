#!/usr/bin/env python3
"""
Verify all control_flow_conditional .md outputs.
"""

from pathlib import Path
import sys

def verify_all_md_outputs():
    """Verify all .md output files are correct."""
    output_dir = Path('examples/outputs/control_flow_conditional')
    
    if not output_dir.exists():
        print(f"❌ Output directory does not exist: {output_dir}")
        return False
    
    # Expected files with their characteristics
    expected_files = {
        'processed_empty.md': {'size': 0, 'processing': 'Empty file', 'check': 'empty message'},
        'processed_tiny.md': {'size': 33, 'processing': 'Expanded', 'check': 'expanded content'},
        'processed_small.md': {'size': 448, 'processing': 'Expanded', 'check': 'bytes not kilobytes'},
        'processed_exact_threshold.md': {'size': 1000, 'processing': 'Expanded', 'check': 'explains X pattern'},
        'processed_repeated_x.md': {'size': 1000, 'processing': 'Expanded', 'check': 'explains X pattern'},
        'processed_just_over.md': {'size': 1001, 'processing': 'Compressed', 'check': 'bullet points'},
        'processed_large.md': {'size': 4920, 'processing': 'Compressed', 'check': 'bullet points'},
        'processed_sample.md': {'size': 150, 'processing': 'Expanded', 'check': 'expanded content'},
        'processed_special_chars.md': {'size': 52, 'processing': 'Expanded', 'check': 'handles special chars'},
        'processed_multiline.md': {'size': 481, 'processing': 'Expanded', 'check': 'handles multiline'},
        'processed_long_line.md': {'size': 2000, 'processing': 'Compressed', 'check': 'bullet points with content'},
        'processed_repeated_a.md': {'size': 2000, 'processing': 'Compressed', 'check': 'correct count'},
        'processed_medium_repetitive.md': {'size': 420, 'processing': 'Expanded', 'check': 'expanded content'}
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
        
        # Check it's a markdown file with proper headers
        if content.startswith("# Processed File"):
            checks.append("✓ markdown")
        else:
            checks.append("✗ not markdown")
            all_good = False
        
        # Check processing type
        if f"Processing type: {expected['processing']}" in content:
            checks.append("✓ type")
        else:
            checks.append("✗ wrong type")
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
                checks.append("✓ empty")
            else:
                checks.append("✗ no empty msg")
                all_good = False
                
        elif expected['check'] == 'expanded content':
            if len(content) > 500 and "## Result" in content:
                checks.append("✓ expanded")
            else:
                checks.append("✗ not expanded")
                all_good = False
                
        elif expected['check'] == 'bytes not kilobytes':
            if "kilobytes" not in content.lower() and "448 bytes" in content:
                checks.append("✓ units")
            else:
                checks.append("✗ wrong units")
                all_good = False
                
        elif expected['check'] == 'explains X pattern':
            if "repetitive" in content.lower() or "pattern" in content.lower():
                if content.count('X' * 50) == 0:  # Should NOT just be X's
                    checks.append("✓ explains")
                else:
                    checks.append("✗ just X's")
                    all_good = False
            else:
                checks.append("✗ no explanation")
                all_good = False
                
        elif expected['check'] == 'bullet points':
            if "•" in content and content.count("•") >= 3:
                checks.append("✓ bullets")
            else:
                checks.append("✗ no bullets")
                all_good = False
                
        elif expected['check'] == 'handles special chars':
            if "émoji" in content.lower() or "special" in content.lower():
                checks.append("✓ special")
            else:
                checks.append("✗ no special")
                all_good = False
                
        elif expected['check'] == 'handles multiline':
            if "line" in content.lower() and len(content) > 500:
                checks.append("✓ multiline")
            else:
                checks.append("✗ multiline issue")
                all_good = False
                
        elif expected['check'] == 'bullet points with content':
            bullets = content.count("•")
            if bullets >= 3:
                lines = content.split('\n')
                bullet_lines = [l for l in lines if l.strip().startswith("•")]
                if all(len(l.strip()) > 5 for l in bullet_lines):
                    checks.append("✓ bullets OK")
                else:
                    checks.append("✗ empty bullets")
                    all_good = False
            else:
                checks.append("✗ no bullets")
                all_good = False
                
        elif expected['check'] == 'correct count':
            # For repeated_a which should have count of 2000 A's
            if "2000" in content or "2,000" in content:
                checks.append("✓ count")
            else:
                checks.append("✗ wrong count")
                all_good = False
        
        status = "✅" if all("✓" in c for c in checks) else "⚠️"
        results.append(f"{status} {filename}: {', '.join(checks)}")
    
    print("\n" + "=" * 80)
    print("VERIFICATION RESULTS FOR .MD FILES")
    print("=" * 80)
    
    for result in results:
        print(result)
    
    print("\n" + "-" * 80)
    if all_good:
        print("✅ ALL .MD FILES VERIFIED SUCCESSFULLY!")
        return True
    else:
        print("⚠️ Some issues found - review above")
        return False

if __name__ == "__main__":
    success = verify_all_md_outputs()
    sys.exit(0 if success else 1)