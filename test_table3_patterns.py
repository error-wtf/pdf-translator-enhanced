"""Test Table 3 specific patterns."""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from scientific_postprocessor import ScientificPostProcessor, RepairMode

processor = ScientificPostProcessor(RepairMode.SAFE_REPAIR)

# Test Table 3 specific patterns
tests = [
    # B1: 10 + kaputter Exponent
    ('10?25 s/s', '10^{-25} s/s'),
    ('10?19 s/s', '10^{-19} s/s'),
    
    # Units
    ('s?s', 's/s'),
    ('s/ s', 's/s'),
    
    # Existing patterns
    ('10?¹?', '10^{-19}'),
    ('(1,1?¹?)', '(1.1 × 10^{-19})'),
    ('1,1?¹?', '1.1 × 10^{-19}'),
    
    # Standard notation
    ('1.1 × 10^{-25}', '1.1 × 10^{-25}'),  # Should pass through unchanged
]

print('=== Table 3 Pattern Tests ===')
all_ok = True
for test, expected in tests:
    result, report = processor.process(test)
    ok = expected in result
    status = 'OK' if ok else 'FAIL'
    if not ok:
        all_ok = False
    print(f'{status}: "{test}" -> "{result}" (expected: "{expected}")')

print()
print('ALL OK' if all_ok else 'SOME FAILED')
