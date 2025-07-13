"""
Specific tests to cover lines 148 and 150 in ambiguity_resolver.py.

These lines are in the _classify_ambiguity method and return 'boolean' and 'number'
respectively when specific conditions are met.
"""

import pytest
from orchestrator.compiler.ambiguity_resolver import AmbiguityResolver


class TestAmbiguityLines148And150:
    """Test cases specifically for lines 148 and 150."""
    
    def test_line_148_boolean_classification(self):
        """Test line 148: return 'boolean' when choose/select + true/false."""
        resolver = AmbiguityResolver()
        
        # These should all trigger line 148
        # The key is: has "choose" or "select", does NOT have ":" and ",", has "true" or "false"
        test_cases = [
            ("Choose true", "config.option"),
            ("Select false", "config.flag"),
            ("choose true option", "settings.value"),
            ("select false setting", "params.toggle"),
            ("Choose between true and false", "config.bool"),
            ("Select true for this", "option.value"),
        ]
        
        for content, context in test_cases:
            result = resolver._classify_ambiguity(content, context)
            assert result == "boolean", f"Failed for '{content}' - got {result}"
            
    def test_line_150_number_classification(self):
        """Test line 150: return 'number' when choose/select + number keywords."""
        resolver = AmbiguityResolver()
        
        # These should all trigger line 150
        # The key is: has "choose" or "select", does NOT have ":" and ",", has number keywords
        test_cases = [
            ("Choose number", "config.value"),
            ("Select size", "config.dimension"),
            ("choose count value", "settings.total"),
            ("select amount needed", "params.quantity"),
            ("Choose the number of items", "config.limit"),
            ("Select size for buffer", "option.buffer"),
        ]
        
        for content, context in test_cases:
            result = resolver._classify_ambiguity(content, context)
            assert result == "number", f"Failed for '{content}' - got {result}"
    
    def test_verify_conditions_for_line_148(self):
        """Verify the exact conditions that should trigger line 148."""
        resolver = AmbiguityResolver()
        
        # This should NOT trigger line 148 (has : and ,)
        result1 = resolver._classify_ambiguity("Choose: true, false", "config.option")
        assert result1 != "boolean"  # Should be "value" due to line 142
        
        # This SHOULD trigger line 148 (no : and ,)
        result2 = resolver._classify_ambiguity("Choose true", "config.option")
        assert result2 == "boolean"
        
    def test_verify_conditions_for_line_150(self):
        """Verify the exact conditions that should trigger line 150."""
        resolver = AmbiguityResolver()
        
        # This should NOT trigger line 150 (has : and ,)
        result1 = resolver._classify_ambiguity("Choose: number, size", "config.option")
        assert result1 != "number"  # Should be "value" due to line 142
        
        # This SHOULD trigger line 150 (no : and ,)
        result2 = resolver._classify_ambiguity("Choose number", "config.option")
        assert result2 == "number"