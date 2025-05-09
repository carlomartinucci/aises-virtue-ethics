# -*- coding: utf-8 -*-
"""
Shared configuration for scenario file limits.
"""
from collections import defaultdict

# Default file limit to use if a scenario is not explicitly listed below.
DEFAULT_FILE_LIMIT = 50

SCENARIO_SFT_FILE_LIMITS = defaultdict(
    lambda: DEFAULT_FILE_LIMIT,
    {
        "ethicsunwrapped": 55,
        "murdoughcenter": 33,
        "markkula": 82
        # Add other SFT-relevant scenarios and their limits here
    }
)
