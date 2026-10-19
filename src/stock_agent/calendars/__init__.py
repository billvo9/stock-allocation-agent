"""
Exchange session calendars.

Modules:
    nyse        frozen NYSE full-day sessions, 2003-01-02 to 2024-12-31 (fail closed);
                the research calendar
    nyse_rules  rule-based NYSE closures and sessions, equal to the frozen list through 2024;
                the prospective path's calendars only. Two versions, resolvable by id:
                2003-01-02 to 2030-12-31 (`CALENDAR_ID`, the one consumers use) and
                2003-01-02 to 2027-12-31; a per-year table says which years' holiday
                schedules were verified against NYSE's published schedule
"""
