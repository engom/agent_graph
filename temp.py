# Map visit codes to visit numbers
visit_map = {
    "SCREENING": 1,
    "BASELINE": 2,
    "WEEK 1": 3,
    "WEEK 2": 4,
    "WEEK 4": 5,
    "WEEK 8": 6,
    "WEEK 12": 7,
    "FOLLOW-UP": 8,
    "UNSCHEDULED": 99
}

if VISITC is None:
    0
else:
    visit_map.get(VISITC.upper(), 0)