"""
Reads JurisdictionsMetadata.json and prints the total case count across all jurisdictions.
"""

import json

with open('JurisdictionsMetadata.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

total_case_count = 0
missing = []

for jurisdiction in data:
    try:
        count = jurisdiction['case_count']
        total_case_count += count
        print(f"{jurisdiction['name']}: {count}")
    except KeyError:
        missing.append(jurisdiction['name'])

if missing:
    print(f"Jurisdictions missing case_count: {missing}")

print(f"Total cases: {total_case_count}")
