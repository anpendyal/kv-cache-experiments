import json

# Open and load the JSON file
with open('JurisdictionsMetadata.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

total_case_count = 0
errors = []
for jurisdiction in data:
    
    print(jurisdiction['name'])
    try: 
        total_case_count += jurisdiction['case_count']
        print(jurisdiction['case_count'])
    except:
        errors.append(jurisdiction['name'])
    print()

print(errors)
print(total_case_count)