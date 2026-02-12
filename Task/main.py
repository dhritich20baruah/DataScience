import pandas as pd
from datetime import datetime
from dateutil.parser import parse

df = pd.read_excel("signup.xls")

initial_col = df.columns[0]
raw_data = [initial_col] + df[initial_col].tolist()

parsed_rows = []
required_cols = ["name", "email", "signup_date", "plan", "notes"]

for line in raw_data:
    if line == initial_col:
        continue

    parts = [p.strip() for p in line.split(',')]

    email_index = -1
    for i, part in enumerate(parts):
        if "@" in part:
            email_index = i
            break

    if email_index != -1:
        name = ", ".join(parts[:email_index])
        email = parts[email_index]

        remaining = parts[email_index+1:]

        date = remaining[0] if len(remaining) > 0 else ""
        plan = remaining[1] if len(remaining) > 1 else ""
        notes = ", ".join(remaining[2:]) if len(remaining) > 2 else ""

        parsed_rows.append([name, email, date, plan, notes])

df = pd.DataFrame(parsed_rows, columns=required_cols)

# Function to formate dates
def standardize_dates(d):
    try:
        for formate in ('%Y-%m-%d', '%m/%d/%y', '%d/%m/%y', '%Y/%m/%d'):
            try:
                return datetime.strptime(d, formate).strftime('%Y-%m-%d')
            except ValueError:
                continue
        return d
    except:
        return d
    
df['signup_date'] = df['signup_date'].apply(standardize_dates)

#Remove invalid dates
def is_date(string):
    try: 
        parse(string, fuzzy=False)
        return True
    except (ValueError, TypeError):
        return False
    
df = df[df["signup_date"].apply(is_date)]

#Low Quality
df_quarantine = pd.DataFrame(columns=df.columns)

check_quality = df['name'].str.contains('test user|garbage', case=False, na=False)

df_quarantine = df[check_quality].copy()

df = df[~check_quality]

#Context
df['signup_date'] = pd.to_datetime(df['signup_date'], errors='coerce')

df = df.sort_values(by=['email', 'signup_date'], ascending=True)

df['is_multi_plan'] = df.duplicated(subset=['email'], keep=False)

df = df.drop_duplicates(subset=['email'], keep='last')

#Save to CSV
df_quarantine.to_csv("quarantine.csv", index=False)
df.to_csv("members_final.csv", index=False)

print(df)
print(df_quarantine)