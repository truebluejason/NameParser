import pandas as pd


df = pd.read_csv("data/cleaned.csv")
firstnames, middlenames, lastnames = df['first'].dropna(), df['middle'].dropna(), df['last'].dropna()

def process_names(name_type, names, max_name_len = 12):
    name_dict = {}
    for name in names:
        if name != name: continue
        if len(name)>max_name_len: continue
        if not name.isascii(): continue
        if name in name_dict: name_dict[name] += 1
        else: name_dict[name] = 1
    name_df = {'name': [], 'count': []}
    for name, count in name_dict.items():
        name_df['name'].append(name)
        name_df['count'].append(count)
    name_df = pd.DataFrame(name_df)
    name_df.to_csv(f"data/{name_type}.csv", index=False)

process_names("first", firstnames, 12)
process_names("middle", middlenames, 12)
process_names("last", lastnames, 20)