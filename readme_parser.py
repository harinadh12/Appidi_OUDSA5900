import os
import re
import pandas as pd


PARSE_KEYS = {
    "age": {
        'Key': 'Age',
        'delimiter': ':'
    },
    "height": {
        'Key': 'Height',
        'delimiter': ':'
    },
    "weight": {
        'Key': 'Weight',
        'delimiter': ':'
    },
    "gender": {
        'Key': 'Gender',
        'delimiter': ':'
    },
    "dominant_hand": {
        'Key': 'Dominant',
        'delimiter': ':'
    },
    "coffee_today": {
        'Key': 'Did you drink coffee today',
        'delimiter': '? '
    },
    "coffee_last_hour": {
        'Key': 'Did you drink coffee within the last hour',
        'delimiter': '? '
    },
    "sports_today": {
        'Key': 'Did you do any sports today',
        'delimiter': '? '
    },
    "smoker": {
        'Key': 'Are you a smoker',
        'delimiter': '? '
    },
    "smoke_last_hour": {
        'Key': 'Did you smoke within the last hour',
        'delimiter': '? '
    },
    "feel_ill_today": {
        'Key': 'Do you feel ill today',
        'delimiter': '? '
    }
}
    
DATA_PATH = 'WESAD/'
file_suffix = '_readme.txt'



readme_paths = {sub_dir: DATA_PATH + sub_dir + '/' 
                        for sub_dir in os.listdir(DATA_PATH)
                            if re.match('^S[0-9]{1,2}$', sub_dir)}

    
def parse_readme(subject_id):
    with open(readme_paths[subject_id] + subject_id + file_suffix, 'r') as f:
        x = f.read().split('\n')

    readme_dict = {}

    for item in x:
        for key in PARSE_KEYS.keys():
            if item.startswith(PARSE_KEYS[key]['Key']):
                k,v = item.split(PARSE_KEYS[key]['delimiter'])
                readme_dict[key] = v
    return readme_dict


def parse_all_subjects():
    
    dframes = []

    for subject_id, path in readme_paths.items():
        readme_dict = parse_readme(subject_id)
        df = pd.DataFrame(readme_dict, index=[subject_id])
        dframes.append(df)

    df = pd.concat(dframes)
    df.to_csv(DATA_PATH + 'readmes.csv')

if __name__=="__main__":
    parse_all_subjects()
