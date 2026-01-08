import numpy as np
import re

_patterns = [r"\'", r"\"", r"\.", r"<br \/>", r",", r"\(", r"\)", r"\!", r"\?", r"\;", r"\:", r"\s+"]
_replacements = [" '  ", "", " . ", " ", " , ", " ( ", " ) ", " ! ", " ? ", " ", " ", " "]
_patterns_dict = list((re.compile(p), r) for p, r in zip(_patterns, _replacements))

# package 에서 복붙해옴 걍
def basic_english_normalize(line):
    line = line.lower()
    for pattern_re, replaced_str in _patterns_dict:
        line = pattern_re.sub(replaced_str, line)
    return line

def data_prcessing(df):
    # 공백 제거
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df = df.dropna()

    df['Title'] = df['Title'].apply(basic_english_normalize)
    df['Description'] = df['Description'].apply(basic_english_normalize)
    return df.sample(frac=0.25, random_state=42) # 4분의 1만 쓴다.
