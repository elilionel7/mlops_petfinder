"""
dict
key: what sort of encoding needed to be performed
values: columns that need to be encoded base of their key(encoding)
"""
COLS_CONFIG = {
    "one_hot_encode_cols": ["Type", "Gender"],
    "label_encode_cols": ["Vaccinated", "Sterilized", "Color1", "Color2"],
    "ordinal_encode_cols": {
        "Health": ["Healthy", "Minor Injury", "Serious Injury"],
        "FurLength": ["Short", "Medium", "Long"],
        "MaturitySize": ["Small", "Medium", "Large"],
    },
    "count_encode_col": "Breed1",
    "target_col": "Adopted",
}
