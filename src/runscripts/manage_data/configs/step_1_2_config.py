COLUMN_NAMES_TO_DROP = [
    "ALT",
    "AST",
    "GGTP",
    "wsp, Pr,",
    "INR",
    "Erytrocyty",
    "Hb",
    "leukocyty",
    "Płytki krwi",
    "mean_for_age",
    "SD_populacji",
]
COLUMNS_TO_FILL_WITH_VALUES_FROM_NORMAL_DISTRIBUTION = [
    "age",
    "insulina 0",
    "HOMA IR",
    "Z_score_BMI",
    "GPx",
    "GSH",
    "ApoE",
    "HDL",
    "ApoA1",
    "TCH",
    "TG",
    "LDL",
    "ApoB",
    "LCAT",
    "VLDL",
    "Lp(a)",
    "TBARS",
]

TEST_SIZE_FOR_REGRESSION_MODELS = 0.1
RANDOM_STATE_FOR_REGRESSION_MODELS = 515
SEED = 300464

AGE_UNTRUSTY = 5
WEIGHT_UNTRUSTY = 40
