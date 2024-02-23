from dataclasses import dataclass


@dataclass
class Config:
    FILE_NAME = "IMDB Dataset.csv"
    VOCAB_SIZE = 10000
    MAX_LENGTH = 2470
    TRUNC_TYPE = 'post'
    PADDING_TYPE = 'post'
    OOV_TOKEN = "<OOV>"
