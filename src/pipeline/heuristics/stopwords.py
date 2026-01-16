# src/pipeline/heuristics/stopwords.py

# Guaraní functional stopwords
GUARANI_STOPWORDS = {
    "ha", "pe", "upe", "upéva", "upeva",
    "re", "ne", "nde", "ore", "ñande",
    "kuéra", "kuera",
    "ramo", "jave", "rire",
    "niko", "piko", "ningo",
    "hína", "hina",
    "va", "vaʼekue", "va'ekue",
    "haguã", "hagua",
    "guarã", "guara",
    "guive", "peve",
    "rupi", "gui", "py",
    "avei", "umi", "ko", "peteĩ",
}

# Spanish noise stopwords (interference)
SPANISH_STOPWORDS = {
    "de", "del", "al",
    "la", "el", "los", "las",
    "le", "un", "uno", "una",
    "e", "o", "a",
    "que", "en", "por", "para", "con",
    "se", "sí", "no", "su",
}

# Unified set
STOPWORDS = GUARANI_STOPWORDS | SPANISH_STOPWORDS
