def clean_filename(name: str) -> str:
    name = name.replace("_", " ")
    name = name.replace("Produksi", "")
    name = name.replace("menurut Provinsi", "")
    name = name.replace("2021-2022", "")
    return name.strip().lower()