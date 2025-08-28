
def read_txt(filename) -> str:
    with open(filename, "r", encoding="utf-8-sig") as file:
        content = file.read()  # Read the entire content of the file
        return content
