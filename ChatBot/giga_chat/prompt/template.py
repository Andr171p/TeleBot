def prompt_text(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.readlines()

    return "".join(text)

