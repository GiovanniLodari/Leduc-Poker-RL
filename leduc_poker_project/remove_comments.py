import os
import re
import tokenize
import io

def remove_python_comments(content):
    io_obj = io.StringIO(content)
    out = ""
    last_lineno = -1
    last_col = 0
    try:
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col and token_type != tokenize.COMMENT:
                out += (" " * (start_col - last_col))
            if token_type != tokenize.COMMENT:
                out += token_string
            last_col = end_col
            last_lineno = end_line
        return out
    except:
        return content

def remove_html_comments(content):
    return re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)

def remove_css_js_comments(content):
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    content = re.sub(r'(?<!:)\/\/.*', '', content)
    return content

def cleanup_empty_lines(content):
    lines = content.splitlines()
    cleaned = []
    for line in lines:
        if line.strip():
            cleaned.append(line)
        elif cleaned and cleaned[-1].strip():
            cleaned.append(line)
    return "\n".join(cleaned)

def process_file(path):
    ext = os.path.splitext(path)[1].lower()
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception as e:
        print(f"Skipping {path} due to error: {e}")
        return

    new_content = content
    if ext == ".py":
        new_content = remove_python_comments(content)
    elif ext == ".html":
        new_content = remove_html_comments(content)
        new_content = remove_css_js_comments(new_content)
    elif ext in [".css", ".js"]:
        new_content = remove_css_js_comments(content)
    
    new_content = cleanup_empty_lines(new_content)
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content + "\n")

def walk_and_process(directory):
    for root, dirs, files in os.walk(directory):
        if any(skip in root for skip in ["__pycache__", "checkpoints", "logs", ".git", ".venv"]):
            continue
        for file in files:
            if file == "remove_comments.py":
                continue
            if file.endswith((".py", ".html", ".css", ".js")):
                path = os.path.join(root, file)
                print(f"Processing {path}...")
                process_file(path)

if __name__ == "__main__":
    target = "/home/giova/MachineLearningProject/leduc_poker_project"
    walk_and_process(target)
