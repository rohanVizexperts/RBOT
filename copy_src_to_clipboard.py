import os
import pyperclip

src_dir = os.path.join(os.path.dirname(__file__), "src")
output = []

for filename in sorted(os.listdir(src_dir)):
    filepath = os.path.join(src_dir, filename)
    if os.path.isfile(filepath):
        output.append(filename)
        output.append('.')
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            output.append(f.read())
        output.append('.')

final_output = '\n\n'.join(output)
pyperclip.copy(final_output)
print("All src files copied to clipboard in the requested format.") 