# Save as reverse_hebrew.py
input_file = "hebrew_text.txt"
output_file = "hebrew_text_rtl.txt"

with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        fout.write(line[::-1] + "\n")