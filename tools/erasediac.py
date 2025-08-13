import re

def remove_hebrew_punctuation(text):
    # ניקוד עברי: טווח התווים מ־\u0591 עד \u05C7 (כולל טעמי מקרא)
    hebrew_nikud_pattern = re.compile(r"[\u0591-\u05C7]")
    # סימני פיסוק רגילים
    punctuation_pattern = re.compile(r"[\"',:;.!?()\[\]{}״׳־–]")

    # הסרת הניקוד והפיסוק
    text_no_nikud = hebrew_nikud_pattern.sub('', text)
    text_clean = punctuation_pattern.sub('', text_no_nikud)
    return text_clean

def clean_hebrew_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    cleaned_lines = [remove_hebrew_punctuation(line.strip()) for line in lines]

    with open(output_path, 'w', encoding='utf-8') as outfile:
        outfile.write("\n".join(cleaned_lines))

# דוגמה לשימוש:
input_file = "setX_with_diac.txt"
output_file = "setX_no_diac.txt"
clean_hebrew_file(input_file, output_file)
