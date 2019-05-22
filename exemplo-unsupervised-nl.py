# clean() is a modified version of extract_signature() found in bruteforce.py in the GitHub repository linked above
cleaned_email, _ = clean(email)

lines = cleaned_email.split('\n')
lines = [line for line in lines if line != '']
cleaned_email = ' '.join(lines)