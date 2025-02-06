import torch

def remove_text_after_comma(s):
    paren_level = 0
    for i, c in enumerate(s):
        if c == '(':
            paren_level += 1
        elif c == ')':
            paren_level -= 1
        elif c == ',' and paren_level == 0:
            return s[:i]
    return s

def return_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')