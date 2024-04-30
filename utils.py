from contextlib import suppress


def escape_important(text):
    return text.replace("\\)", "\0\1").replace("\\(", "\0\2")


def unescape_important(text):
    return text.replace("\0\1", ")").replace("\0\2", "(")


def parse_parentheses(string):
    result = []
    current_item = ""
    nesting_level = 0
    for char in string:
        if char == "(":
            if nesting_level == 0:
                if current_item:
                    result.append(current_item)
                    current_item = "("
                else:
                    current_item = "("
            else:
                current_item += char
            nesting_level += 1
        elif char == ")":
            nesting_level -= 1
            if nesting_level == 0:
                result.append(current_item + ")")
                current_item = ""
            else:
                current_item += char
        else:
            current_item += char
    if current_item:
        result.append(current_item)
    return result


def _remove_weights(string):
    a = parse_parentheses(string)
    out = []
    for x in a:
        if len(x) >= 2 and x[-1] == ")" and x[0] == "(":
            x = x[1:-1]
            xx = x.rfind(":")
            if xx > 0:
                with suppress(Exception):
                    x = x[:xx]
            out += _remove_weights(x)
        else:
            out += [x]
    return out


def remove_weights(text: str):
    text = escape_important(text)
    parsed_weights = _remove_weights(text)
    return "".join([unescape_important(segment) for segment in parsed_weights])
