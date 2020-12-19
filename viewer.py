with open("rockyou.txt", 'r', encoding = 'ISO-8859-1') as F:
    chars = set()
    for row in F:
        chars.update(list(row))

    print(''.join(sorted(chars)))

'''!"#$%&'()*+,-./0123456789;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~"'''