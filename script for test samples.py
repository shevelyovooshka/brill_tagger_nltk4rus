with open('VK_RNC.txt', 'r', encoding='utf-8') as f:
    o = []
    for t in f:
        if t == '\n':
            o.append('\n')
        t = t.split('\t')
        try:
            o.append(t[1])
        except IndexError:
            continue

with open('VK_TEST.txt', 'w', encoding='utf-8') as j:
    print(' '.join(o), file=j)
