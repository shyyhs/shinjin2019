with open('neg.txt') as f:
    lines = f.readlines()

with open('n_tagged.txt', 'w') as f:
    for line in lines:
        f.write('-1\t' + line)
