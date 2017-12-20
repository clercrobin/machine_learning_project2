# -*- coding: utf-8 -*-


tempdictionary = open('dictionaries/dictionary4.txt', 'rb')
l=[]
i = 0
for word in tempdictionary:
    word = word.decode('utf8')
    word = word.strip()
    print(i)
    if i%2==0:
        l.append(word)
    else:
        l[(i-1)//2] = l[(i-1)//2]+" "+word
    print(l)
    i+=1
tempdictionary.close()