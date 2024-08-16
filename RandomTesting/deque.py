import collections

d = collections.deque(maxlen=5)

d.append('one')
d.append('two')
d.append('three')
d.append('four')
d.append('five')
d.append('six')
d.append('seven')

print(d)
print(d[0])

