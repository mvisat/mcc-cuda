from PIL import Image

arr = []
for s in open('area.txt'):
  arr.append(s.rstrip())

w = len(arr[0])
h = len(arr)

im = Image.new('RGB', (w,h))
px = im.load()
for i in range(h):
  for j in range(w):
    if arr[i][j] == '0':
      px[j,i] = 255,255,255
    else:
      px[j,i] = 0,0,0
im.save('area.jpg')
