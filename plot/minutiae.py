from PIL import Image

name = input("file name: ")
f = open(name)

w = int(f.readline())
h = int(f.readline())
f.readline()
n = int(f.readline())
im = Image.new('RGB', (w,h), 'black')
px = im.load()

for _ in range(n):
  s = f.readline().split()
  x, y = int(s[0]), int(s[1])
  px[x,y] = 255,255,255
f.close()
im.save('minutiae.jpg')
