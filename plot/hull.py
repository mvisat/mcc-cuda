from PIL import Image, ImageDraw

f = open('hull.txt')

arr = []
w = int(f.readline())
h = int(f.readline())
n = int(f.readline())
im = Image.new('RGB', (w,h), 'black')
px = im.load()
draw = ImageDraw.Draw(im)

for _ in range(n):
  s = f.readline().split()
  x, y = int(s[0]), int(s[1])
  arr.append((x,y))
arr.append(arr[0])
draw.line(arr)
im.save('hull.png')
