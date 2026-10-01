import pymupdf

pdf = pymupdf.open("02_Изометрии_10_листов.pdf")
DPI = 320
zoom = DPI / 72
matrix = pymupdf.Matrix(zoom, zoom)

for i, page in enumerate(pdf, start=1):
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    out = f"dataset/list_{i:02d}.png"
    pix.save(out)
    print(out, pix.width, "x", pix.height)