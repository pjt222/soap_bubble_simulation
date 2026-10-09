import sys, zlib, struct
def read_png(path):
    data = open(path, 'rb').read(); assert data[:8] == b'\x89PNG\r\n\x1a\n'
    pos = 8; idat = b''; w = h = ct = bd = None
    while pos < len(data):
        ln = struct.unpack('>I', data[pos:pos+4])[0]; typ = data[pos+4:pos+8]; body = data[pos+8:pos+8+ln]; pos += 12+ln
        if typ == b'IHDR': w, h, bd, ct = struct.unpack('>IIBB', body[:10])
        elif typ == b'IDAT': idat += body
    raw = zlib.decompress(idat); bpp = {6: 4, 2: 3}[ct]; stride = w*bpp
    rows = []; prev = bytearray(stride); i = 0
    for _ in range(h):
        f = raw[i]; line = bytearray(raw[i+1:i+1+stride]); i += 1+stride
        for x in range(stride):
            a = line[x-bpp] if x >= bpp else 0; b = prev[x]; c = prev[x-bpp] if x >= bpp else 0
            if f == 1: line[x] = (line[x]+a) & 255
            elif f == 2: line[x] = (line[x]+b) & 255
            elif f == 3: line[x] = (line[x]+((a+b)>>1)) & 255
            elif f == 4:
                p = a+b-c; pa, pb, pc = abs(p-a), abs(p-b), abs(p-c)
                line[x] = (line[x]+(a if pa <= pb and pa <= pc else (b if pb <= pc else c))) & 255
        rows.append(line); prev = line
    return w, h, ct, rows, bpp
for p in sys.argv[1:]:
    w, h, ct, rows, bpp = read_png(p)
    px = lambda x, y: tuple(rows[y][x*bpp:(x+1)*bpp])
    print(p.split('/')[-1], f"{w}x{h} ct={ct}", "corners:", px(5, 5), px(w-6, 5), px(5, h-6), px(w-6, h-6))
