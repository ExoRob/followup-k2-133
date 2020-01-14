def HMS2deg(ra='', dec=''):
    RA, DEC, rs, ds = '', '', 1, 1
    if dec:
        D, M, S = [float(i) for i in dec.split()]
        if str(D)[0] == '-':
            ds, D = -1, abs(D)
        deg = D + (M / 60) + (S / 3600)
        DEC = '{0}'.format(deg * ds)

    if ra:
        H, M, S = [float(i) for i in ra.split()]
        if str(H)[0] == '-':
            rs, H = -1, abs(H)
        deg = (H * 15) + (M / 4) + (S / 240)
        RA = '{0}'.format(deg * rs)

    return float(RA), float(DEC)


ra1, dec1 = HMS2deg("04 19 24.2904491684", "26 14 20.265563399")
ra2, dec2 = HMS2deg("04 40 35.6389148796", "25 00 36.047186428")

dist = ((ra1 - ra2)**2. + (dec1 - dec2)**2.)**0.5

print dist
