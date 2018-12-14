
write = lambda s : print(s, end='', flush=True)


EPSILON = 1e-8

def fstr(x):
    if abs(x.imag)<EPSILON and abs(x.real-int(round(x.real)))<EPSILON:
        x = int(round(x.real))
    elif abs(x.imag)<EPSILON:
        x = x.real
        for digits in range(4):
            rx = round(x, digits)
            if abs(x-rx)<EPSILON:
                x = rx
        
    return str(x)

