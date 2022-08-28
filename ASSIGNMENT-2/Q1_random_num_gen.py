def My_Random(x0):
    a = 1103515245
    c = 12345
    m = 32768
    y = (((a*(x0)+c))%m)    
    return y