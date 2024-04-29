def seconds_to_time(seconds):
    mint, sec = divmod(seconds, 60)
    hour, mint = divmod(mint, 60)
    # return (hour, mint, round(sec, 2))
    return f'{hour} hr {mint} min {sec:.2f} s'
    
def time_to_seconds(time_str):
    h, _, m, _, s, _ = time_str.split()
    return float(h) * 3600 + float(m) * 60 + float(s)