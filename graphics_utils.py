from matplotlib import pyplot as pl


def draw_plain_graphics_with_avarage_window(values, window_size):
    token_count = len(values) // window_size
    value_to_draw = list()
    for i in range(token_count):
        value_to_draw.append(sum(values[window_size*i: min(window_size*(i+1), len(values))]) / window_size)
    pl.plot(range(len(value_to_draw)), value_to_draw)
    pl.show()
