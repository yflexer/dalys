import matplotlib.colors as mcolors

marker_dict = {".": "point", ",": "pixel", "o": "circle", "v": "triangle_down",
               "^": "triangle_up", "<": "triangle_left", ">": "triangle_right",
               "1": "tri_down", "2": "tri_up", "3": "tri_left", "4": "tri_right",
               "8": "octagon", "s": "square", "p": "pentagon", "*": "star",
               "h": "hexagon1", "H": "hexagon2", "+": "plus", "D": "diamond",
               "d": "thin_diamond"}

exclude_key = ['white', 'xkcd:white', 'floralwhite', 'whitesmoke']
std_color = tuple(key for key in mcolors.BASE_COLORS if key not in exclude_key)
css4_color = tuple(key for key in mcolors.CSS4_COLORS if key not in exclude_key)
xkcd_color = tuple(key for key in mcolors.XKCD_COLORS if key not in exclude_key)
tab_color = tuple(key for key in mcolors.TABLEAU_COLORS if key not in exclude_key)
colors = tuple(set(xkcd_color + css4_color))
