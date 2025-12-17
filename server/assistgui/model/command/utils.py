class Time:
    def __init__(self, time_str):
        self.time_str = time_str
        self.time_int = self.time_to_int(time_str)

    def time_to_int(self, time_str):
        hh, mm, ss, ff = map(int, time_str.split(":"))
        return ((hh * 3600 + mm * 60 + ss) * 100) + ff

    def int_to_time(self, time_int):
        ff = time_int % 100
        time_int //= 100
        ss = time_int % 60
        time_int //= 60
        mm = time_int % 60
        hh = time_int // 60
        return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"

    def __add__(self, other):
        return Time(self.int_to_time(self.time_int + other.time_int))

    def __sub__(self, other):
        return Time(self.int_to_time(self.time_int - other.time_int))

    def __mul__(self, multiplier):
        return Time(self.int_to_time(self.time_int * multiplier))

    def __truediv__(self, divisor):
        return Time(self.int_to_time(self.time_int // divisor))

    def __str__(self):
        return self.time_str


def format_gui(data, indent=0, in_elements=False, inner_elements=False):
    lines = []
    if isinstance(data, dict):
        for key, value in data.items():
            if key == 'elements':
                lines.append(' ' * indent + str(key) + ':')
                lines.extend(format_gui(value, indent + 2, True))
            elif key in ['rectangle', 'position']:
                lines.append(' ' * indent + str(key) + ': ' + str(value))
            elif key in ['type']:
                continue
            else:
                lines.append(' ' * indent + str(key) + ':')
                lines.extend(format_gui(value, indent + 2))
    elif isinstance(data, list):
        if in_elements:
            for value in data:
                lines.extend(format_gui(value, indent, False, True))
        elif inner_elements:
            element_line = []
            for element in data:
                if type(element) is dict:
                    name = element.get('name', '')
                    rectangle = element.get('rectangle', [])
                    position = element.get('position', [])
                    if position:
                        element_line.append(f"{name} {position}")
                    else:
                        element_line.append(f"{name} {rectangle}")
            lines.append(' ' * indent + ', '.join(element_line))
        else:
            for value in data:
                lines.extend(format_gui(value, indent))
    else:
        return [' ' * indent + str(data)]
    return lines


def compress_gui(com_gui):
    # compress gui
    for window_name, window_data in com_gui.items():
        for panel_item in window_data:
            for row in panel_item.get("elements", []):
                if type(row) is list:
                    for element in row:
                        try:
                            element['position'] = [int((element['rectangle'][0] + element['rectangle'][2]) / 2),
                                                   int((element['rectangle'][1] + element['rectangle'][3]) / 2)]
                            del element['rectangle']
                        except TypeError:
                            print(element, row, panel_item)
                elif type(row) is dict:
                    row['position'] = [int((row['rectangle'][0] + row['rectangle'][2]) / 2),
                                       int((row['rectangle'][1] + row['rectangle'][3]) / 2)]
                    del row['rectangle']

    return com_gui
