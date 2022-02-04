def parse_coords(current_list):
    coords_for_current_part = []
    lat_long_pairs = current_list.split(',')
    for pair in lat_long_pairs:
        split_pair = pair.split()
        longitude = float(split_pair[0])
        latitude = float(split_pair[1])
        coords_for_current_part.append([longitude, latitude])    
    return coords_for_current_part

def create_recursive_list(delim_start, delim_end, current_list):
    if delim_start == "" or delim_end == "":
        raise ValueError("invalid delimiters")
    start_pos = current_list.find(delim_start)
    end_pos = current_list.find(delim_end)
    current_selection = current_list[start_pos+1:end_pos+len(delim_end)-1]
    new_delim_start = delim_start[1:]
    new_delim_end = delim_end[1:]
    if current_selection[0] == '(' and current_selection[-1] ==')':
        current_start_idx = current_selection.find(new_delim_start)
        coords_for_current_list = []
        while current_start_idx != -1:
            current_end_idx = current_selection.find(new_delim_end)
            current_part_selection = current_selection[current_start_idx:current_end_idx+len(new_delim_end)]
            coords_for_current_list.append(create_recursive_list(new_delim_start, new_delim_end, current_part_selection))
            current_selection = current_selection[current_end_idx+len(new_delim_end):]
            current_start_idx = current_selection.find(new_delim_start)
        return coords_for_current_list
    elif current_selection[0] == '(' or current_selection[-1] == ')':
        raise ValueError("improper start and end")
    else:
        return parse_coords(current_selection)
        

def get_polygons_and_other_fields(filepath):
    coords_for_all_lines = []
    other_fields_for_all_lines = []
    with open(filepath, "r") as f:
        cnt = 0
        lines = f.readlines()
        print(len(lines))
        keys = lines[0].split(';')
        excel_start_utf = '\ufeff'
        if keys[0].startswith(excel_start_utf):
            keys[0] = keys[0][len(excel_start_utf):]
        newline = '\n'
        if keys[-1].endswith(newline):
            keys[-1] = keys[-1][:-len(newline)]
        print(keys)
        used_keys = keys[1:]
        for line in lines[1:]:
            fields = line.split(';')
            coords_for_all_lines.append(create_recursive_list("(((", ")))", fields[0]))
            current_line_fields = {}
            for key, field in zip(used_keys, fields[1:]):
                current_line_fields[key] = field
            other_fields_for_all_lines.append(current_line_fields)
    return coords_for_all_lines, other_fields_for_all_lines
















if __name__ == "__main__":
    coords_for_all_lines, other_fields_for_all_lines = get_polygons_and_other_fields("nynta.csv")
    print(coords_for_all_lines[0][0])
    print(other_fields_for_all_lines[0])