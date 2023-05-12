def find_indices_type_value(list_to_check, type_to_find):
    indices = []
    for idx, value in enumerate(list_to_check):
        if type(value) == type_to_find:
            indices.append(idx)
    return indices

def find_indices_item_to_find(list_to_check, item_to_find):
    indices = []
    for idx, value in enumerate(list_to_check):
        if value == item_to_find:
            indices.append(idx)
    return indices