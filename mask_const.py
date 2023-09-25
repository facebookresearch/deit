import numpy as np

def _generate_masks(xs, ys):
    masks = []
    zeros = np.zeros([14,14], dtype=bool)
    y_offset = 0
    for y in ys:
        x_offset = 0
        for x in xs:
            mask = zeros.copy()
            mask[y_offset:y_offset+y, x_offset:x_offset+x] = 1
            masks.append(mask)
            x_offset += x
        y_offset += y
    return masks

DIVISION_SPECS_14_14 = {
    2: {"xs": [14], "ys": [7, 7]},
    4: {"xs": [7, 7], "ys": [7, 7]},
    8: {"xs": [7, 7], "ys": [4, 3, 3, 4]},
    16: {"xs": [4, 3, 3, 4], "ys": [4, 3, 3, 4]},
    3: {"xs": [14], "ys": [5, 4, 5]},
    6: {"xs": [7, 7], "ys": [5, 4, 5]},
    9: {"xs": [5, 4, 5], "ys": [5, 4, 5]},
    12: {"xs": [4, 3, 3, 4], "ys": [5, 4, 5]},
}

DIVISION_MASKS_WIDE_14_14 = {k: _generate_masks(**v) for k, v in DIVISION_SPECS_14_14.items()}
DIVISION_MASKS_14_14 = {k: [masks, [np.rot90(m).copy() for m in masks]] for k, masks in DIVISION_MASKS_WIDE_14_14.items()}