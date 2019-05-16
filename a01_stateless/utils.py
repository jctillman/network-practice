import numpy as np

def list_equals(lst1, lst2):

    if len(lst1) != len(lst2):
        return False

    for i in range(len(lst1)):
        if lst1[i] != lst2[i]:
            return False
    
    return True

def output_dims(tensor_shape, size, stride):
    W = tensor_shape[0]
    H = tensor_shape[1]
    C = tensor_shape[2]

    dist = int((size - 1) / 2)
    
    start_x = dist
    end_x = W - dist
    end_W = W - dist * 2

    start_y = dist
    end_y = H - dist
    end_H = H - dist * 2
    return end_W, end_H

'''
Given a [W, H, C] tensor
and a size S, this transforms
the tensor to a
[Patch_Num, S x S x C]
tensor.

This enables it to be multiplied with a
[S x S x C, Out_Channels] kernel of
some kind in order to
to produce something like 
[Patch_Num, Out_Channels]
'''
def to_patches_2d(tensor, size=3, stride = 1):
    
    
    assert len(tensor.shape) == 3

    W = tensor.shape[0]
    H = tensor.shape[1]
    C = tensor.shape[2]

    dist = int((size - 1) / 2)
    
    start_x = dist
    end_x = W - dist
    end_W = W - dist * 2

    start_y = dist
    end_y = H - dist
    end_H = H - dist * 2

    patches = []
    for center_x in range(start_x, end_x):
        for center_y in range(start_y, end_y):
            left = center_x - dist
            right = center_x + dist + 1
            top = center_y - dist
            bot = center_y + dist + 1
            peeled = tensor[left:right, top:bot, :]
            patches.append(peeled.reshape(-1))
            assert peeled.size == size * size * C

    patches = np.array(patches)

    assert patches.shape[1] == size * size * C
    assert patches.shape[0] == end_W * end_H
    return patches

'''
Transforms a [Patch_Num, Out_Channels]
matrix to a [W, H, C] tensor.

Needs to be given tensor and dimensions 
to which to restore it.
'''
def from_patches_2d(patches, width, height, channels, size=1, reconstruct=False):

    assert patches.shape[0] == (width - size + 1) * (height - size + 1)
    assert patches.shape[1] % channels == 0

    ret = np.zeros((width, height, channels))

    patch = 0
    reshaped = patches.reshape((-1, size, size, channels))
    if reconstruct:
        for x in range(width - size + 1):
            for y in range(height - size + 1):
                ret[x:x+size,y:y+size,:] = reshaped[patch]
                patch = patch + 1
    else:
        for x in range(width - size + 1):
            for y in range(height - size + 1):
                ret[x:x+size,y:y+size,:] = ret[x:x+size,y:y+size,:] + reshaped[patch]
                patch = patch + 1

    return ret








    
