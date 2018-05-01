import numpy as np
from skimage.io import imsave
import os
import socket
import struct

def write_asc(path, vertices):
    '''
    Args:
        vertices: shape = (nver, 3)
    '''
    if path.split('.')[-1] == 'asc':
        np.savetxt(path, vertices)
    else:
        np.savetxt(path + '.asc', vertices)

def send_udp(obj_name, vertices, colors, triangles):
    dstip = "127.0.0.1"
    dstport = 6000
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    data = bytearray()

    STEP = 1000
    count = 0
    startIndex = 0

    for i in range(vertices.shape[0]):
        # s = 'v {} {} {} \n'.format(vertices[0,i], vertices[1,i], vertices[2,i])
        # s = 'v {} {} {} {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2], colors[i, 0], colors[i, 1], colors[i, 2])
        data.extend(struct.pack('f', vertices[i, 0]));
        data.extend(struct.pack('f', vertices[i, 1]));
        data.extend(struct.pack('f', vertices[i, 2]));
        data.extend(struct.pack('f', colors[i, 0]));
        data.extend(struct.pack('f', colors[i, 1]));
        data.extend(struct.pack('f', colors[i, 2]));
        count += 1
        if count >= STEP or i == vertices.shape[0] - 1:
            data = struct.pack('i', startIndex) + struct.pack('i', count) + data
            sock.sendto(data, (dstip, dstport))
            count = 0
            startIndex = i + 1
            data = bytearray()



def write_obj(obj_name, vertices, colors, triangles):
    ''' Save 3D face model
    Args:
        obj_name: str
        vertices: shape = (nver, 3)
        colors: shape = (nver, 3)
        triangles: shape = (ntri, 3)
    '''
    triangles = triangles.copy()
    triangles += 1 # meshlab start with 1
    
    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'
        
    # write obj
    with open(obj_name, 'w') as f:
        
        # write vertices & colors
        for i in range(vertices.shape[0]):
            # s = 'v {} {} {} \n'.format(vertices[0,i], vertices[1,i], vertices[2,i])
            s = 'v {} {} {} {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2], colors[i, 0], colors[i, 1], colors[i, 2])
            f.write(s)

        # write f: ver ind/ uv ind
        [k, ntri] = triangles.shape
        for i in range(triangles.shape[0]):
            # s = 'f {}/{} {}/{} {}/{} \n'.format(triangles[i, 0], triangles[i, 0], triangles[i, 1], triangles[i, 1], triangles[i, 2], triangles[i, 2])
            s = 'f {} {} {}\n'.format(triangles[i, 0], triangles[i, 1], triangles[i, 2])
            f.write(s)


def write_obj_with_texture(obj_name, vertices, colors, triangles, texture, uv_coords):
    ''' Save 3D face model with texture. Ref: https://github.com/patrikhuber/eos/blob/bd00155ebae4b1a13b08bf5a991694d682abbada/include/eos/core/Mesh.hpp
    Args:
        obj_name: str
        vertices: shape = (nver, 3)
        colors: shape = (nver, 3)
        triangles: shape = (ntri, 3)
        texture: shape = (256,256,3)
        uv_coords: shape = (nver, 3) max value<=1
    '''
    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'
    mtl_name = obj_name.replace('.obj', '.mtl')
    texture_name = obj_name.replace('.obj', '_texture.png')
    
    triangles = triangles.copy()
    triangles += 1 # mesh lab start with 1
    
    # write obj
    with open(obj_name, 'wb') as f:
        # first line: write mtlib(material library)
        s = "mtllib {}\n".format(os.path.abspath(mtl_name))
        f.write(s)

        # write vertices
        for i in range(vertices.shape[0]):
            s = 'v {} {} {} {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2], colors[i, 0], colors[i, 1], colors[i, 2])
            f.write(s)
        
        # write uv coords
        for i in range(uv_coords.shape[0]):
            s = 'vt {} {}\n'.format(uv_coords[i,0], 1 - uv_coords[i,1])
            f.write(s)

        f.write("usemtl FaceTexture\n")

        # write f: ver ind/ uv ind
        for i in range(triangles.shape[0]):
            s = 'f {}/{} {}/{} {}/{}\n'.format(triangles[i,0], triangles[i,0], triangles[i,1], triangles[i,1], triangles[i,2], triangles[i,2])
            f.write(s)

    # write mtl
    with open(mtl_name, 'wb') as f:
        f.write("newmtl FaceTexture\n")
        s = 'map_Kd {}\n'.format(os.path.abspath(texture_name)) # map to image
        f.write(s)

    # write texture as png
    imsave(texture_name, texture)