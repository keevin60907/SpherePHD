'''
The module is used to construct the adjcency table of the subdivision icosahedron
'''
import numpy as np

def connect(adj_table, id1, id2):
    '''
    connect two pixel on sphere

    Args:
        adj_table(list[list])   : the adjacent table
        id1(int), id2(int)      : the ids we want to connect together
    '''
    if id2 not in adj_table[id1]:
        adj_table[id1].append(id2)
    if id1 not in adj_table[id2]:
        adj_table[id2].append(id1)

def edge_connect(adj_table, subdivision, face1, face2, staggered=False):
    '''
    the function connect the point on the edse of two faces,
    and only link the right side of face1

    Args:
        adj_table(list[list])   : the adjacent table
        subdivision(int)        : the size of the icosahedron
        face1(int), face2(int)  : the faces we want to connect together
        staggered(bool)         : whether the faces are in the same orientation
    '''
    edge_length = 2**subdivision
    pixel_in_face = 4**subdivision
    if not staggered:
        face1_right_edge = [pixel_in_face*face1 + (i+1)**2 - 1 for i in range(edge_length)]
        face2_left_edge = [pixel_in_face*face2 + i**2 for i in range(edge_length)]
        for layer in range(edge_length):
            connect(adj_table, face1_right_edge[layer], face2_left_edge[layer])
    elif face1 < 10:
        face1_left_edge = [pixel_in_face*face1 + i**2 for i in range(edge_length)]
        face2_left_edge = [pixel_in_face*face2 + i**2 for i in range(edge_length)]
        for layer in range(edge_length):
            connect(adj_table, face1_left_edge[edge_length-layer-1], face2_left_edge[layer])
    else:
        face1_right_edge = [pixel_in_face*face1 + (i+1)**2 - 1 for i in range(edge_length)]
        face2_right_edge = [pixel_in_face*face2 + (i+1)**2 - 1 for i in range(edge_length)]
        for layer in range(edge_length):
            connect(adj_table, face1_right_edge[layer], face2_right_edge[edge_length-layer-1])

def make_inner_link(adj_table, subdivision):
    '''
    find the vertical and horizontal connections

    eg. subdivision = 2
             00             ... Layer 0
          01 02 03          ... Layer 1
       04 05 06 07 08       ... Layer 2
    09 10 11 12 13 14 15    ... Layer 3

    Vertical Connection     : 00--02, 01--05, 03--07...
    Horizontal Connection   : [01--02, 02--03], [04--05, 05--06]...
    '''
    for face in range(20):
        # number order is the same for each face,
        # so we only need to shift the points by 4**subdivision
        start_id = face*(4**subdivision)
        end_id = (face+1)*(4**subdivision)
        for layer in range(2**subdivision):
            row = range(layer**2, (layer+1)**2, 2)
            for point in row:
                # find the vertical connection
                if start_id + point + 2*(layer+1) < end_id:
                    connect(adj_table, start_id + point, start_id + point + 2*(layer+1))
                # find the horizontal connection
                if (point + 1) < (layer+1)**2:
                    connect(adj_table, start_id + point, start_id + point + 1)
                    connect(adj_table, start_id + point + 1, start_id + point + 2)
    return adj_table

def make_edge_link(adj_table, subdivision):
    '''
    find the horizontal connection, and other edge connections

    eg. the expanded view of icosahedron
    00    01    02    03    04      _____ Horizontal Connection 1
    05 10 06 11 07 12 08 13 09 14   _____
       15    16    17    18    19         Horizontal Connection 2

    Horizontal Connection   : 00--05, 01--06, 02--07, 10--15...
    Edges Connection        : 00--01, 01--02, 05--10, 15--16...
    '''
    edge_length = 2**subdivision
    pixel_in_face = 4**subdivision
    left_point = (2**subdivision-1)**2
    right_point = 4**subdivision - 1
    # left point of face_0 connect to right point of face_5
    # 5*4^s + (4^s - 1) - (4^s - 2*2^s + 1) = 5*4^s + 2*2^s - 2
    max_diff = 5 * 4**subdivision + 2 * 2**subdivision - 2

    for face in range(5):
        for i, pixel in enumerate(range(left_point, right_point + 2, 2)):
            # link between face_0 and face_5
            shifted_top = pixel_in_face*face + pixel
            connect(adj_table, shifted_top, shifted_top + max_diff - 4 * (i % edge_length))
            # link between face_10 and face_15
            shifted_bottom = pixel_in_face*10 + pixel_in_face*face + pixel
            connect(adj_table, shifted_bottom, shifted_bottom + max_diff - 4 * (i % edge_length))

    for i in range(5):
        # five triangles in top layer
        edge_connect(adj_table, subdivision, i, (i+1)%5)
        # ten staggered triangles in middle layer
        edge_connect(adj_table, subdivision, 5+i, 10+i, True)
        edge_connect(adj_table, subdivision, 10+i, 5+(i+1)%5, True)
        # five triangles in bottom layer
        edge_connect(adj_table, subdivision, 15+(i+1)%5, 15+i)

    return adj_table

def make_adjacency_table(subdivision):
    '''
    the main progress of building a adjacency table
    by divide the connections into two types:
    1) insdide the triangle
    2) on the edge of the triangle

    Args:
        subdividion(int)    : how many points on the icosahedron projection

    Return:
        adj_table(np.array) : with size(20*(4**subdivision), 3),
                              the ith row means the other 3 points are linked to the point i
    '''
    adj_table = [[i] for i in range(20*(4**subdivision))]
    adj_table = make_inner_link(adj_table, subdivision)
    adj_table = make_edge_link(adj_table, subdivision)
    adj_table = np.array(adj_table)
    return adj_table

def main():
    '''
    for testing only...
    '''
    table_1 = make_adjacency_table(1)
    table_2 = make_adjacency_table(2)
    table_7 = make_adjacency_table(7)

if __name__ == '__main__':
    main()
