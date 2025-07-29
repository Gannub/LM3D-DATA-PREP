def get_connected_vertices(obj_path, vertex_index, steps=3):
    faces = []

    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('f '):
                parts = line.strip().split()[1:]
                face = [int(p.split('/')[0]) - 1 for p in parts]
                faces.append(face)

    visited = set([vertex_index])
    current_level = set([vertex_index])

    for _ in range(steps):
        next_level = set()
        for vertex in current_level:
            for face in faces:
                if vertex in face:
                    next_level.update(face)
        next_level.difference_update(visited)
        visited.update(next_level)
        current_level = next_level

    return visited


obj_path = 'fitScanResult_2023.obj'
v1 = 3929
v2 = 3930
vert1 = get_connected_vertices(obj_path, v1)
vert2 = get_connected_vertices(obj_path, v2)

# combine the sets
combined_vertices = vert1.union(vert2)

print(f"Combined connected vertices: {combined_vertices}")
