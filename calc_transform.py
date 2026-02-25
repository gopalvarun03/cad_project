
class GP_Dir:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        # Normalize? gp_Dir is normalized.
        l = (x*x + y*y + z*z)**0.5
        if l != 0:
            self.x /= l
            self.y /= l
            self.z /= l
    def __repr__(self):
        return f"gp_Dir({self.x:.6g}, {self.y:.6g}, {self.z:.6g})"

def transform(vec, m):
    # m is 3x3 matrix
    # vec is (x,y,z)
    x, y, z = vec
    nx = m[0][0]*x + m[0][1]*y + m[0][2]*z
    ny = m[1][0]*x + m[1][1]*y + m[1][2]*z
    nz = m[2][0]*x + m[2][1]*y + m[2][2]*z
    return (nx, ny, nz)

# M derived from observation
# col1(X -> Y): 0, 1, 0
# col2(Y -> -Z): 0, 0, -1
# col3(Z -> -X): -1, 0, 0
M = [
    [0, 0, -1],
    [1, 0, 0],
    [0, -1, 0]
]

# Compute M^2
M2 = [[0,0,0], [0,0,0], [0,0,0]]
for i in range(3):
    for j in range(3):
        val = 0
        for k in range(3):
            val += M[i][k] * M[k][j]
        M2[i][j] = val

print("M2:")
for row in M2:
    print(row)

# Target views to transform
# "front": gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0), gp_Dir(0, 1, 0)),
# "right": gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 1, 0), gp_Dir(-1, 0, 0)),
# "top":   gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1), gp_Dir(0, 1, 0)),
# "iso":   gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(1, 1, 1), gp_Dir(-1, 1, 0))

targets = {
    "front": {"n": (1,0,0), "vx": (0,1,0)},
    "right": {"n": (0,1,0), "vx": (-1,0,0)},
    "top":   {"n": (0,0,1), "vx": (0,1,0)},
    "iso":   {"n": (1,1,1), "vx": (-1,1,0)}
}

print("\nTransformed Views (2 transitions):")
for name, vecs in targets.items():
    new_n = transform(vecs["n"], M2)
    new_vx = transform(vecs["vx"], M2)
    # format nicely for user
    # Check for integers to print cleanly
    def fmt(v):
        return [int(x) if x == int(x) else x for x in v]
    
    print(f'"{name}": gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir{tuple(fmt(new_n))}, gp_Dir{tuple(fmt(new_vx))})')
