STUDENT_ID = "40403384"

BASE_SEED = int(STUDENT_ID[-2])
MAZE_SIZE = 15 + (BASE_SEED % 4)

print("Student ID:", STUDENT_ID)
print("Base seed:", BASE_SEED)
print("Maze size:", MAZE_SIZE)
