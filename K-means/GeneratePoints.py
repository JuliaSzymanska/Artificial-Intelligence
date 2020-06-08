from shapely.geometry import Polygon, Point
import random


def generator(poly, number_of_random_points):
    minx, miny, maxx, maxy = poly.bounds
    list_of_points = []
    for i in range(number_of_random_points):
        while True:
            p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
            if poly.contains(p):
                list_of_points.append(p)
                break
    return list_of_points


def find_points():
    number_of_points = int(input("How many vertices will the figure have? "))
    points = []
    for i in range(0, number_of_points):
        x, y = input("Give x and y vertex: ").split()
        points.append([])
        points[i].append(float(x))
        points[i].append(float(y))
    p = Polygon(points)
    number_of_random_points = input("How many random points to generate?")
    random_points = generator(p, int(number_of_random_points))
    individual_points = [(pt.x, pt.y) for pt in random_points]
    with open("Data/RandomPoints.txt", 'w') as f:
        f.writelines(','.join(str(j) for j in i) + '\n' for i in individual_points)
