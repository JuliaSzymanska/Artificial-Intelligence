from shapely.geometry import Polygon, Point
import random


def generator(poly, numberOfRandomPoints):
    minx, miny, maxx, maxy = poly.bounds
    listOfPoints = []
    for i in range(numberOfRandomPoints):
        while True:
            p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
            if poly.contains(p):
                listOfPoints.append(p)
                break
    return listOfPoints


def findPoints():
    numberOfPoints = int(input("Ile punktow bedzie posiadac figura? "))
    points = []
    for i in range(0, numberOfPoints):
        x, y = input("Podaj x i y punktu: ").split()
        points.append([])
        points[i].append(float(x))
        points[i].append(float(y))
    p = Polygon(points)
    numberOfRandomPoints = input("Ile losowych punktow wygenerowac?")
    randomPoints = generator(p, int(numberOfRandomPoints))
    individual_points = [(pt.x, pt.y) for pt in randomPoints]
    with open("RandomPoints.txt", 'w') as f:
        f.writelines(','.join(str(j) for j in i) + '\n' for i in individual_points)
