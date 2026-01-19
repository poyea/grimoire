= Computational Geometry

*Algorithms for geometric problems:* Computational geometry handles points, lines, polygons, and spatial queries efficiently. Essential for graphics, robotics, GIS, and CAD systems [de Berg et al. 2008].

*See also:* Segment Trees (for range queries), Graphs (for geometric graph algorithms), Binary Search (for geometric search)

*Numerical precision:* Floating-point errors cause geometric predicates to fail. Use epsilon comparisons or exact arithmetic when necessary.

```cpp
const double EPS = 1e-9;

bool equal(double a, double b) { return abs(a - b) < EPS; }
int sign(double x) { return (x > EPS) - (x < -EPS); }

struct Point {
    double x, y;

    Point(double x = 0, double y = 0) : x(x), y(y) {}

    Point operator+(const Point& p) const { return {x + p.x, y + p.y}; }
    Point operator-(const Point& p) const { return {x - p.x, y - p.y}; }
    Point operator*(double t) const { return {x * t, y * t}; }
    Point operator/(double t) const { return {x / t, y / t}; }

    double dot(const Point& p) const { return x * p.x + y * p.y; }
    double cross(const Point& p) const { return x * p.y - y * p.x; }
    double norm() const { return sqrt(x * x + y * y); }
    double norm2() const { return x * x + y * y; }

    Point rotate(double theta) const {
        return {x * cos(theta) - y * sin(theta),
                x * sin(theta) + y * cos(theta)};
    }

    bool operator<(const Point& p) const {
        if (!equal(x, p.x)) return x < p.x;
        return y < p.y;
    }

    bool operator==(const Point& p) const {
        return equal(x, p.x) && equal(y, p.y);
    }
};

using Vector = Point;
```

== Orientation and Cross Product

*Cross product sign:* Determines relative orientation of three points.

```cpp
// Returns: > 0 if counter-clockwise, < 0 if clockwise, 0 if collinear
double orientation(const Point& a, const Point& b, const Point& c) {
    return (b - a).cross(c - a);
}

// CCW test (counter-clockwise)
bool ccw(const Point& a, const Point& b, const Point& c) {
    return orientation(a, b, c) > EPS;
}

// Collinear test
bool collinear(const Point& a, const Point& b, const Point& c) {
    return abs(orientation(a, b, c)) < EPS;
}
```

*Geometric interpretation:*
- Cross product = signed area of parallelogram
- $|a times b|$ = $|a| dot |b| dot sin(theta)$

== Convex Hull

*Problem:* Find smallest convex polygon containing all points.

=== Graham Scan: $O(n log n)$

```cpp
vector<Point> convexHullGraham(vector<Point> points) {
    int n = points.size();
    if (n < 3) return points;

    // Find lowest point (tie-break by x)
    int minIdx = 0;
    for (int i = 1; i < n; i++) {
        if (points[i].y < points[minIdx].y ||
            (equal(points[i].y, points[minIdx].y) && points[i].x < points[minIdx].x)) {
            minIdx = i;
        }
    }
    swap(points[0], points[minIdx]);
    Point pivot = points[0];

    // Sort by polar angle relative to pivot
    sort(points.begin() + 1, points.end(), [&pivot](const Point& a, const Point& b) {
        double cross = (a - pivot).cross(b - pivot);
        if (abs(cross) > EPS) return cross > 0;
        // Collinear: closer point first
        return (a - pivot).norm2() < (b - pivot).norm2();
    });

    // Build hull
    vector<Point> hull;
    for (const Point& p : points) {
        while (hull.size() > 1 &&
               (hull.back() - hull[hull.size() - 2]).cross(p - hull.back()) <= EPS) {
            hull.pop_back();
        }
        hull.push_back(p);
    }

    return hull;
}
```

=== Andrew's Monotone Chain: $O(n log n)$

```cpp
vector<Point> convexHullAndrew(vector<Point> points) {
    int n = points.size();
    if (n < 3) return points;

    sort(points.begin(), points.end());

    vector<Point> hull;

    // Build lower hull
    for (int i = 0; i < n; i++) {
        while (hull.size() > 1 &&
               (hull.back() - hull[hull.size() - 2]).cross(points[i] - hull.back()) <= EPS) {
            hull.pop_back();
        }
        hull.push_back(points[i]);
    }

    // Build upper hull
    int lowerSize = hull.size();
    for (int i = n - 2; i >= 0; i--) {
        while (hull.size() > lowerSize &&
               (hull.back() - hull[hull.size() - 2]).cross(points[i] - hull.back()) <= EPS) {
            hull.pop_back();
        }
        hull.push_back(points[i]);
    }

    hull.pop_back();  // Remove duplicate last point
    return hull;
}
```

*Complexity:* $O(n log n)$ dominated by sorting

*Use cases:*
- Collision detection (convex shapes)
- Smallest enclosing shapes
- Computational biology (protein structures)

== Line Segment Intersection

*Problem:* Determine if two line segments intersect.

```cpp
// Check if point c lies on segment ab
bool onSegment(const Point& a, const Point& b, const Point& c) {
    return collinear(a, b, c) &&
           min(a.x, b.x) - EPS <= c.x && c.x <= max(a.x, b.x) + EPS &&
           min(a.y, b.y) - EPS <= c.y && c.y <= max(a.y, b.y) + EPS;
}

// Check if segments ab and cd intersect
bool segmentsIntersect(const Point& a, const Point& b,
                       const Point& c, const Point& d) {
    double d1 = orientation(c, d, a);
    double d2 = orientation(c, d, b);
    double d3 = orientation(a, b, c);
    double d4 = orientation(a, b, d);

    if (sign(d1) * sign(d2) < 0 && sign(d3) * sign(d4) < 0) {
        return true;  // Proper intersection
    }

    // Check collinear cases
    if (onSegment(c, d, a)) return true;
    if (onSegment(c, d, b)) return true;
    if (onSegment(a, b, c)) return true;
    if (onSegment(a, b, d)) return true;

    return false;
}

// Find intersection point (assumes segments intersect)
Point lineIntersection(const Point& a, const Point& b,
                       const Point& c, const Point& d) {
    double a1 = b.y - a.y, b1 = a.x - b.x, c1 = a1 * a.x + b1 * a.y;
    double a2 = d.y - c.y, b2 = c.x - d.x, c2 = a2 * c.x + b2 * c.y;
    double det = a1 * b2 - a2 * b1;

    return {(b2 * c1 - b1 * c2) / det, (a1 * c2 - a2 * c1) / det};
}
```

== Sweep Line Algorithm (Bentley-Ottmann)

*Problem:* Find all intersections among n line segments.

*Idea:* Sweep vertical line left-to-right, maintain active segments in BST.

```cpp
struct Segment {
    Point p1, p2;
    int id;

    double yAt(double x) const {
        if (equal(p1.x, p2.x)) return p1.y;
        return p1.y + (p2.y - p1.y) * (x - p1.x) / (p2.x - p1.x);
    }
};

struct Event {
    double x;
    int type;  // 0 = left endpoint, 1 = right endpoint, 2 = intersection
    int seg1, seg2;

    bool operator<(const Event& e) const {
        if (!equal(x, e.x)) return x < e.x;
        return type < e.type;
    }
};

class SweepLine {
    double sweepX;

public:
    vector<pair<int, int>> findIntersections(vector<Segment>& segments) {
        vector<pair<int, int>> intersections;
        priority_queue<Event, vector<Event>, greater<Event>> events;

        // Add endpoint events
        for (int i = 0; i < segments.size(); i++) {
            if (segments[i].p1.x > segments[i].p2.x) {
                swap(segments[i].p1, segments[i].p2);
            }
            events.push({segments[i].p1.x, 0, i, -1});
            events.push({segments[i].p2.x, 1, i, -1});
        }

        set<pair<double, int>> active;  // (y-coordinate, segment id)

        auto getY = [&](int id) {
            return segments[id].yAt(sweepX);
        };

        auto checkIntersection = [&](int id1, int id2) {
            if (id1 < 0 || id2 < 0) return;
            if (segmentsIntersect(segments[id1].p1, segments[id1].p2,
                                  segments[id2].p1, segments[id2].p2)) {
                Point inter = lineIntersection(
                    segments[id1].p1, segments[id1].p2,
                    segments[id2].p1, segments[id2].p2);
                if (inter.x > sweepX + EPS) {
                    events.push({inter.x, 2, id1, id2});
                }
            }
        };

        while (!events.empty()) {
            Event e = events.top();
            events.pop();
            sweepX = e.x;

            if (e.type == 0) {  // Left endpoint
                double y = getY(e.seg1);
                auto it = active.insert({y, e.seg1}).first;

                auto above = next(it);
                auto below = (it != active.begin()) ? prev(it) : active.end();

                if (above != active.end()) checkIntersection(e.seg1, above->second);
                if (below != active.end()) checkIntersection(e.seg1, below->second);

            } else if (e.type == 1) {  // Right endpoint
                double y = getY(e.seg1);
                auto it = active.find({y, e.seg1});

                if (it != active.end()) {
                    auto above = next(it);
                    auto below = (it != active.begin()) ? prev(it) : active.end();

                    if (above != active.end() && below != active.end()) {
                        checkIntersection(above->second, below->second);
                    }
                    active.erase(it);
                }

            } else {  // Intersection
                intersections.push_back({e.seg1, e.seg2});
                // Swap order in active set (simplified - full implementation more complex)
            }
        }

        return intersections;
    }
};
```

*Complexity:* $O((n + k) log n)$ where k = number of intersections

== Point in Polygon

*Problem:* Test if point is inside polygon.

=== Ray Casting (Any Polygon)

```cpp
bool pointInPolygon(const Point& p, const vector<Point>& polygon) {
    int n = polygon.size();
    int crossings = 0;

    for (int i = 0; i < n; i++) {
        const Point& a = polygon[i];
        const Point& b = polygon[(i + 1) % n];

        // Check if horizontal ray from p crosses edge ab
        if ((a.y <= p.y && p.y < b.y) || (b.y <= p.y && p.y < a.y)) {
            double x = a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y);
            if (p.x < x) {
                crossings++;
            }
        }
    }

    return crossings % 2 == 1;  // Odd = inside
}
```

=== Winding Number (Handles Complex Polygons)

```cpp
int windingNumber(const Point& p, const vector<Point>& polygon) {
    int n = polygon.size();
    int winding = 0;

    for (int i = 0; i < n; i++) {
        const Point& a = polygon[i];
        const Point& b = polygon[(i + 1) % n];

        if (a.y <= p.y) {
            if (b.y > p.y && orientation(a, b, p) > 0) {
                winding++;  // Upward crossing
            }
        } else {
            if (b.y <= p.y && orientation(a, b, p) < 0) {
                winding--;  // Downward crossing
            }
        }
    }

    return winding;  // Non-zero = inside
}
```

*Complexity:* $O(n)$ for n-vertex polygon

== Polygon Area

```cpp
double polygonArea(const vector<Point>& polygon) {
    double area = 0;
    int n = polygon.size();

    for (int i = 0; i < n; i++) {
        area += polygon[i].cross(polygon[(i + 1) % n]);
    }

    return abs(area) / 2;
}

// Signed area (positive if CCW, negative if CW)
double signedArea(const vector<Point>& polygon) {
    double area = 0;
    int n = polygon.size();

    for (int i = 0; i < n; i++) {
        area += polygon[i].cross(polygon[(i + 1) % n]);
    }

    return area / 2;
}
```

*Shoelace formula:* $A = frac(1,2) |sum_(i=0)^(n-1) (x_i y_(i+1) - x_(i+1) y_i)|$

== Closest Pair of Points

*Problem:* Find minimum distance between any two points.

```cpp
double closestPairRec(vector<Point>& px, vector<Point>& py,
                      int left, int right) {
    if (right - left <= 3) {
        // Brute force for small sets
        double minDist = DBL_MAX;
        for (int i = left; i < right; i++) {
            for (int j = i + 1; j < right; j++) {
                minDist = min(minDist, (px[i] - px[j]).norm());
            }
        }
        return minDist;
    }

    int mid = (left + right) / 2;
    double midX = px[mid].x;

    // Split py into left and right halves
    vector<Point> pyLeft, pyRight;
    for (const Point& p : py) {
        if (p.x <= midX) pyLeft.push_back(p);
        else pyRight.push_back(p);
    }

    double dl = closestPairRec(px, pyLeft, left, mid);
    double dr = closestPairRec(px, pyRight, mid, right);
    double d = min(dl, dr);

    // Check strip
    vector<Point> strip;
    for (const Point& p : py) {
        if (abs(p.x - midX) < d) {
            strip.push_back(p);
        }
    }

    // Check pairs in strip
    for (int i = 0; i < strip.size(); i++) {
        for (int j = i + 1; j < strip.size() && strip[j].y - strip[i].y < d; j++) {
            d = min(d, (strip[i] - strip[j]).norm());
        }
    }

    return d;
}

double closestPair(vector<Point> points) {
    vector<Point> px = points, py = points;
    sort(px.begin(), px.end());
    sort(py.begin(), py.end(), [](const Point& a, const Point& b) {
        return a.y < b.y;
    });

    return closestPairRec(px, py, 0, points.size());
}
```

*Complexity:* $O(n log n)$

*Key insight:* Strip contains at most 7 points within distance d of any point (geometric packing argument).

== Rotating Calipers

*Problem:* Find diameter (farthest pair) of convex polygon.

```cpp
double convexDiameter(const vector<Point>& hull) {
    int n = hull.size();
    if (n < 2) return 0;
    if (n == 2) return (hull[0] - hull[1]).norm();

    int j = 1;
    double maxDist = 0;

    for (int i = 0; i < n; i++) {
        // Find antipodal point for edge (i, i+1)
        Vector edge = hull[(i + 1) % n] - hull[i];

        while ((edge.cross(hull[(j + 1) % n] - hull[i])) >
               (edge.cross(hull[j] - hull[i]))) {
            j = (j + 1) % n;
        }

        maxDist = max(maxDist, (hull[i] - hull[j]).norm());
        maxDist = max(maxDist, (hull[(i + 1) % n] - hull[j]).norm());
    }

    return maxDist;
}
```

*Applications:*
- Minimum enclosing rectangle
- Minimum width of convex polygon
- Bridge between two convex hulls

*Complexity:* $O(n)$ for convex polygon with n vertices

== Voronoi Diagram and Delaunay Triangulation

*Voronoi diagram:* Partition plane into regions closest to each site.

*Delaunay triangulation:* Dual of Voronoi, maximizes minimum angle.

```cpp
// Fortune's algorithm for Voronoi (simplified interface)
struct VoronoiDiagram {
    vector<Point> sites;
    vector<vector<Point>> cells;  // Vertices of each cell

    // Full implementation requires beach line and sweep line data structures
    void compute(const vector<Point>& points);
};

// Bowyer-Watson algorithm for Delaunay triangulation
struct Triangle {
    int v[3];  // Vertex indices

    bool circumcircleContains(const Point& p, const vector<Point>& points) const {
        Point a = points[v[0]], b = points[v[1]], c = points[v[2]];

        double ax = a.x - p.x, ay = a.y - p.y;
        double bx = b.x - p.x, by = b.y - p.y;
        double cx = c.x - p.x, cy = c.y - p.y;

        double det = (ax * ax + ay * ay) * (bx * cy - cx * by) -
                     (bx * bx + by * by) * (ax * cy - cx * ay) +
                     (cx * cx + cy * cy) * (ax * by - bx * ay);

        return det > EPS;  // Assumes CCW orientation
    }
};

class DelaunayTriangulation {
    vector<Point> points;
    vector<Triangle> triangles;

public:
    void addPoint(const Point& p) {
        int idx = points.size();
        points.push_back(p);

        // Find triangles whose circumcircle contains p
        vector<Triangle> badTriangles;
        vector<Triangle> goodTriangles;

        for (const Triangle& t : triangles) {
            if (t.circumcircleContains(p, points)) {
                badTriangles.push_back(t);
            } else {
                goodTriangles.push_back(t);
            }
        }

        // Find boundary of polygon hole
        set<pair<int, int>> boundary;
        for (const Triangle& t : badTriangles) {
            for (int i = 0; i < 3; i++) {
                int a = t.v[i], b = t.v[(i + 1) % 3];
                if (a > b) swap(a, b);

                auto edge = make_pair(a, b);
                if (boundary.count(edge)) {
                    boundary.erase(edge);
                } else {
                    boundary.insert(edge);
                }
            }
        }

        // Create new triangles
        triangles = goodTriangles;
        for (auto& [a, b] : boundary) {
            triangles.push_back({{a, b, idx}});
        }
    }

    const vector<Triangle>& getTriangles() const { return triangles; }
};
```

*Complexity:*
- Voronoi (Fortune's): $O(n log n)$
- Delaunay (Bowyer-Watson): $O(n^2)$ worst case, $O(n log n)$ expected

== Half-Plane Intersection

*Problem:* Find intersection of n half-planes.

```cpp
struct HalfPlane {
    Point p, d;  // Point on boundary and direction (left side is inside)

    // Line equation: d.y * x - d.x * y + c = 0 where c = d.x * p.y - d.y * p.x
    double eval(const Point& q) const {
        return d.cross(q - p);  // Positive = inside
    }

    Point intersect(const HalfPlane& other) const {
        double t = other.d.cross(other.p - p) / other.d.cross(d);
        return p + d * t;
    }

    bool operator<(const HalfPlane& other) const {
        // Sort by angle
        double a1 = atan2(d.y, d.x);
        double a2 = atan2(other.d.y, other.d.x);
        return a1 < a2;
    }
};

vector<Point> halfPlaneIntersection(vector<HalfPlane> planes) {
    sort(planes.begin(), planes.end());

    deque<HalfPlane> dq;
    deque<Point> pts;

    for (const HalfPlane& h : planes) {
        while (!pts.empty() && h.eval(pts.back()) < -EPS) {
            dq.pop_back();
            pts.pop_back();
        }
        while (!pts.empty() && h.eval(pts.front()) < -EPS) {
            dq.pop_front();
            pts.pop_front();
        }

        if (!dq.empty()) {
            pts.push_back(h.intersect(dq.back()));
        }
        dq.push_back(h);
    }

    // Close the polygon
    while (pts.size() > 1 && dq.front().eval(pts.back()) < -EPS) {
        dq.pop_back();
        pts.pop_back();
    }
    while (pts.size() > 1 && dq.back().eval(pts.front()) < -EPS) {
        dq.pop_front();
        pts.pop_front();
    }

    if (dq.size() > 1) {
        pts.push_back(dq.front().intersect(dq.back()));
    }

    return vector<Point>(pts.begin(), pts.end());
}
```

*Complexity:* $O(n log n)$

== Performance Summary

#table(
  columns: 4,
  align: (left, center, center, left),
  table.header([Algorithm], [Time], [Space], [Problem]),
  [Convex Hull], [$O(n log n)$], [$O(n)$], [Smallest enclosing polygon],
  [Line Intersection], [$O(1)$], [$O(1)$], [Two segments],
  [Sweep Line], [$O((n+k) log n)$], [$O(n)$], [All intersections],
  [Point in Polygon], [$O(n)$], [$O(1)$], [Containment test],
  [Closest Pair], [$O(n log n)$], [$O(n)$], [Minimum distance],
  [Rotating Calipers], [$O(n)$], [$O(1)$], [Diameter, width],
  [Delaunay], [$O(n log n)$], [$O(n)$], [Triangulation],
)

== References

*Primary Sources:*

*de Berg, M., Cheong, O., van Kreveld, M., & Overmars, M. (2008)*. Computational Geometry: Algorithms and Applications (3rd ed.). Springer-Verlag. ISBN 978-3-540-77973-5.

*Preparata, F.P. & Shamos, M.I. (1985)*. Computational Geometry: An Introduction. Springer-Verlag.

*Algorithms & Theory:*

*Graham, R.L. (1972)*. An Efficient Algorithm for Determining the Convex Hull of a Finite Planar Set. Information Processing Letters 1(4): 132-133.

*Andrew, A.M. (1979)*. Another Efficient Algorithm for Convex Hulls in Two Dimensions. Information Processing Letters 9(5): 216-219.

*Bentley, J.L. & Ottmann, T.A. (1979)*. Algorithms for Reporting and Counting Geometric Intersections. IEEE Transactions on Computers C-28(9): 643-647.

*Fortune, S. (1987)*. A Sweepline Algorithm for Voronoi Diagrams. Algorithmica 2(1): 153-174.

*Shamos, M.I. & Hoey, D. (1975)*. Closest-Point Problems. FOCS 1975, pp. 151-162.

*Toussaint, G.T. (1983)*. Solving Geometric Problems with the Rotating Calipers. IEEE MELECON 1983.
