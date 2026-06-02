= Geospatial Data Management

Spatial queries — "what is within 5 km of this point?", "do these polygons intersect?", "rank these candidates by distance" — break the standard B-Tree because the natural keys (latitude, longitude, polygon vertices) have no useful one-dimensional order. Three families of techniques dominate: *space-partitioning trees* (R-tree and friends), *space-filling curves* that linearize 2D coordinates (GeoHash, S2, H3), and *hybrid grid + cell IDs* that bridge the two.

*See also:* _Storage Engines_, _Joins and Aggregation_, _Multi-Model Databases_

== The Spatial Index Problem

A 2D point has no total order that preserves proximity in all directions. A B-Tree on $(x, y)$ via lexicographic ordering supports range queries on $x$ but degenerates to a full scan on $y$. Spatial indexes must approximate 2D locality.

*Operations to support:*

- *Point-in-polygon* / *region query*: $"WHERE ST_Within"("point", "polygon")$.
- *Range query*: all features within bounding box.
- *k-NN*: $k$ nearest features to a query point.
- *Spatial join*: pairs of features satisfying `ST_Intersects(R.geom, S.geom)`.

== R-Tree and Variants

The R-tree (Guttman, 1984) generalizes the B-tree by indexing *Minimum Bounding Rectangles* (MBRs). Each inner node holds child MBRs that cover the union of leaves below.

```
Root MBR (0,0)-(100,100)
├── Child MBR (0,0)-(50,50) → leaf cells (points / polygons inside)
├── Child MBR (40,30)-(80,70) → leaf cells          ← children may overlap!
└── Child MBR (60,60)-(100,100) → leaf cells
```

Overlap is the R-tree's biggest pathology: a point query may descend multiple children. Variants differ by how they split overflowing nodes:

#table(
  columns: (auto, auto),
  [*Variant*], [*Key Idea*],
  [R-tree (Guttman)], [Quadratic / linear split heuristics minimizing MBR area increase.],
  [R\*-tree (Beckmann 1990)], [Forced reinsertion, minimize overlap & margin, default in many systems.],
  [R+-tree (Sellis 1987)], [Non-overlapping splits; objects duplicated into multiple leaves.],
  [Hilbert R-tree], [Order entries by Hilbert curve before bulk-load.],
  [STR (Sort-Tile-Recursive) bulk load], [Sort by $x$, partition into $sqrt(n/B)$ vertical slabs, sort each by $y$.],
  [Priority R-tree], [Worst-case-optimal window queries.],
)

=== Search

```python
def rtree_search(node, query_box, hits):
    if not intersects(node.mbr, query_box): return
    if node.is_leaf:
        for entry in node.entries:
            if intersects(entry.mbr, query_box):
                hits.append(entry)
    else:
        for child in node.children:
            rtree_search(child, query_box, hits)
```

=== k-NN with Best-First Search

Hjaltason & Samet (1999) showed that *best-first* traversal using a priority queue of (node, mindist) pairs is incremental and optimal in I/O.

```python
import heapq
def knn(root, q, k):
    pq = [(mindist(root.mbr, q), root)]
    out = []
    while pq and len(out) < k:
        d, node = heapq.heappop(pq)
        if node.is_leaf_entry:
            out.append((d, node)); continue
        for child in node.children:
            heapq.heappush(pq, (mindist(child.mbr, q), child))
    return out
```

== PostGIS and GiST

PostgreSQL ships geospatial support through *PostGIS*. PostGIS registers operator classes for the *GiST* (Generalized Search Tree, Hellerstein 1995) framework — a templated balanced tree that can implement R-tree behavior given user-provided `consistent`, `union`, `compress`, `decompress`, `penalty`, `picksplit`, and `same` operators.

```sql
CREATE EXTENSION postgis;
CREATE TABLE places (id BIGSERIAL PRIMARY KEY,
                     name TEXT,
                     geom GEOMETRY(Point, 4326));
CREATE INDEX places_geom_gix ON places USING GIST (geom);

-- Range query
SELECT id, name
FROM   places
WHERE  geom && ST_MakeEnvelope(-122.5, 37.7, -122.4, 37.8, 4326);

-- k-NN: ORDER BY geom <-> point uses the index
SELECT id, name, geom <-> ST_SetSRID(ST_MakePoint(-122.45, 37.75), 4326) AS dist
FROM   places
ORDER BY geom <-> ST_SetSRID(ST_MakePoint(-122.45, 37.75), 4326)
LIMIT  10;
```

The `<->` distance operator on a GiST index performs best-first k-NN — without it, the planner sorts a full scan.

=== SP-GiST and BRIN

*SP-GiST* implements space-partitioning trees (quad-tree, k-d tree) — useful for non-overlapping decompositions of point data.

*BRIN* (Block Range Index) is invaluable for spatially-clustered data: per block range it stores the bounding box; queries skip blocks whose MBR misses. With CLUSTER on a Hilbert curve order, BRIN gives near-R-tree selectivity at $approx 0.01$% the size.

== Space-Filling Curves and GeoHash

A space-filling curve maps 2D coordinates to a single integer such that nearby integers correspond (mostly) to nearby points. The *Z-order curve* (Morton, 1966) interleaves bit positions; the *Hilbert curve* preserves locality more faithfully but is more expensive to compute.

```
Z-order encoding of (x=3, y=5) on 4-bit coordinates:
  x = 0011, y = 0101
  Interleave (y bit, x bit) starting MSB:
  → 0 0 0 1 1 0 1 1  = 0x1B = 27
```

*GeoHash* (Niemeyer, 2008) is base-32 Z-order with each character encoding 5 bits ($2.5$ bits per axis), giving cells halved alternately in latitude and longitude.

```
GeoHash "9q8yyk" encodes a ~150 m cell over San Francisco.
Prefix "9q8" is the parent ~20 km cell; prefix "9q" is the ~600 km region.
```

Range queries become *prefix searches* on the GeoHash string — but cells at the equator/prime-meridian boundary have hash discontinuities. Querying a circle requires expanding to the 8 neighbour cells.

== Google S2

S2 (Google, ~2005) projects the sphere onto an inscribed cube, applies a quadratic-to-spherical adjustment to make cell areas more uniform, and recursively subdivides each face with a Hilbert curve. A *CellID* is a 64-bit integer of `(face_id (3) | hilbert_path (up to 60) | trailing 1)`.

```python
import s2sphere
ll = s2sphere.LatLng.from_degrees(37.7749, -122.4194)
cell = s2sphere.CellId.from_lat_lng(ll).parent(15)  # ~1 km cell
print(cell.id(), cell.to_token())  # 64-bit ID, base16 token
# Neighbours, covering for a region, etc.
region = s2sphere.LatLngRect.from_point_pair(...)
coverer = s2sphere.RegionCoverer(); coverer.max_cells = 8
covering = coverer.get_covering(region)   # list of CellIds
```

*S2 strengths:* hierarchical (parent cells = prefix truncation), 30 levels giving $approx 0.7 "cm"^2$ resolution, neighbours computable arithmetically. Used in Google Maps, MongoDB 2dsphere, CockroachDB spatial.

*S2 covering* of a polygon returns a set of CellIDs whose union approximates the region — store features by CellID prefix, query by intersecting with the covering set.

== Uber H3

H3 (Uber, 2018) uses *hexagonal* cells over an icosahedron. Hexagons tile uniformly with 6 equidistant neighbours (squares have 8 neighbours at two different distances). 16 resolutions give cells from $approx 4250000$ km² (res 0) to $approx 0.9$ m² (res 15).

```python
import h3
h3.latlng_to_cell(37.7749, -122.4194, 9)   # → '8928308280fffff'
h3.grid_disk('8928308280fffff', k=2)        # neighbours within 2 hops
h3.cell_area('8928308280fffff', 'km^2')     # ~0.105
h3.cell_to_parent('8928308280fffff', 6)
```

*Caveats:* hexagons cannot perfectly tile a sphere — H3 has 12 *pentagons* at icosahedron vertices, which require special-case handling. Parent/child containment is *approximate* (a child may straddle a parent boundary), unlike S2.

#table(
  columns: (auto, auto, auto, auto),
  [*System*], [*Cell Shape*], [*Hierarchy*], [*Notable Use*],
  [GeoHash], [Rectangle], [String prefix], [Twitter geo, Elasticsearch],
  [S2], [Quadrilateral on cube face], [Exact prefix], [Maps, MongoDB 2dsphere],
  [H3], [Hexagon], [Approximate], [Uber ETA, Foursquare, ride-sharing],
  [Quadkey (Bing)], [Rectangle], [Exact prefix], [Bing Maps tiles],
)

== Spatial Joins

Naive nested-loop spatial join is $O(|R| dot |S|)$. Practical strategies:

=== Index-Nested-Loop

For each $r in R$, probe $S$'s R-tree with `r.mbr`. Efficient if $|R|$ is small or $S$ has a tight index.

=== Sweep-Line / PBSM

*PBSM* (Patel & DeWitt, 1996) — *Partition Based Spatial Merge*: tile space into a grid; assign each feature to all cells it overlaps; per cell, nested-loop only the locals.

=== Synchronous R-Tree Traversal

If both inputs have R-trees, traverse them in parallel; at each level recurse into pairs of children whose MBRs intersect (Brinkhoff et al., 1993).

=== Distributed Spatial Join

SpatialSpark, GeoSpark/Sedona, BigSpatialData all partition by space-filling curve, broadcast small side, or apply distributed PBSM.

```sql
-- PostGIS spatial join using GiST
SELECT a.id, b.id
FROM   parcels a JOIN buildings b
       ON ST_Intersects(a.geom, b.geom);
-- Plan: Nested Loop with Index Cond using buildings_geom_gix
```

== Trajectory and Range-K Queries

Moving-object databases (e.g. SECONDO, MobilityDB extension on PostGIS) treat trajectories as functions $f: T -> "point"$. Indexes include the *3D R-tree* (treating time as a third dimension) and the *TB-tree* (Trajectory-Bundle tree) that keeps complete trajectories in single leaves.

Range-$k$ ("$k$-NN over time") and *spatiotemporal joins* compose the above with temporal predicates.

== Operational Notes

- Always store geometries in a *projected* coordinate system matching the workload's scale (UTM for city-scale, Web Mercator for tile-aligned data). Latitude/longitude (SRID 4326) is *not* a metric space — distances are great-circle.
- For point-only data at scale, S2 or H3 cell IDs in a B-tree often outperform a GiST R-tree because they pack densely and partition naturally.
- *Geometry validity* matters: invalid polygons (self-intersections, wrong winding) silently break $"ST_Within"$. Use `ST_MakeValid` on ingest.
- For static reference data, *bulk-load* the R-tree via STR — incremental insertion of millions of polygons can produce a degenerate index with 5× the query time.

== Further Reading

Guttman, A. (1984). "R-Trees: A Dynamic Index Structure for Spatial Searching." SIGMOD.

Beckmann, N. et al. (1990). "The R\*-tree: An Efficient and Robust Access Method for Points and Rectangles." SIGMOD.

Leutenegger, S. et al. (1997). "STR: A Simple and Efficient Algorithm for R-Tree Packing." ICDE.

Hjaltason, G., Samet, H. (1999). "Distance Browsing in Spatial Databases." TODS.

Hellerstein, J., Naughton, J., Pfeffer, A. (1995). "Generalized Search Trees for Database Systems." VLDB (GiST).

Brinkhoff, T., Kriegel, H.-P., Seeger, B. (1993). "Efficient Processing of Spatial Joins Using R-Trees." SIGMOD.

Patel, J., DeWitt, D. (1996). "Partition Based Spatial-Merge Join." SIGMOD.

Morton, G. (1966). "A Computer Oriented Geodetic Data Base and a New Technique in File Sequencing." IBM Tech Report.

Google (2017). "S2 Geometry Library." s2geometry.io documentation.

Brodsky, I., Sahr, K. et al. (2018). "H3: Uber's Hexagonal Hierarchical Spatial Index." Uber Engineering Blog & h3geo.org.

Yu, J., Wu, J., Sarwat, M. (2015). "GeoSpark: A Cluster Computing Framework for Processing Large-Scale Spatial Data." SIGSPATIAL.

Eldawy, A., Mokbel, M. (2015). "SpatialHadoop: A MapReduce Framework for Spatial Data." ICDE.
