use stripack_sys::ffi::{
    addnod, bnodes, circum, crlist, delnod, edge, getnp, inside, nbcnt, nearnd, scoord, trans,
    trfind, trlist, trmesh,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TriangulationError {
    #[error("not enough nodes")]
    NotEnoughNodes,
    #[error("the first three nodes are collinear")]
    CollinearNodes,
    #[error("all coordinate inputs must have the same length")]
    IncorrectNodeCount,
    #[error("all coordinates must form unit vectors")]
    NotUnitVectors,
}

#[derive(Debug, Error)]
pub enum AddNodeError {
    #[error("invalid k value")]
    InvalidIndex,
    #[error("all nodes are collinear")]
    CollinearNodes,
}

#[derive(Debug, Error)]
pub enum CircumcenterError {
    #[error("all coordinates are collinear")]
    Collinear,
}

#[derive(Debug, Error)]
pub enum DeleteNodeError {
    #[error("invalid node index")]
    InvalidNodeIndex,
    #[error("not enough space")]
    NotEnoughSpace,
    #[error("invalid or corrupt triangulation")]
    InvalidTriangulation,
    #[error(
        "node_index indexes an interior node with four or more neighbors, none of which can be swapped out due to collinearity"
    )]
    Collinear,
    #[error("optimization error")]
    OptimizationError,
}

#[derive(Debug, Error)]
pub enum ForceAdjacentError {
    #[error("given nodes are identical or invalid")]
    InvalidNodePair,
    #[error("cannot make nodes adjacent")]
    CannotMakeAdjacent,
    #[error("error in optimization")]
    OptimError,
}

#[derive(Debug, Error)]
pub enum InsideError {
    #[error("the input indexes are invalid")]
    InvalidIndexList,
    #[error("the indicated region has a self-intersection")]
    SelfIntersection,
    #[error("could not determine whether the point is interior to the indicated region")]
    ConsistencyFailure,
}

#[derive(Debug, Error)]
pub enum NearestNodeError {
    #[error("invalid or corrupt triangulation")]
    InvalidTriangulation,
}

#[derive(Debug, Error)]
pub enum TriangleMeshError {
    #[error("invalid or corrupt triangulation")]
    InvalidTriangulation,
}

#[derive(Debug, Error)]
pub enum VoronoiCellError {
    #[error("not enough nodes in the triangulation to produce Voronoi cells")]
    NotEnoughNodes,
    #[error("a triangle is degenerate (has vertices lying on a common geodesic)")]
    DegenerateTriangle,
}

/// The boundary nodes for a triangulation.
#[derive(Debug, Clone)]
pub struct BoundaryInfo {
    /// Ordered sequence of boundary node indexes (counterclockwise)
    /// Empty if nodes don't lie in a single hemisphere
    pub nodes: Vec<usize>,
    /// The number of arcs in the triangulation
    pub num_arcs: usize,
    /// The number of triangles in the triangulation
    pub num_triangles: usize,
}

#[derive(Debug, Clone)]
pub enum LocationInfo {
    InsideTriangle {
        /**
        The unnormalized barycentric coordinates of the central projection of `p` onto the
        underlying planar triangle if `p` is in the convex hull of the nodes.
        */
        barycentric_coords: [f64; 3],
        /**
        The counterclockwise-ordered vertex indices of a triangle containing `p` if `p` is contained
        in a triangle.
        */
        bounding_triangle_indices: [usize; 3],
    },
    ///The rightmost and leftmost (boundary) nodes that are visible from `p`
    OutsideConvexHull {
        leftmost_visible_index: usize,
        rightmost_visible_index: usize,
    },
    ///All nodes are coplanar
    Coplanar,
}

#[derive(Debug, Clone)]
pub struct NodeDeletionInfo {
    /// The indexes of the endpoints of the new arcs added.
    pub new_arc_endpoints: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct NearestNode {
    /// The index of the nearest node to `p`.
    pub index: usize,
    /// The arc length between `p` and the node given by `index`. Because both points are on the
    /// unit sphere, this is also the angular separation in radians.
    pub arc_length: f64,
}

#[derive(Debug, Copy, Clone)]
pub struct SphericalCoordinates {
    /// The latitude in the range `[-PI/2, PI/2]`, or `0` if `norm = 0`.
    pub latitude: f64,
    /// The longitude in the range `[-PI, PI]`, or `0` if the point lies on the Z-axis.
    pub longitude: f64,
    /// The magnitude (Euclidean norm) of `p`
    pub norm: f64,
}

pub struct MeshData {
    /// The positions of points on the unit sphere.
    pub positions: Vec<[f64; 3]>,
    /// The indices of the vertices
    pub indices: Vec<usize>,
    /// The indices of the edges/arcs between vertices.
    pub arc_indices: Vec<[usize; 3]>,
    /// The neighbors, if any, of each of the vertices.
    pub neighbors: Vec<[Option<usize>; 3]>,
}

pub struct VoronoiCell {
    /// The position of the circumcircle of the triangle.
    pub position: [f64; 3],
    /// The circumradii (the arc length or angle between the circumcenter and associated
    /// triangle vertex).
    pub radius: f64,
}

/**
A Delaunay triangulation on the unit sphere.

The Delaunay triangulation is defined as a set of (spherical) triangles with the following five properties:
* The triangle vertices are nodes.
* No triangle contains a node other than its vertices.
* The interiors of the triangles are pairwise disjoint.
* The union of triangles is the convex hull of the set of nodes (the smallest convex set that contains the nodes). If the nodes are not contained in a single hemisphere, their convex hull is the entire sphere and there are no boundary nodes. Otherwise, there is at least three boundary nodes.
* The interior of the circumcircle of each triangle contains no node.

The first four properties define a triangulation, and the last property results in a triangulation which is as close as possible to equiangular in a certain sense and which is uniquely defined unless four or more nodes lie in a common plane. This property makes the triangulation well-suited for solving closest-point problems and for triangle based interpolation.

Provided the nodes are randomly ordered, the algorithm has expected time complexity O(N*log(N)) for most nodal distributions. Note, however, that the complexity may be as high as O(N**2) if, for example, the nodes are ordered on increasing latitude.
*/
#[derive(Debug, Clone)]
pub struct DelaunayTriangulation {
    /// The number of nodes in the triangulation
    n: usize,
    /// The `x`-coordinates of the nodes in the triangulation
    x: Vec<f64>,
    /// The `y`-coordinates of the nodes in the triangulation
    y: Vec<f64>,
    /// The `z`-coordinates of the nodes in the triangulation
    z: Vec<f64>,
    /**
    Nodal indexes which, along with [`lptr`], [`lend`], and [`lnew`] define the triangulation as a
    set of [`n`] adjacency lists; counterclockwise-ordered sequences of neighboring nodes such that the first and last neighbors of a boundary node are boundary nodes (the first neighbor of an interior node is arbitrary). In order to distinguish between interior and boundary nodes, the last neighbor of each boundary node is represented by the negative of its index.
    */
    list: Vec<i32>,
    /**
    Set of points ([`list`] indexes) in one-to-one correspondence with the elements of [`list`]. `list[lptr[i]]` indexes the node which follows `list[i]` in cyclical counterclockwise order (the first neighbor follows the last neighbor).
    */
    lptr: Vec<i32>,
    /**
    [`n`] pointers to adjacency lists. `lend[k]` points to the last neighbor of node `k`. `list[lend[k]] < 0` if and only if `k` is a boundary node.
    */
    lend: Vec<i32>,
    /**
    Pointer to the first empty location in [`list`] and [`lptr`] (list length plus one).
    */
    lnew: i32,
}

impl DelaunayTriangulation {
    /**
    # Arguments
    * `x` - The `x`-coordinates of the input nodes.
    * `y` - The `y`-coordinates of the input nodes.
    * `z` - The `z`-coordinates of the input nodes.
    # Returns
    The Delaunay triangulation of the input points.

    # Errors
    * If `x`, `y`, and `z` do not share the same length.
    * If `x`, `y`, and `z` do not define only unit vectors.
    * If the length of `x`, `y`, and `z` is less than `3`.
    * If the first three nodes are collinear.

    # Panics
    * If the number of coordinates is greater than [`i32::MAX`].
    */
    pub fn new(
        x: impl Into<Vec<f64>>,
        y: impl Into<Vec<f64>>,
        z: impl Into<Vec<f64>>,
    ) -> Result<Self, TriangulationError> {
        let x = x.into();
        let y = y.into();
        let z = z.into();

        if x.len() != y.len() || x.len() != z.len() {
            return Err(TriangulationError::IncorrectNodeCount);
        }

        for i in 0..x.len() {
            if ((x[i].powi(2) + y[i].powi(2) + z[i].powi(2)).sqrt() - 1.0).abs() > f64::EPSILON {
                return Err(TriangulationError::NotUnitVectors);
            }
        }

        let n =
            i32::try_from(x.len()).unwrap_or_else(|_| panic!("n must be less than {}", i32::MAX));
        let list_size = if n >= 3 { 6 * (n - 2) } else { 0 } as usize;
        let mut list = vec![0i32; list_size];
        let mut lptr = vec![0i32; list_size];
        let mut lend = vec![0i32; n as usize];
        let mut lnew = 0i32;

        let mut near = vec![0i32; n as usize];
        let mut next = vec![0i32; n as usize];
        let mut dist = vec![0.0f64; n as usize];
        let mut ier = 0i32;

        unsafe {
            trmesh(
                &raw const n,
                x.as_ptr(),
                y.as_ptr(),
                z.as_ptr(),
                list.as_mut_ptr(),
                lptr.as_mut_ptr(),
                lend.as_mut_ptr(),
                &raw mut lnew,
                near.as_mut_ptr(),
                next.as_mut_ptr(),
                dist.as_mut_ptr(),
                &raw mut ier,
            );
        };

        match ier {
            -1 => return Err(TriangulationError::NotEnoughNodes),
            -2 => return Err(TriangulationError::CollinearNodes),
            _ => {}
        }

        Ok(Self {
            n: n as usize,
            x,
            y,
            z,
            list,
            lptr,
            lend,
            lnew,
        })
    }

    /**
    Adds a node to a triangulation of the convex hull of nodes `0, ..., n-2`, producing a
    triangulation of the convex hull of nodes `0, ..., n-1`.

    The algorithm consists of the following steps: node `n` is located relative to the
    triangulation ([find]), its index is added to the data structure ([intadd] or [bdyadd]), and a sequence of swaps ([swptst] and [swap]) are applied to the arcs opposite `n` so that all arcs incident on node `n` and opposite node `n` are locally optimal (statisfy the circumcircle test).

    Thus, if a Delaunay triangulation of nodes `0` through `n - 2` is input, a Delaunay
    triangulation of nodes `0` through `n - 1` will be output.

    # Arguments
    * `start_node`: The index of a node in which [`find`] begins its search.
    * `p`: The coordinate of the new node to add.

    # Errors
    * If `start_node` is invalid.
    * If all nodes (including `start_node`) are collinear (lie on a common geodesic).

    # Panics
    * If `index` is greater than [`i32::MAX`].
     */
    pub fn add_node<'a>(
        &mut self,
        start_node: usize,
        p: impl Into<&'a [f64; 3]>,
    ) -> Result<(), AddNodeError> {
        let k = self.n + 1;
        let p = p.into();
        self.x.push(p[0]);
        self.y.push(p[1]);
        self.z.push(p[2]);

        let new_list_size = 6 * (self.n + 1 - 2);
        self.list.resize(new_list_size, 0);
        self.lptr.resize(new_list_size, 0);
        self.lend.resize(self.n + 1, 0);

        let mut ier = 0i32;
        let k = i32::try_from(k)
            .unwrap_or_else(|_| panic!("The new number of nodes must be less than {}", i32::MAX));
        let start_node_idx = i32::try_from(start_node + 1)
            .unwrap_or_else(|_| panic!("expected start_node to be less than {}", i32::MAX));

        unsafe {
            addnod(
                &raw const start_node_idx,
                &raw const k,
                self.x.as_ptr(),
                self.y.as_ptr(),
                self.z.as_ptr(),
                self.list.as_mut_ptr(),
                self.lptr.as_mut_ptr(),
                self.lend.as_mut_ptr(),
                &raw mut self.lnew,
                &raw mut ier,
            );
        }

        match ier {
            -1 => return Err(AddNodeError::InvalidIndex),
            -2 => return Err(AddNodeError::CollinearNodes),
            _ => {}
        }

        Ok(())
    }

    /**
    Returns the boundary nodes of a triangulation.

    Given a triangulation of `n` nodes on the unit sphere, this method returns an array containing the indexes (if any) of the counterclockwise sequence of boundary nodes, that is, the nodes on the boundary of the convex hull of the set of nodes. The boundary is empty if the nodes do not lie on a single hemisphere. The numbers of boundary nodes, arcs, and triangles are also returned.

    # Panics
    * If the number of nodes, arcs, or triangles in the `DelaunayTriangluation` is set to a value less than 0.
    */
    #[must_use]
    pub fn boundary_nodes(&self) -> BoundaryInfo {
        let n = i32::try_from(self.n)
            .unwrap_or_else(|_| panic!("expected number of nodes to be less than {}", i32::MAX));
        let mut nodes = vec![0i32; self.n];
        let mut nb = 0i32;
        let mut na = 0i32;
        let mut nt = 0i32;
        unsafe {
            bnodes(
                &raw const n,
                self.list.as_ptr(),
                self.lptr.as_ptr(),
                self.lend.as_ptr(),
                nodes.as_mut_ptr(),
                &raw mut nb,
                &raw mut na,
                &raw mut nt,
            );
        }

        let num_nodes = usize::try_from(nb)
            .expect("number of boundary nodes to be greater than or equal to zero");

        nodes.resize(num_nodes, 0);
        let nodes = nodes
            .into_iter()
            .map(|x| {
                usize::try_from(x - 1).unwrap_or_else(|_| {
                    panic!("Expected index to be greater than or equal to zero")
                })
            })
            .collect();

        BoundaryInfo {
            nodes,
            num_arcs: usize::try_from(na)
                .expect("number of arcs to be greater than or equal to zero"),
            num_triangles: usize::try_from(nt)
                .expect("number of triangles to be greater than or equal to zero"),
        }
    }

    /**
    Locates a point relative to a triangulation.

    This method locates a point `p` relative to a [`DelaunayTriangulation`]. If `p` is contained in a triangle, the three vertex indices and barycentric coordinates are returned. Otherwise, the indices of the visible boundary nodes are returned.

    # Arguments:
    * `start_node` - The index of a node at which [`find`] begins its search. Search time depends on
      the proximity of this node to `p`.
    * `p` - The `x`, `y`, and `z` coordinates (in that order) of the point `p` to be located.

    # Returns
    The [`LocationInfo`] of the point relative to the triangulation.

    # Panics
    * If the start node is greater than [`i32::MAX`].
    * If the triangulation is invalid.
    */
    pub fn find<'a>(&self, start_node: usize, p: impl Into<&'a [f64; 3]>) -> LocationInfo {
        let p = p.into();
        let nst = i32::try_from(start_node + 1)
            .unwrap_or_else(|_| panic!("Expected start_node to be less than {}", i32::MAX));

        let mut b1 = 0.0f64;
        let mut b2 = 0.0f64;
        let mut b3 = 0.0f64;
        let mut i1 = 0i32;
        let mut i2 = 0i32;
        let mut i3 = 0i32;
        let n = i32::try_from(self.n).unwrap_or_else(|_| {
            panic!(
                "expected number of nodes in triangulation to be less than {}",
                i32::MAX
            )
        });

        unsafe {
            trfind(
                &raw const nst,
                p.as_ptr(),
                &raw const n,
                self.x.as_ptr(),
                self.y.as_ptr(),
                self.z.as_ptr(),
                self.list.as_ptr(),
                self.lptr.as_ptr(),
                self.lend.as_ptr(),
                &raw mut b1,
                &raw mut b2,
                &raw mut b3,
                &raw mut i1,
                &raw mut i2,
                &raw mut i3,
            );
        };

        let i1 = usize::try_from(i1)
            .unwrap_or_else(|_| panic!("expected i1 to be greater than or equal to zero"));
        let i2 = usize::try_from(i2)
            .unwrap_or_else(|_| panic!("expected i1 to be greater than or equal to zero"));
        let i3 = usize::try_from(i3)
            .unwrap_or_else(|_| panic!("expected i1 to be greater than or equal to zero"));

        match (i1, i2, i3) {
            (0, 0, 0) => LocationInfo::Coplanar,
            (_, _, 0) => LocationInfo::OutsideConvexHull {
                leftmost_visible_index: (i1 - 1),
                rightmost_visible_index: (i2 - 1),
            },
            _ => LocationInfo::InsideTriangle {
                barycentric_coords: [b1, b2, b3],
                bounding_triangle_indices: [(i1 - 1), (i2 - 1), (i3 - 1)],
            },
        }
    }

    /**
    Swaps arcs to force two nodes to be adjacent.

    Given a [`DelaunayTriangulation`] and a pair of nodal indexes, `node_idx_1` and `node_idx_2`, this method will swap arcs as necessary to force `node_idx_1` and `node_idx_2` to be adjacent. Only arcs which intersect `node_idx_1-node_idx-2` are swapped out. The resulting triangulation is as close as possible to a Delaunay triangulation in the sense that all arcs other than `in1-in2` are locally optimal.

    A sequence of calls to [`force_adjacent`] may be used to force the presence of a set of edges defining the boundary of a non-convex and/or multiply connected region, or to introduce barriers into the triangulation. Note that [`get_nearest_nodes`] will not necessarily return closest nodes if the triangulation has been constrained by a call to [`force_adjacent`]. However, this is appropriate in some applications, such as triangle-based interpolation on a nonconvex domain.

    # Arguments
    * `node_idx_1`, `node_idx_2` - The indexes of nodes defining a pair of nodes to be connected
      by an arc.

    # Errors
    * If the two node indexes are identical or not present in the triangulation.
    * If the two nodes cannot be made to be adjacent
    * If an error occurred during optimization of the triangulation

    # Panics
    * If `node_idx_1` or `node_idx_2` are not less than [`i32;:MAX`].
    * If the [`DelaunayTriangulation`] has less than three nodes.
     */
    pub fn force_adjacent(
        &mut self,
        node_idx_1: usize,
        node_idx_2: usize,
    ) -> Result<(), ForceAdjacentError> {
        let in1 = i32::try_from(node_idx_1 + 1)
            .unwrap_or_else(|_| panic!("expected node_idx_1 to be less than {}", i32::MAX));
        let in2 = i32::try_from(node_idx_2 + 1)
            .unwrap_or_else(|_| panic!("expected node_idx_2 to be less than {}", i32::MAX));
        let mut ier = 0;

        let mut lwk = i32::try_from(self.n - 3)
            .unwrap_or_else(|_| panic!("expected at least three nodes in the triangulation"));
        let mut iwk = vec![0; 2 * (self.n - 3)];

        unsafe {
            edge(
                &raw const in1,
                &raw const in2,
                self.x.as_ptr(),
                self.y.as_ptr(),
                self.z.as_ptr(),
                &raw mut lwk,
                iwk.as_mut_ptr(),
                self.list.as_mut_ptr(),
                self.lptr.as_mut_ptr(),
                self.lend.as_mut_ptr(),
                &raw mut ier,
            );
        }

        match ier {
            1 => return Err(ForceAdjacentError::InvalidNodePair),
            3 => return Err(ForceAdjacentError::CannotMakeAdjacent),
            4 => return Err(ForceAdjacentError::OptimError),
            _ => {}
        }

        Ok(())
    }

    /**
    Gets the position of the vertex with the given index.

    # Arguments
    * The nodal index of the vertex.

    # Returns
    The `x`, `y`, and `z` coordinates of the vertex.
    */
    #[must_use]
    pub fn get_vertex_pos(&self, node_idx: usize) -> Option<[f64; 3]> {
        let x = self.x.get(node_idx).copied()?;
        let y = self.y.get(node_idx).copied()?;
        let z = self.z.get(node_idx).copied()?;

        Some([x, y, z])
    }

    /**
    Determines if a point is inside a polygonal region.

    This method locates a point `p` relative to a polygonal region on the surface of the unit sphere, returning `true` if and only if `p` is contained in the region. The region is defined by a cyclically ordered sequence of vertices which form a positively-oriented simple closed curve. Adjacent vertices need not be distinct but the curve must not be self-intersecting. Also, while polygon edges are by definition restricted to a single hemisphere, the region is not so restricted. Its interior is the region to the left as the vertices are traversed in order.

    The algorithm consists of selecting a point `q` in the region and then finding all points at which the great circle defined by `p` and `q` intersects the boundary of the region. `p` lies outside the region if and only if there is an even number of intersection points between `q` and `p`. `q` is taken to be a point immediately to the left of a directed boundary edge -- the first one that results in no consistency-check failures.

    If `p` is close to the polygon boundary, the problem is ill-conditioned and the decision may be incorrect. Also, an incorrect decision may result from a poor choice of `q` (if, for example, a boundary edge lies on the great circle defined by `p` and `q`). A more reliable result could be obtained by a sequence of calls to [`is_inside`] with the vertices cyclically permuted before each call (to alter the choice of `q`).

    # Arguments
    * `p` - The coordinates of the point (unit vector) to be located.
    * `region_node_idx` - The indexes of a cyclically-ordered (and CCW-ordered) sequence of
      vertices that define a region. The last vertex is followed by the first.

    # Returns
    Returns `true` if and only if `p` lies inside the region.

    # Errors
    * If a value in `region_node_idxs` is invalid.
    * If there is a self-intersection described by `region_node_idxs`.
    * If the point is very close to the region boundary

    # Panics
    * If the number of indices defining the region is greater than [`i32::MAX`].
    * If any index is greater than [`i32::MAX`].
     */
    pub fn is_inside<'a>(
        &self,
        p: impl Into<&'a [f64; 3]>,
        region_node_idxs: &[usize],
    ) -> Result<bool, InsideError> {
        let p = p.into();
        let lv = i32::try_from(region_node_idxs.len()).unwrap_or_else(|_| {
            panic!(
                "expected the number of nodes in the region to be less than {}",
                i32::MAX
            )
        });
        let nv = lv;

        let mut xv = Vec::with_capacity(region_node_idxs.len());
        let mut yv = Vec::with_capacity(region_node_idxs.len());
        let mut zv = Vec::with_capacity(region_node_idxs.len());
        let mut listv = Vec::with_capacity(region_node_idxs.len());

        for &idx in region_node_idxs {
            xv.push(self.x[idx]);
            yv.push(self.y[idx]);
            zv.push(self.z[idx]);
            listv.push(
                i32::try_from(idx + 1)
                    .unwrap_or_else(|_| panic!("expected index to be less than {}", i32::MAX)),
            );
        }

        let mut ier = 0;

        let result = unsafe {
            inside(
                p.as_ptr(),
                &raw const lv,
                xv.as_ptr(),
                yv.as_ptr(),
                zv.as_ptr(),
                &raw const nv,
                listv.as_ptr(),
                &raw mut ier,
            )
        };

        match ier {
            2 => return Err(InsideError::InvalidIndexList),
            3 => return Err(InsideError::SelfIntersection),
            4 => return Err(InsideError::ConsistencyFailure),
            _ => {}
        }

        Ok(result)
    }

    /**
    Returns the nearest node to a given point.

    Given a point `p` on the surface of the unit sphere and a [`DelaunayTriangulation`], this method returns the index of the nearest triangulation node to `p`.

    The algorithm consists of implicitly adding `p` to the triangulation, finding the nearest neighbor to `p`, and implicitly deleting `p` from the triangulation. Thus, it is based on the fact that, if `p` is a node in a Delaunay triangulation, the nearest node to `p` is a neighbor of `p`.

    For large values of `n`, this procedure will be faster than the naive approach of computing the distance from `p` to every node.

    Note that the number of candidates for [`nearest_node`] (neighbors of `p`) is limited to `25`.

    # Arguments

    * `p` - The Cartesian coordinates of the point `p` to be located relative to the
      triangulation. It is assumed that `p[0]**2 + p[1]**2 + p[2]**2 = 1`, that is, that the point lies on the unit sphere.
    * `start_node` - The index of the node at which the search is to begin. The search time depends on the proximity of this node to `p`. If no good candidate is known, any value
      between `0` and `n` will do.

    # Returns
    A [`NearestNode`] struct which contains the index of the nearest node to `p`, and the arc length between `p` and the closest node.

    # Errors
    * If the triangulation has less than three nodes or is invalid.

    # Panics
    * If the triangulation is invalid
    * If the number of nodes is greater than [`i32::MAX`].
     */
    pub fn nearest_node<'a>(
        &self,
        p: impl Into<&'a [f64; 3]>,
        start_node: usize,
    ) -> Result<NearestNode, NearestNodeError> {
        if self.n < 3 {
            return Err(NearestNodeError::InvalidTriangulation);
        }
        let p = p.into();
        let n = i32::try_from(self.n)
            .unwrap_or_else(|_| panic!("expected n to be less than {}", i32::MAX));
        let ist = i32::try_from(start_node + 1)
            .unwrap_or_else(|_| panic!("expected ist to be less than {}", i32::MAX));
        let mut al = 0.0;
        let index = unsafe {
            nearnd(
                p.as_ptr(),
                &raw const ist,
                &raw const n,
                self.x.as_ptr(),
                self.y.as_ptr(),
                self.z.as_ptr(),
                self.list.as_ptr(),
                self.lptr.as_ptr(),
                self.lend.as_ptr(),
                &raw mut al,
            )
        };

        Ok(NearestNode {
            index: usize::try_from(index - 1)
                .unwrap_or_else(|_| panic!("expected index to be greater than 0")),
            arc_length: al,
        })
    }

    /**
    Gets the `k` nearest nodes to `node_idx`.

    The algorithm uses the property of a Delaunay triangulation that the `k`-th closest node to `node_idx` is a neighbor of one of the `k-1` closest nodes to `node_idx`.

    # Arguments

    * `node_idx` - The index of the node to find the nearest neighbors.
    * `k` - The number of nearest neighbors to find.

    # Returns
    A vector of [`NearestNode`] structs.

    # Panics
    * If the triangulation is invalid.
    * If the `node_idx` or `k` value is greater than [`i32::MAX`].
    */
    #[must_use]
    pub fn nearest_nodes(&self, node_idx: usize, k: usize) -> Vec<NearestNode> {
        if k == 0 {
            return vec![];
        }
        let mut npts = vec![0i32; k + 1];
        let mut distances = vec![0.0; k];
        npts[0] = i32::try_from(node_idx)
            .unwrap_or_else(|_| panic!("expected node_idx to be less than {}", i32::MAX));

        // NB: Fortran arrays are 1-indexed
        for i in 2..=k + 1 {
            let l = i32::try_from(i)
                .unwrap_or_else(|_| panic!("expected i to be less than {}", i32::MAX));
            let mut df = 0.0f64;
            let mut ier = 0;
            unsafe {
                getnp(
                    self.x.as_ptr(),
                    self.y.as_ptr(),
                    self.z.as_ptr(),
                    self.list.as_ptr(),
                    self.lptr.as_ptr(),
                    self.lend.as_ptr(),
                    &raw const l,
                    npts.as_mut_ptr(),
                    &raw mut df,
                    &raw mut ier,
                );
            }

            distances[i - 2] = df;
        }

        npts.into_iter()
            .skip(1)
            .zip(distances)
            .map(|(index, distance)| NearestNode {
                index: usize::try_from(index - 1).unwrap_or_else(|_| {
                    panic!("Expected index to be greater than or equal to zero")
                }),
                arc_length: (-distance).acos(),
            })
            .collect()
    }

    /**
    Returns the number of neighbors of a node.

    This function returns the number of neighbors of a node `node_idx` in a [`DelaunayTriangulation`].

    The number of neighbors also gives the order of the Voronoi polygon containing the point. Thus, a neighbor count of `6` means the node is contained in a `6`-sided Voronoi region.

    This function is identical to the similarly named function in TRIPACK.

    # Arguments

    * `node_idx` - Index of the node of which to count neighbors.

    # Returns

    The number of neighbors of `node_idx`.

    # Panics

    * If the count of neighbors is less than zero.
     */
    #[must_use]
    pub fn neighbor_count(&self, node_idx: usize) -> usize {
        let lpl = self.lend[node_idx];
        unsafe { nbcnt(&raw const lpl, self.lptr.as_ptr()) }
            .try_into()
            .unwrap_or_else(|_| panic!("expected count to be greater than or equal to 0"))
    }

    /**
    Deletes a node from a triangulation.

    This method deletes node `node_index` (along with all arcs incident on node `node_index`) from a triangulation of `n` nodes on the unit sphere, and inserts arcs as necessary to produce a triangulation of the remaining `n - 1` nodes. If a Delaunay triangulation is input, a Delaunay triangulation will be the result, and thus [`remove_node`] reverses the effect of a call to [`add_node`].

    Note that the deletion may result in all remaining nodes being collinear. This situation is not flagged.

    # Arguments

    * `node_index` - The index (for `x`, `y`, and `z`) of the node to be deleted. `0 <= node_index < n`.

    # Errors

    * If `node_idx` is invalid
    * If not enough space can be reserved for reporting the new arcs
    * If the triangulation is invalid
    * If `node_index` indexes an interior node with four or more neighbors, none of which can be
      swapped out due to collinearity, and `node_index` cannot therefore be deleted.
    * If optimization produces an error

    # Panics
    * If `n` > [`i32::MAX`]
    * If `node_idx` > [`i32::MAX`]
    * If `n < 0`
    */
    pub fn remove_node(&mut self, node_idx: usize) -> Result<(), DeleteNodeError> {
        let mut n = i32::try_from(self.n)
            .unwrap_or_else(|_| panic!("expected n to be less than {}", i32::MAX));

        let neighbors = self.neighbor_count(node_idx);
        let lwk = if self.is_boundary(node_idx) {
            neighbors + 1 - 3
        } else {
            neighbors - 3
        };

        let mut iwk = vec![0i32; 2 * lwk];
        let mut lwk = i32::try_from(lwk)
            .unwrap_or_else(|_| panic!("expected lwk to be less than {}", i32::MAX));
        let mut ier = 0i32;
        let k = i32::try_from(node_idx + 1)
            .unwrap_or_else(|_| panic!("expected node_idx to be less than {}", i32::MAX));

        unsafe {
            delnod(
                &raw const k,
                &raw mut n,
                self.x.as_mut_ptr(),
                self.y.as_mut_ptr(),
                self.z.as_mut_ptr(),
                self.list.as_mut_ptr(),
                self.lptr.as_mut_ptr(),
                self.lend.as_mut_ptr(),
                &raw mut self.lnew,
                &raw mut lwk,
                iwk.as_mut_ptr(),
                &raw mut ier,
            );
        };

        match ier {
            1 => return Err(DeleteNodeError::InvalidNodeIndex),
            2 => return Err(DeleteNodeError::NotEnoughSpace),
            3 => return Err(DeleteNodeError::InvalidTriangulation),
            4 => return Err(DeleteNodeError::Collinear),
            5 => return Err(DeleteNodeError::OptimizationError),
            _ => {}
        }

        self.n = usize::try_from(n)
            .unwrap_or_else(|_| panic!("expected n to be greater than or equal to zero"));

        self.x.resize(self.n, 0.0);
        self.y.resize(self.n, 0.0);
        self.z.resize(self.n, 0.0);
        self.lend.resize(self.n, 0);

        let new_size = 6 * (self.n - 2);
        self.list.resize(new_size, 0);
        self.lptr.resize(new_size, 0);

        Ok(())
    }

    /**
    Get the triangle list from the triangulation.

    # Returns
    The [`MeshData`] containing the triangle list, positions, neighbors, and arcs.

    # Errors
    If the [`DelaunayTriangulation`] is corrupt.

    # Panics
    If the result from [`trlist`] has negative values.
    */
    pub fn triangle_mesh(&self) -> Result<MeshData, TriangleMeshError> {
        let n = i32::try_from(self.n)
            .unwrap_or_else(|_| panic!("number of nodes to be less than {}", i32::MAX));
        let nrow = 9;
        let mut nt = 0;
        let mut ier = 0;
        let max_triangles = 2 * self.n - 4;
        let mut ltri = vec![0i32; (nrow as usize) * max_triangles];
        unsafe {
            trlist(
                &raw const n,
                self.list.as_ptr(),
                self.lptr.as_ptr(),
                self.lend.as_ptr(),
                &raw const nrow,
                &raw mut nt,
                ltri.as_mut_ptr(),
                &raw mut ier,
            );
        };

        if ier == 2 {
            return Err(TriangleMeshError::InvalidTriangulation);
        }

        let mut positions = Vec::with_capacity(self.n);
        for i in 0..self.n {
            positions.push([self.x[i], self.y[i], self.z[i]]);
        }
        let triangle_count = usize::try_from(nt).unwrap_or_else(|_| {
            panic!("expected number of triangles to be greater than or equal to zero")
        });

        let mut indices = Vec::with_capacity(triangle_count * 3);
        let mut arc_indices = Vec::with_capacity(triangle_count);
        let mut neighbors = Vec::with_capacity(triangle_count);

        for t in 0..triangle_count {
            let v1 = usize::try_from(ltri[t * nrow as usize] - 1).unwrap_or_else(|_| {
                panic!("expected index in triangle list to be greater than or equal to zero")
            });
            let v2 = usize::try_from(ltri[1 + t * nrow as usize] - 1).unwrap_or_else(|_| {
                panic!("expected index in triangle list to be greater than or equal to zero")
            });
            let v3 = usize::try_from(ltri[2 + t * nrow as usize] - 1).unwrap_or_else(|_| {
                panic!("expected index in triangle list to be greater than or equal to zero")
            });

            indices.push(v1);
            indices.push(v2);
            indices.push(v3);

            let n1 = usize::try_from(ltri[3 + t * nrow as usize]).unwrap_or_else(|_| {
                panic!("expected index in triangle list to be greater than or equal to zero")
            });
            let n2 = usize::try_from(ltri[4 + t * nrow as usize]).unwrap_or_else(|_| {
                panic!("expected index in triangle list to be greater than or equal to zero")
            });
            let n3 = usize::try_from(ltri[5 + t * nrow as usize]).unwrap_or_else(|_| {
                panic!("expected index in triangle list to be greater than or equal to zero")
            });

            let mut tri_neighbors = [None; 3];
            if n1 > 0 {
                tri_neighbors[0] = Some(n1 - 1);
            }
            if n2 > 0 {
                tri_neighbors[1] = Some(n2 - 1);
            }
            if n3 > 0 {
                tri_neighbors[2] = Some(n3 - 1);
            }

            neighbors.push(tri_neighbors);

            let a1 = usize::try_from(ltri[6 + t * nrow as usize] - 1).unwrap_or_else(|_| {
                panic!("expected index in triangle_list to be greater than or equal to zero")
            });
            let a2 = usize::try_from(ltri[7 + t * nrow as usize] - 1).unwrap_or_else(|_| {
                panic!("expected index in triangle_list to be greater than or equal to zero")
            });
            let a3 = usize::try_from(ltri[8 + t * nrow as usize] - 1).unwrap_or_else(|_| {
                panic!("expected index in triangle_list to be greater than or equal to zero")
            });

            arc_indices.push([a1, a2, a3]);
        }

        Ok(MeshData {
            positions,
            indices,
            arc_indices,
            neighbors,
        })
    }

    /**
    Returns triangle circumcenters and other information.

    Given a [`DelaunayTriangulation`] of nodes on the surface of a unit sphere, this method returns the set of triangle circumcenters corresponding to Voronoi vertices, along with the circumradii.

    A triangle circumcenter is the point (unit vector) lying at the same angular distance from the three vertices and contained in the same hemisphere as the vertices. (Note that the negative of a circumcenter is also equidistant from the vertices.) If the triangulation covers the surface, the Voronoi vertices are the circumcenters of the triangle in the [`DelaunayTriangulation`]. In this case, the structure is not modified.

    On the other hand, if the nodes are contained in a single hemisphere, the triangulation is implicitly extended to the entire surface by adding pseudo-arcs (of length greater than 180 degrees) between boundary nodes forming pseudo-triangles whose 'circumcenters' are included in the list. This extension of the triangulation actually consists of a triangulation of the set of boundary nodes in which the swap test is reversed (a non-empty circumcircle test). The negative circumcenters are stored as pseudo-triangle 'circumcenters'. The [`DelaunayTriangulation`] is modified to contain this extended triangulation. If it is necessary to save the original structure, it should be cloned prior to calling this method.

    # Returns
    A [`Vec`] of [`VoronoiCell`] structures.

    # Errors
    * If there are less than `3` nodes in the triangulation.
    * If any degenerate triangle is encountered.

    # Panics
    * If there are more than [`i32::MAX`] nodes in the triangulation.
    */
    pub fn voronoi_cells(&mut self) -> Result<Vec<VoronoiCell>, VoronoiCellError> {
        let n = i32::try_from(self.n).unwrap_or_else(|_| {
            panic!(
                "expected number of nodes in triangulation to be less than {}",
                i32::MAX,
            );
        });
        let boundary_nodes = self.boundary_nodes();
        let ncol = i32::try_from(boundary_nodes.nodes.len().saturating_sub(2).max(1))
            .unwrap_or_else(|_| {
                panic!(
                    "expected the number of boundary nodes to be less than {}",
                    i32::MAX
                )
            });

        let mut ltri = vec![0; 6 * ncol as usize];
        let nt = 2 * self.n - 4;
        let mut listc = vec![0; 3 * nt];
        let mut nb = 0;
        let mut xc = vec![0.0; nt];
        let mut yc = vec![0.0; nt];
        let mut zc = vec![0.0; nt];
        let mut rc = vec![0.0; nt];
        let mut ier = 0;

        unsafe {
            crlist(
                &raw const n,
                &raw const ncol,
                self.x.as_ptr(),
                self.y.as_ptr(),
                self.z.as_ptr(),
                self.list.as_ptr(),
                self.lend.as_ptr(),
                self.lptr.as_mut_ptr(),
                &raw mut self.lnew,
                ltri.as_mut_ptr(),
                listc.as_mut_ptr(),
                &raw mut nb,
                xc.as_mut_ptr(),
                yc.as_mut_ptr(),
                zc.as_mut_ptr(),
                rc.as_mut_ptr(),
                &raw mut ier,
            );
        }

        match ier {
            1 => return Err(VoronoiCellError::NotEnoughNodes),
            3 => return Err(VoronoiCellError::DegenerateTriangle),
            _ => {}
        }

        let mut voronoi_cells = Vec::with_capacity(nt);

        for i in 0..nt {
            voronoi_cells.push(VoronoiCell {
                position: [xc[i], yc[i], zc[i]],
                radius: rc[i],
            });
        }

        Ok(voronoi_cells)
    }

    fn is_boundary(&self, node_idx: usize) -> bool {
        let mut lp = (self.lend[node_idx] - 1) as usize;
        loop {
            let neighbor = self.list[lp];
            if neighbor < 0 {
                return true;
            }
            if (neighbor - 1) as usize == node_idx {
                break;
            }
            lp = self.lptr[lp] as usize;
        }

        false
    }
}

/// Computes the arc cosine with input truncated in the range `[-1, 1]`.
#[must_use]
pub fn arc_cosine(c: f64) -> f64 {
    unsafe { stripack_sys::ffi::arc_cosine(&raw const c) }
}

/**
Computes the areas of a spherical triangle on the unit sphere.

# Arguments

* `v1`, `v2`, `v3` - The Cartesian coordinates of unit vectors (the three triangle vertices in any
  order). These vectors, if nonzero, are implicitly scaled to have length `1`.

# Returns

The area of the spherical triangle defined by `v1`, `v2`, and `v3`, in the range `[0, 2 * PI]` (the area of a hemisphere). `0` if and only if `v1`, `v2`, and `v3` lie in (or close to) a plane containing the origin.
*/
pub fn areas<'a, 'b, 'c>(
    v1: impl Into<&'a [f64; 3]>,
    v2: impl Into<&'b [f64; 3]>,
    v3: impl Into<&'c [f64; 3]>,
) -> f64 {
    let v1 = v1.into();
    let v2 = v2.into();
    let v3 = v3.into();

    unsafe { stripack_sys::ffi::areas(v1.as_ptr(), v2.as_ptr(), v3.as_ptr()) }
}

/**
Returns the circumcenter of a spherical triangle.

Returns the circumcenter of a spherical triangle on the unit sphere: the point on the sphere surface that is equally distant from the three triangle vertices and lies in the same hemisphere, where distance is taken to be arc-length on the sphere surface.

# Arguments

* `v1`, `v2`, `v3` - The coordinates of teh three triangle vertices (unit vectors) in
  counterclockwise order.

# Returns
The coordinates of the circumcenter. `c = (v2 - v1) X (v3 - v1)` normalized to a unit vector.

# Errors

* If `v1`, `v2`, and `v3` are collinear.
*/
pub fn circumcenter<'a, 'b, 'c>(
    v1: impl Into<&'a [f64; 3]>,
    v2: impl Into<&'b [f64; 3]>,
    v3: impl Into<&'c [f64; 3]>,
) -> Result<[f64; 3], CircumcenterError> {
    let v1 = v1.into();
    let v2 = v2.into();
    let v3 = v3.into();
    let mut c = [0.0f64; 3];
    let mut ier = 0i32;
    unsafe {
        circum(
            v1.as_ptr(),
            v2.as_ptr(),
            v3.as_ptr(),
            c.as_mut_ptr(),
            &raw mut ier,
        );
    }

    if ier == 1 {
        return Err(CircumcenterError::Collinear);
    }

    Ok(c)
}

/// Converts from Cartesian to spherical coordinates (latitude, longitude, radius).
pub fn spherical_coordinates<'a>(p: impl Into<&'a [f64; 3]>) -> SphericalCoordinates {
    let p = p.into();
    let mut plat = 0.0;
    let mut plon = 0.0;
    let mut pnrm = 0.0;
    unsafe {
        scoord(
            &raw const p[0],
            &raw const p[1],
            &raw const p[2],
            &raw mut plat,
            &raw mut plon,
            &raw mut pnrm,
        );
    };

    SphericalCoordinates {
        latitude: plat,
        longitude: plon,
        norm: pnrm,
    }
}

/**
Transform spherical coordinates into Cartesian coordinates.

# Arguments
* `latitudes` - The latitudes of the nodes in radians.
* `longitudes` - The longitudes of the nodes in radians.

# Returns
A vector of the transformed spherical coordinates in the range `[-1, 1]`. `x**2 + y**2 + z**2 = 1`.

# Panics
* If latitudes and longitudes do not have the same length.
* If the number of latitudes or longitudes is greater than [`i32::MAX`].
*/
pub fn cartesian_coordinates<'a, 'b>(
    latitudes: impl Into<&'a [f64]>,
    longitudes: impl Into<&'b [f64]>,
) -> Vec<[f64; 3]> {
    let latitudes = latitudes.into();
    let longitudes = longitudes.into();
    assert_eq!(
        latitudes.len(),
        longitudes.len(),
        "latitudes and longitudes must have the same length."
    );

    let size = latitudes.len();
    let mut x = vec![0.0; size];
    let mut y = vec![0.0; size];
    let mut z = vec![0.0; size];
    let n = i32::try_from(size).unwrap_or_else(|_| {
        panic!(
            "expected length of latitudes or longitudes to be less than {}",
            i32::MAX
        )
    });

    unsafe {
        trans(
            &raw const n,
            latitudes.as_ptr(),
            longitudes.as_ptr(),
            x.as_mut_ptr(),
            y.as_mut_ptr(),
            z.as_mut_ptr(),
        );
    };

    let mut result = Vec::with_capacity(size);
    for i in 0..size {
        result.push([x[i], y[i], z[i]]);
    }

    result
}

#[cfg(test)]
mod test {
    use super::*;
    use proptest::prelude::*;
    use std::f64::consts::PI;

    fn fibonacci_sphere(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let gr = f64::midpoint(1.0, 5.0f64.sqrt());
        let mut x_points = Vec::with_capacity(n);
        let mut y_points = Vec::with_capacity(n);
        let mut z_points = Vec::with_capacity(n);

        for i in 0..n {
            let theta = 2.0 * PI * i as f64 / gr;
            let phi = (1.0 - 2.0 * (i as f64 + 0.5) / n as f64).acos();
            let x = phi.sin() * theta.cos();
            let y = phi.sin() * theta.sin();
            let z = phi.cos();

            let norm = (x.powi(2) + y.powi(2) + z.powi(2)).sqrt();

            x_points.push(x / norm);
            y_points.push(y / norm);
            z_points.push(z / norm);
        }

        (x_points, y_points, z_points)
    }

    proptest! {
        #[test]
        fn test_creating_delaunay_triangulation(n in 3usize..30) {
            let (x, y, z) = fibonacci_sphere(n);
            let triangulation = DelaunayTriangulation::new(x.clone(), y.clone(), z.clone()).expect("to create a triangulation");
            for i in 0..n {
                prop_assert_eq!(triangulation.get_vertex_pos(i).expect("to find position for vertex"), [x[i], y[i], z[i]]);
            }
        }

        #[test]
        fn test_creating_delaunay_triangulation_iteratively(n in 3usize..30) {
            let (x, y, z) = fibonacci_sphere(n);
            let test_triangulation = DelaunayTriangulation::new(x.clone(), y.clone(), z.clone()).expect("to create a triangulation");
            let mut triangulation = DelaunayTriangulation::new(vec![], vec![], vec![]).expect("to create a triangulation");
            for i in 0..n {
                let position = [x[i], y[i], z[i]];
                triangulation.add_node(0, &position).expect("to add node");
                prop_assert_eq!(triangulation.get_vertex_pos(i).expect("to find position for vertex"), position);
                prop_assert_eq!(triangulation.get_vertex_pos(i).expect("to find position for vertex"), test_triangulation.get_vertex_pos(i).expect("to find position for vertex"));
            }
        }
    }

    #[test]
    fn test_creating_delaunay_triangulation_rejects_non_unit_vectors() {
        let triangulation = DelaunayTriangulation::new(vec![2.0], vec![2.0], vec![2.0]);
        let Err(TriangulationError::NotUnitVectors) = triangulation else {
            panic!("unexpected result: {triangulation:?}");
        };
    }

    #[test]
    fn test_creating_delaunay_triangulation_rejects_degenerate_triangle() {
        let triangulation = DelaunayTriangulation::new(
            vec![1.0, 0.0, -1.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0],
        );
        let Err(TriangulationError::CollinearNodes) = triangulation else {
            panic!("unexpected result: {triangulation:?}");
        };
    }

    #[test]
    fn test_creating_delaunay_triangulation_rejects_mismatched_inputs() {
        let triangulation = DelaunayTriangulation::new(vec![], vec![0.0], vec![0.0, 0.0]);
        let Err(TriangulationError::IncorrectNodeCount) = triangulation else {
            panic!("unexpected result: {triangulation:?}");
        };
    }

    #[test]
    fn test_creating_delaunay_triangulation_rejects_not_enough_nodes() {
        let (x, y, z) = fibonacci_sphere(2);
        let triangulation = DelaunayTriangulation::new(x, y, z);
        let Err(TriangulationError::NotEnoughNodes) = triangulation else {
            panic!("unexpected result: {triangulation:?}");
        };
    }

    proptest! {
        #[test]
        fn test_boundary_nodes(n in 9usize..50) {
            let (x, y, z) = fibonacci_sphere(n);
            let mut x_hemi = Vec::with_capacity(n);
            let mut y_hemi = Vec::with_capacity(n);
            let mut z_hemi = Vec::with_capacity(n);
            for i in 0..n {
                if x[i] > 0.1 {
                    x_hemi.push(x[i]);
                    y_hemi.push(y[i]);
                    z_hemi.push(z[i]);
                    println!("[{}, {}, {}]", x[i], y[i], z[i]);
                }
            }

            let n_hemi = x_hemi.len();
            let triangulation = DelaunayTriangulation::new(x_hemi, y_hemi, z_hemi).expect("to make a triangulation");
            let boundary = triangulation.boundary_nodes();
            prop_assert!(!boundary.nodes.is_empty());
            prop_assert!(boundary.nodes.len() >= 3);
            let nb = boundary.nodes.len();
            prop_assert_eq!(boundary.num_triangles, 2 * n_hemi - 2 - nb);
            prop_assert_eq!(boundary.num_arcs, 3 * n_hemi - 3 - nb);

            let sphere_triangulation = DelaunayTriangulation::new(x, y, z).expect("to make a triangulation");
            let sphere_boundary = sphere_triangulation.boundary_nodes();
            prop_assert!(sphere_boundary.nodes.is_empty());
            prop_assert_eq!(sphere_boundary.num_triangles, 2 * n - 4);
            prop_assert_eq!(sphere_boundary.num_arcs, 3 * n - 6);
        }
    }

    #[test]
    fn test_neighbor_count_tetrahedron() {
        let (x, y, z) = fibonacci_sphere(4);
        let triangulation = DelaunayTriangulation::new(x, y, z).expect("to make a triangulation");

        for i in 0..4 {
            let count = triangulation.neighbor_count(i);
            assert_eq!(count, 3, "expected 3 neighbors, got {count}");
        }
    }

    proptest! {
        #[test]
        fn test_neighbor_count(n in 5usize..50) {
            let (x, y, z) = fibonacci_sphere(n);
            let triangulation = DelaunayTriangulation::new(x, y, z).expect("to make a triangulation");

            let boundary = triangulation.boundary_nodes();
            let mut total_neighbors = 0;
            for i in 0..n {
                let count = triangulation.neighbor_count(i);

                prop_assert!(count >= 3, "node {} only has {} neighbors", i, count);
                prop_assert!(count < n, "node {} has {} neighbors", i, count);
                total_neighbors += count;
            }

            prop_assert_eq!(total_neighbors, 2 * boundary.num_arcs, "sum of neighbor counts ({}) shoudl equal 2 * arcs ({})", total_neighbors, 2 * boundary.num_arcs);
        }
    }
}
