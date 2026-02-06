use stripack_sys::ffi::{
    addnod, bnodes, circum, delnod, getnp, nbcnt, nearnd, scoord, trans, trfind, trlist, trmesh,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TriangulationError {
    #[error("Not enough nodes")]
    NotEnoughNodes,
    #[error("The first three nodes are collinear")]
    CollinearNodes,
    #[error("All coordinate inputs must have the same length")]
    IncorrectNodeCount,
    #[error("All coordinates must form unit vectors")]
    NotUnitVectors,
}

#[derive(Debug, Error)]
pub enum AddNodeError {
    #[error("Invalid k value")]
    InvalidIndex,
    #[error("All nodes are collinear")]
    CollinearNodes,
}

#[derive(Debug, Error)]
pub enum CircumcenterError {
    #[error("All coordinates are collinear")]
    Collinear,
}

#[derive(Debug, Error)]
pub enum DeleteNodeError {
    #[error("Invalid node index")]
    InvalidNodeIndex,
    #[error("Not enough space")]
    NotEnoughSpace,
    #[error("Invalid or corrupt triangulation")]
    InvalidTriangulation,
    #[error(
        "node_index indexes an interior node with four or more neighbors, none of which can be swapped out due to collinearity"
    )]
    Collinear,
    #[error("Optimization error")]
    OptimizationError,
}

#[derive(Debug, Error)]
pub enum NearestNodeError {
    #[error("Invalid or corrupt triangulation")]
    InvalidTriangulation,
}

#[derive(Debug, Error)]
pub enum TriangleMeshError {
    #[error("Invalid or corrupt triangulation")]
    InvalidTriangulation,
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
    pub positions: Vec<[f64; 3]>,
    pub indices: Vec<usize>,
    pub arc_indices: Vec<[usize; 3]>,
    pub neighbors: Vec<[Option<usize>; 3]>,
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
    pub fn add_node(
        &mut self,
        start_node: usize,
        p: impl for<'a> Into<&'a [f64; 3]>,
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
    pub fn get_nearest_nodes(&self, node_idx: usize, k: usize) -> Vec<NearestNode> {
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
