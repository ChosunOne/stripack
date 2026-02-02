use stripack_sys::ffi::{addnod, bnodes, circum, delnod, trmesh};
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
pub struct NodeDeletionInfo {
    /// The indexes of the endpoints of the new arcs added.
    pub new_arc_endpoints: Vec<usize>,
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
    Deletes a node from a triangulation.

    This method deletes node `node_index` (along with all arcs incident on node `node_index`) from a triangulation of `n` nodes on the unit sphere, and inserts arcs as necessary to produce a triangulation of the remaining `n - 1` nodes. If a Delaunay triangulation is input, a Delaunay triangulation will be the result, and thus [`delete_node`] reverses the effect of a call to [`add_node`].

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
    pub fn delete_node(&mut self, node_idx: usize) -> Result<(), DeleteNodeError> {
        let mut n = i32::try_from(self.n)
            .unwrap_or_else(|_| panic!("expected n to be less than {}", i32::MAX));

        //TODO: Calculate the correct value using nnb - 3, where nnb is the number of
        //neighbors of node k, including an extra pseudo-node if k is a boundary node.
        let mut lwk = n;
        let mut iwk = vec![0i32; 2 * self.n];
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
