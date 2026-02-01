use stripack_sys::ffi::trmesh;
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
    n: i32,
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

        let n = i32::try_from(x.len()).unwrap_or_else(|_| panic!("n < {}", i32::MAX));
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
            n,
            x,
            y,
            z,
            list,
            lptr,
            lend,
            lnew,
        })
    }
}
