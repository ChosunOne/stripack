# STRIPACK

![build](https://github.com/ChosunOne/stripack/workflows/Run%20Tests/badge.svg)
[![Documentation](https://docs.rs/stripack/badge.svg)](https://docs.rs/stripack)
![Crates.io](https://img.shields.io/crates/v/stripack)
![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)
![Downloads](https://img.shields.io/crates/d/stripack)

Safe Rust wrapper for STRIPACK - Delaunay triangulation on the unit sphere.
 Features
- **Delaunay Triangulation**: Create triangulations from points on the unit sphere
- **Dynamic Operations**: Add/remove nodes while maintaining Delaunay property
- **Triangle Mesh**: Extract triangle indices, neighbor information, and edge data
- **Voronoi Diagrams**: Generate Voronoi cells from triangulation
- **Point Location**: Find which triangle contains a point or locate nearest nodes
- **Boundary Detection**: Identify boundary nodes for hemispherical data
- **Utility Functions**: Circumcenters, spherical areas, coordinate conversions

## Quick Start
```rust
use stripack::DelaunayTriangulation;
// Create a triangulation from unit vectors
let triangulation = DelaunayTriangulation::new(x, y, z)?;
// Get triangle mesh
let mesh = triangulation.triangle_mesh()?;
// Find nearest node to a point
let nearest = triangulation.nearest_node(&[1.0, 0.0, 0.0], 0)?;
// Generate Voronoi diagram
let cells = triangulation.voronoi_cells()?;
```

## Safety
This crate provides a safe wrapper around the STRIPACK Fortran library via stripack-sys.
All unsafe FFI calls are encapsulated with proper error handling.

## License
MIT OR Apache-2.0

## Attribution

**STRIPACK** — Delaunay Triangulation and Voronoi Diagram on the Surface of a Sphere

- Original author: Robert J. Renka, University of North Texas
- Reference: R. J. Renka, "Algorithm 772: STRIPACK: Delaunay Triangulation and Voronoi Diagram on the Surface of a Sphere", *ACM Transactions on Mathematical Software*, Vol. 23, No. 3, September 1997, pp. 416-434.
- DOI: [10.1145/275323.275329](https://doi.org/10.1145/275323.275329)
