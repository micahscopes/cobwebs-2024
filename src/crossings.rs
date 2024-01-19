use fdg_sim::{
    petgraph::stable_graph::{EdgeIndex, IndexType, NodeIndex},
    Field, ForceGraph, Node,
};
use geo::Coord;
use intersect2d;
use permutation::Permutation;

fn apply_permutation<Ix>(permutation: &Permutation, node: NodeIndex<Ix>) -> NodeIndex<Ix>
where
    Ix: IndexType,
{
    NodeIndex::<Ix>::new(permutation.apply_idx(node.index()))
}

pub fn crossings<T: Field, const D: usize, N: Clone, E>(
    graph: &ForceGraph<T, D, N, E>,
) -> anyhow::Result<usize>
where
    f64: From<T>,
{
    let edge_lines = graph
        .edge_indices()
        .map(|edge_index| {
            let edge = graph.edge_endpoints(edge_index).unwrap();
            let Node(_, source) = graph.node_weight(edge.0).unwrap();
            let Node(_, target) = graph.node_weight(edge.1).unwrap();
            geo::Line::<f64>::new(
                Coord::<f64>::from((source.coords[0].into(), source.coords[1].into())),
                Coord::<f64>::from((target.coords[0].into(), target.coords[1].into())),
            )
        })
        .collect::<Vec<geo::Line<f64>>>();

    let results = intersect2d::algorithm::AlgorithmData::<f64>::default()
        .with_ignore_end_point_intersections(true)?
        .with_lines(edge_lines.into_iter())?
        .compute()?;
    Ok(results.len())
}

pub fn crossings_with_permutation<T: Field, const D: usize, N: Clone, E>(
    graph: &ForceGraph<T, D, N, E>,
    permutation: &Permutation,
) -> anyhow::Result<usize>
where
    f64: From<T>,
{
    let edge_lines = graph
        .edge_indices()
        .map(|edge_index| {
            let edge = graph.edge_endpoints(edge_index).unwrap();
            let source_index_permuted = apply_permutation(&permutation, edge.0);
            let target_index_permuted = apply_permutation(&permutation, edge.1);
            let Node(_, source) = graph.node_weight(source_index_permuted).unwrap();
            let Node(_, target) = graph.node_weight(target_index_permuted).unwrap();
            geo::Line::<f64>::new(
                Coord::<f64>::from((source.coords[0].into(), source.coords[1].into())),
                Coord::<f64>::from((target.coords[0].into(), target.coords[1].into())),
            )
        })
        .collect::<Vec<geo::Line<f64>>>();

    let results = intersect2d::algorithm::AlgorithmData::<f64>::default()
        .with_ignore_end_point_intersections(true)?
        .with_lines(edge_lines.into_iter())?
        .compute()?;
    // .compute_with_permutation(permutation)?;
    // println!("results: {:?}", results.len());
    Ok(results.len())
}
