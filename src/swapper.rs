// use localsearch;
// use permutation::Permutation;
// use fdg_sim::{Force, Field, ForceGraph, Node};
// use crate::crossings::crossings_with_permutation;

// #[derive(Copy, Clone)]
// pub struct Swapper<T: Field> {
//  neighborhood_size: usize
// }

// impl<T: Field, const D: usize, N, E> Force<T, D, N, E> for Swapper<T> {
//     fn apply(&mut self, graph: &mut ForceGraph<T, D, N, E>) {
//         graph
//             .node_weights_mut()
//             .for_each(|Node(_, pos)| pos.coords.scale_mut(self.factor));
//     }
// }


// This force will swap random nodes within a neighborhood distance of each other
// and optimize for minimum crossings.
//
// We will track the specific permutation for each swap iteration and use a
// local search algorithm to find the best permutation.
//
// Ideally this can be used in tandem with other forces simultaneously.