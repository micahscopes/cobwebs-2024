use std::collections::{HashMap, HashSet};

use fdg_sim::petgraph::adj::NodeIndex;
// use fdg_sim::petgraph::stable_graph::NodeIndex;
use fdg_sim::{Field, Force, ForceGraph, Node};
use localsearch::optim::{LocalSearchOptimizer, TabuList, TabuSearchOptimizer};
use localsearch::OptModel;
use localsearch::OptProgress;
use permutation::Permutation;
use rand::seq::IteratorRandom;

use fdg_sim::petgraph::adj::IndexType;
use shuffle::irs::Irs;
use shuffle::shuffler::Shuffler;
use crate::crossings_with_permutation;

// a cloneable node type

struct CloneableNode<T: Field, const D: usize, N: Clone>(Node<T, D, N>);
impl<T: Field, const D: usize, N: Clone> Clone for CloneableNode<T, D, N> {
    fn clone(&self) -> Self {
        Self(Node (self.0.0.clone(), self.0.1.clone()))
    }
}

struct GraphSwapperModel<'a, T: Field, const D: usize, N: Clone, E: Send> {
    graph: &'a ForceGraph<T, D, N, E>,
    neighborhood_size: usize,
    graph_neighbors: Option<HashMap<NodeIndex<u32>, HashSet<NodeIndex<u32>>>>,
    // permutation: Permutation,
    // other fields as needed
}

use fdg_sim::petgraph::visit::Bfs;

impl<'a, T: Field, const D: usize, N: Send + Sync + Clone, E: Send + Sync>
    GraphSwapperModel<'a, T, D, N, E>
{
    pub fn new(graph: &'a ForceGraph<T, D, N, E>, neighborhood_size: usize) -> Self {
        Self {
            graph,
            neighborhood_size,
            graph_neighbors: None,
        }
    }
    pub fn cache_neighbors(&mut self) {
        let mut graph_neighbors: HashMap<NodeIndex, HashSet<NodeIndex>> = HashMap::new();
        let neighborhood_size = self.neighborhood_size;

        for node in self.graph.node_indices() {
            let mut level = 0;
            let mut neighbors = HashSet::new();
            let mut visited = HashSet::new();
            let mut start = HashSet::from_iter(vec![node]);
            while level < neighborhood_size {
                // let new_neighbors = self
                //     .graph
                //     .neighbors_undirected(node);
                let new_neighbors = start
                    .iter()
                    .flat_map(|&node| self.graph.neighbors_undirected(node))
                    // .map(|node| node.index())
                    .collect::<HashSet<_>>();

                visited.extend(start.iter().cloned());
                neighbors.extend(new_neighbors.iter().map(|node| node.index() as u32));
                start = new_neighbors
                    .difference(&visited)
                    .cloned()
                    .collect::<HashSet<_>>();

                level += 1;
            }
            graph_neighbors.insert(node.index() as u32, neighbors);
        }
    }
}

type SolutionType = Permutation;
type ScoreType = usize;
type TransitionType = ();

impl<'a, T: Field, const D: usize, N: Send + Sync + Clone, E: Send + Sync> OptModel
    for GraphSwapperModel<'a, T, D, N, E>
where
    f64: From<T>,
{
    type SolutionType = SolutionType;
    type TransitionType = TransitionType;
    type ScoreType = ScoreType;

    fn evaluate_solution(&self, state: &Self::SolutionType) -> Self::ScoreType {
        let score = crossings_with_permutation(&self.graph, state).unwrap();
        // println!("score: {}, permutation: {:?}", score, state);
        score
    }

    // fn generate_trial_solution<R: rand::Rng>(
    //     &self,
    //     current_state: &Self::SolutionType,
    //     rng: &mut R,
    //     current_score: Option<Self::ScoreType>,
    // ) -> (Self::SolutionType, Self::TransitionType, Self::ScoreType) {
    //     let mut idx = Vec::from_iter(0..self.graph.node_count());
    //     let a = rng.gen_range(0..idx.len());
    //     let mut b = 0;
    //     for _ in 0..1000 {
    //         b = rng.gen_range(0..idx.len());
    //         if a != b {
    //             // idx.swap(a, b);
    //             break;
    //         }
    //     }
    //     idx.swap(a, b);
    //     let new_solution = &Permutation::oneline(idx).inverse() * current_state;
    //     let new_score = self.evaluate_solution(&new_solution);
    //     let trial = (new_solution, (), new_score);
    //     // println!("trial: {:?}", trial);
    //     trial
    // }

    fn generate_trial_solution<R: rand::Rng>(
        &self,
        current_solution: &Self::SolutionType,
        rng: &mut R,
        current_score: Option<Self::ScoreType>,
    ) -> (Self::SolutionType, Self::TransitionType, Self::ScoreType) {
        // this time we're going to pick a random node and swap it with another random node within the specified neighborhood size
        let graph = self.graph;
        let node = graph.node_indices().choose(rng).unwrap();
        let neighborhood_size = self.neighborhood_size;

        // we need to recursively collect the neighbors of the node within the neighborhood size
        // let neighbors = graph.neighbors(node).collect::<HashSet<_>>();

        // pick a random neighbor
        // if the node has no neighbors, then we can't swap it with anything so we'll just return the current solution
        
        let mut neighbors;
        let neighbors = match &self.graph_neighbors {
            Some(graph_neighbors) => graph_neighbors.get(&(node.index() as u32)).unwrap(),
            None => {
                let mut level = 0;
                neighbors = HashSet::new();
                let mut visited = HashSet::new();
                let mut start = HashSet::from_iter(vec![node]);
                while level < neighborhood_size {
                    // let new_neighbors = self
                    //     .graph
                    //     .neighbors_undirected(node);
                    let new_neighbors = start
                        .iter()
                        .flat_map(|&node| self.graph.neighbors_undirected(node))
                        .collect::<HashSet<_>>();

                    visited.extend(start.iter().cloned());
                    neighbors.extend(new_neighbors.iter().map(|node| node.index() as u32));
                    start = new_neighbors
                        .difference(&visited)
                        .cloned()
                        .collect::<HashSet<_>>();

                    level += 1;
                }
                &neighbors
            }
        };

        let scramble_range = [2,4];
        // scramble a random number of neighbors in the scramble range
        let n_to_scramble = rng.gen_range(scramble_range[0]..scramble_range[1]);


        let node_index = node.index() as u32;
        // let neighbor = neighbors.iter().choose(rng).unwrap_or(&node_index);
        let random_neighbors = neighbors.iter().choose_multiple(rng, n_to_scramble);
        let mut random_indices = random_neighbors.iter().map(|node| node.index() as u32).collect::<Vec<_>>();
        // random_indices.push(node_index);

        // scramble their indices
        let mut irs = Irs::default();
        // let mut random_indices_scrambled = random_indices.clone();
        let mut scrambling_indices = Vec::from_iter(0..random_indices.len());


        let _ = irs.shuffle(&mut scrambling_indices, rng);

        // println!("random_indices: {:?}, scrambling_indices: {:?}", random_indices, scrambling_indices);

        let mut idx = Vec::from_iter(0..self.graph.node_count());
        // let a = node.index();
        // let b = neighbor.index();
        // idx.swap(a, b);

        for i in 0..random_indices.len() {
            // println!("exchanging {} with {}", random_indices[i], random_indices[scrambling_indices[i]]);
            idx[random_indices[i] as usize] = random_indices[scrambling_indices[i]] as usize;
            // println!("idx: {:?}", idx);
        }

        // println!("scrambled idx: {:?}", idx);

        let new_solution = &Permutation::oneline(idx).inverse() * current_solution;
        let new_score = self.evaluate_solution(&new_solution);
        let trial = (new_solution, (), new_score);
        // println!("trial: {:?}", trial);
        trial
    }

    fn generate_random_solution<R: rand::Rng>(
        &self,
        rng: &mut R,
    ) -> Result<Self::SolutionType, Box<dyn std::error::Error>> {
        // let mut irs = Irs::default();
        // let mut idx = Vec::from_iter(0..self.graph.node_count());
        // irs.shuffle(&mut idx, rng);
        // Ok(Permutation::oneline(idx).inverse())
        Ok(Permutation::one(self.graph.node_count()))
    }
}

#[derive(Debug, Clone)]
struct PermutationTabuList {
    tabu_list: Vec<SolutionType>,
    max_size: usize,
}
impl PermutationTabuList {
    pub fn new(max_size: usize) -> Self {
        Self {
            tabu_list: Vec::<SolutionType>::new(),
            max_size,
        }
    }
}

impl TabuList for PermutationTabuList {
    type Item = (SolutionType, TransitionType);
    fn contains(&self, item: &Self::Item) -> bool {
        self.tabu_list.iter().any(|x| x == &item.0)
    }
    fn append(&mut self, item: Self::Item) {
        self.tabu_list.push(item.0);
        if self.tabu_list.len() > self.max_size {
            self.tabu_list.remove(0);
        }
    }
}

pub struct Swapper {
    neighborhood_size: usize,
    optimizer: TabuSearchOptimizer<PermutationTabuList>,
    tabu_list: Option<PermutationTabuList>,
    // other fields as needed
}

impl Swapper {
    pub fn new(
        neighborhood_size: usize,
        patience: usize,
        n_trials: usize,
        return_iter: usize,
    ) -> Self {
        // let optimizer = TabuSearchOptimizer::new(100, 100, 1);
        // let tabu_list = PermutationTabuList::new(20);
        let optimizer =
            TabuSearchOptimizer::<PermutationTabuList>::new(patience, n_trials, return_iter);
        Self {
            neighborhood_size,
            optimizer,
            tabu_list: None,
            // initialize other fields as needed
        }
    }
}

impl<T: Field, const D: usize, N: Send + Sync + Clone, E: Send + Sync> Force<T, D, N, E> for Swapper
where
    f64: From<T>,
{
    fn apply(&mut self, graph: &mut ForceGraph<T, D, N, E>) {
        let mut model = GraphSwapperModel {
            graph: graph,
            neighborhood_size: self.neighborhood_size,
            graph_neighbors: None,
        };
        model.cache_neighbors();
        
        

        let callback = |op: OptProgress<SolutionType, ScoreType>| {
            // println!("progress {:?}", op);
            // pb.set_message(format!("best score {:e}", op.score.into_inner()));
            // pb.set_position(op.iter as u64);
        };

        let initial = Permutation::one(graph.node_count());
        let current_score = crossings_with_permutation(&graph, &initial).unwrap();

        if current_score == 0 {
            return;
        }

        // let tabu_list = PermutationTabuList::new(2000);
        // if we already have a tabu list, use it
        let tabu_list = match &self.tabu_list {
            Some(tabu_list) => tabu_list.clone(),
            None => PermutationTabuList::new(200000),
        };

        let (best_permutation, best_score, tabu_list) =
            self.optimizer
                .optimize(&model, Some(initial), 1000, Some(&callback), tabu_list);

        self.tabu_list = Some(tabu_list);

        println!(
            "best score: {}, permutation: {:?}",
            best_score, best_permutation
        );

        // apply the best permutation to the graph
        let shuffled_node_weights: Vec<Node<T, D, N>>;
        {
            // Limit the scope of the first mutable borrow here
            let node_weights = graph.node_weights().collect::<Vec<_>>();
            let shuffled_indices = (0..graph.node_count())
                .map(|i| best_permutation.apply_idx(i))
                .collect::<Vec<_>>();
            shuffled_node_weights = shuffled_indices
                .iter()
                .map(|&i| Node(node_weights[i].0.clone(), node_weights[i].1.clone()))
                .collect::<Vec<_>>();
            // shuffled_node_weights = best_permutation.apply(node_weights);
            // let shuffled_node_weights: Vec<Node<T, D, N>> = best_permutation.iter().map(|&i| node_weights[i]).collect();
        }

        // graph.node_indices()
        graph
            .node_weights_mut()
            .enumerate()
            .for_each(|(i, node_weight)| {
                node_weight.1 = shuffled_node_weights[i].1.clone();
            });
    }
}
