use fdg_sim::petgraph::stable_graph::NodeIndex;
use fdg_sim::{Field, Force, ForceGraph, Node};
use localsearch::OptProgress;
use localsearch::optim::LocalSearchOptimizer;
use localsearch::{optim::HillClimbingOptimizer, OptModel};
use permutation::Permutation;

use crate::crossings_with_permutation;

struct GraphModel<'a, T: Field, const D: usize, N: Clone, E: Send> {
    graph: &'a ForceGraph<T, D, N, E>,
    // permutation: Permutation,
    // other fields as needed
}

use shuffle::irs::Irs;
use shuffle::shuffler::Shuffler;
// use rand::rngs::mock::StepRng;

// let mut rng = StepRng::new(2, 13);

type StateType = Permutation;
type ScoreType = usize;
type TransitionType = ();

impl<'a, T: Field, const D: usize, N: Send + Sync + Clone, E: Send + Sync> OptModel for GraphModel<'a, T, D, N, E>
where
    f64: From<T>,
{
    type StateType = StateType;
    type TransitionType = TransitionType;
    type ScoreType = ScoreType;

    fn evaluate_state(&self, state: &Self::StateType) -> Self::ScoreType {
        crossings_with_permutation(&self.graph, state).unwrap()
    }

    fn generate_trial_state<R: rand::Rng>(
        &self,
        current_state: &Self::StateType,
        rng: &mut R,
        current_score: Option<Self::ScoreType>,
    ) -> (Self::StateType, Self::TransitionType, Self::ScoreType) {
        let mut idx = Vec::from_iter(0..self.graph.node_count());
        let a = rng.gen_range(0..idx.len());
        let mut b = 0;
        for _ in 0..1000 {
            b = rng.gen_range(0..idx.len());
            if a != b {
                // idx.swap(a, b);
                break;
            }
        }
        idx.swap(a, b);
        (current_state * &Permutation::oneline(idx).inverse(), (), 0)
    }

    fn generate_random_state<R: rand::Rng>(
        &self,
        rng: &mut R,
    ) -> Result<Self::StateType, Box<dyn std::error::Error>> {
        let mut irs = Irs::default();
        let mut idx = Vec::from_iter(0..self.graph.node_count());
        irs.shuffle(&mut idx, rng);
        Ok(Permutation::oneline(idx).inverse())
    }
}

#[derive(Copy, Clone)]
pub struct Swapper {
    neighborhood_size: usize,
    optimizer: HillClimbingOptimizer,
    // other fields as needed
}

impl Swapper {
    pub fn new(neighborhood_size: usize) -> Self {
        let optimizer = HillClimbingOptimizer::new(100, 100);
        Self {
            neighborhood_size,
            optimizer,
            // initialize other fields as needed
        }
    }
}

impl<T: Field, const D: usize, N: Send + Sync + Clone, E: Send + Sync> Force<T, D, N, E> for Swapper
where
    f64: From<T>,
{
    fn apply(&mut self, graph: &mut ForceGraph<T, D, N, E>) {
        let model = GraphModel { graph: graph };
        let callback = |op: OptProgress<StateType, ScoreType>| {
            // pb.set_message(format!("best score {:e}", op.score.into_inner()));
            // pb.set_position(op.iter as u64);
        };
    
        let (best_permutation, best_score, ()) =
            self.optimizer.optimize(&model, None, 100, Some(&callback), ());
    
        // apply the best permutation to the graph
        let shuffled_node_weights: Vec<Node<T, D, N>>;
        {
            // Limit the scope of the first mutable borrow here
            let node_weights = graph.node_weights().collect::<Vec<_>>();
            let shuffled_indices = (0..graph.node_count()).map(|i| best_permutation.apply_idx(i)).collect::<Vec<_>>();
            shuffled_node_weights = shuffled_indices.iter().map(|&i| node_weights[i].clone()).collect::<Vec<_>>();
            // shuffled_node_weights = best_permutation.apply(node_weights);
// let shuffled_node_weights: Vec<Node<T, D, N>> = best_permutation.iter().map(|&i| node_weights[i]).collect();
        }
    
        // Now that the first mutable borrow has ended, you can borrow graph as mutable again
        graph.node_weights_mut().enumerate().for_each(|(i, node_weight)| {
            *node_weight = shuffled_node_weights[i].clone();
        });
    }
}
