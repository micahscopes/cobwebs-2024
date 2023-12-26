// use fdg_sim::petgraph::stable_graph::NodeIndex;
use fdg_sim::{Field, Force, ForceGraph, Node};
use localsearch::optim::{TabuList, TabuSearchOptimizer, LocalSearchOptimizer};
use localsearch::OptProgress;
use localsearch::OptModel;
use permutation::Permutation;

use crate::crossings_with_permutation;

struct GraphModel<'a, T: Field, const D: usize, N: Clone, E: Send> {
    graph: &'a ForceGraph<T, D, N, E>,
    // permutation: Permutation,
    // other fields as needed
}

type SolutionType = Permutation;
type ScoreType = usize;
type TransitionType = ();

impl<'a, T: Field, const D: usize, N: Send + Sync + Clone, E: Send + Sync> OptModel
    for GraphModel<'a, T, D, N, E>
where
    f64: From<T>,
{
    type SolutionType = SolutionType;
    type TransitionType = TransitionType;
    type ScoreType = ScoreType;

    fn evaluate_solution(&self, state: &Self::SolutionType) -> Self::ScoreType {
        let score = crossings_with_permutation(&self.graph, state).unwrap();
        println!("score: {}, permutation: {:?}", score, state);        
        score
    }

    fn generate_trial_solution<R: rand::Rng>(
        &self,
        current_state: &Self::SolutionType,
        rng: &mut R,
        current_score: Option<Self::ScoreType>,
    ) -> (Self::SolutionType, Self::TransitionType, Self::ScoreType) {
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
        let new_solution = &Permutation::oneline(idx).inverse() * current_state;
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


#[derive(Debug)]
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
    // other fields as needed
}

impl Swapper {
    pub fn new(neighborhood_size: usize) -> Self {
        // let optimizer = TabuSearchOptimizer::new(100, 100, 1);
        // let tabu_list = PermutationTabuList::new(20);
        let optimizer = TabuSearchOptimizer::<PermutationTabuList>::new(20, 200, 10);
        Self {
            neighborhood_size,
            optimizer,
            // initialize other fields as needed
        }
    }
}

impl<T: Field, const D: usize, N: Send + Sync + Clone, E: Send + Sync>
    Force<T, D, N, E> for Swapper
where
    f64: From<T>,
{
    fn apply(&mut self, graph: &mut ForceGraph<T, D, N, E>) {
        let model = GraphModel { graph: graph };
        let callback = |op: OptProgress<SolutionType, ScoreType>| {
            println!("progress {:?}", op);
            // pb.set_message(format!("best score {:e}", op.score.into_inner()));
            // pb.set_position(op.iter as u64);
        };

        let initial = Permutation::one(graph.node_count());

        let tabu_list = PermutationTabuList::new(2000);

        let (best_permutation, best_score, _) =
            self.optimizer.optimize(&model, Some(initial), 1000, Some(&callback), tabu_list);

        println!("best score: {}, permutation: {:?}", best_score, best_permutation);

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
                .map(|&i| node_weights[i].clone())
                .collect::<Vec<_>>();
            // shuffled_node_weights = best_permutation.apply(node_weights);
            // let shuffled_node_weights: Vec<Node<T, D, N>> = best_permutation.iter().map(|&i| node_weights[i]).collect();
        }

        // graph.node_indices()
        graph
            .node_weights_mut()
            .enumerate()
            .for_each(|(i, node_weight)| {
                *node_weight = shuffled_node_weights[i].clone();
            });
    }
}
