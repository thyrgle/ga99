/// A Simulated Annealing variant for solving Conway's 99 Problem.
/// Conway's 99 Problem: Is there a graph with 99 vertices such that each 
/// neighboring pair of vertices is contained in a unique triangle and each 
/// non-neighboring pair is contained in a unique square? 
/// For more information, see: 
/// https://en.wikipedia.org/wiki/Conway's_99-graph_problem
/// Simulated Annealing Variant outline:
/// 1. Start with a random graph.
/// 2. Randomly add/remove edges to improve the "fitness".
/// 3. If the new graph does not improve the fitness (+ a small slowly
///    increasing penalty factor) reject the candidate, otherwise accept the
///    candidate.
/// 4. Repeat until a candidate with "perfect" fitness is found.
/// The fitness of the graph is explained in the score function.
/// The "temperature" in the standard simulated annealing is simplified in this
/// variant. To see more about simulated annealing see:
/// https://en.wikipedia.org/wiki/Simulated_annealing

use std::{io, thread};
use std::time::Instant;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Datastructure used to represent the graph.
#[derive(Clone)]
struct Graph<const N: usize> {
    adj: [u128; N]
}


impl<const N: usize> Graph<N> {

    /// Initializes a graph with no edges and vertices.
    fn new() -> Self {
        let g: [u128; N] = [0; N];
        Self {
            adj: g
        }
    }
    
    /// Adds the edge {i, j}.
    #[inline(always)]
    fn add_edge(&mut self, i: usize, j: usize) {
        let to_add = &mut self.adj;
        to_add[i] |= 1 << j;
        to_add[j] |= 1 << i;
    }
    
    #[inline(always)]
    fn flip_edge(&mut self, i: usize, j: usize) {
        let to_flip = &mut self.adj;
        to_flip[i] ^= 1 << j;
        to_flip[j] ^= 1 << i;
    }
    
    /// Checks if {i, j} is an edge in the graph.
    #[inline(always)]
    fn contains_edge(&self, i: usize, j: usize) -> bool {
        let to_check = &self.adj;
        (to_check[i] & (1 << j)) > 0
    }
    
    /// Retrieves the common neighbors that two vertices have.
    #[inline(always)]
    fn common_neighbors(&self, i: usize, j: usize) 
        -> u128 {
        let adj = &self.adj;
        adj[i] & adj[j]
    }
}

/// Computes the score/fitness of the graph.
/// The ideal fitness is a value of 0, and the way it is scored is as 
/// follows:
/// If {i,j} is an edge, then i and j are neighbors, so they should be 
/// contained in a unique triangle. Thus, they should have 1 neighbor in 
/// common.
/// Thus, 
/// If {i,j} is not an edge, then i and j are not neighbors, so they 
/// should be contained in a unique square. Thus, they should have 2 
/// neighbors in common.
///
/// Thus, the penalty is either (neighbors in common - 1)^2 or 
/// (neighbors in common - 2)^2 to see how far away from the correct 
/// number of neighbors in common the two vertices really are.
#[inline(always)]
fn score<const N: usize>(g: &Graph<N>) -> usize {
    // Check all pairs of vertices in parallel.
    /*
    (0..N).into_par_iter()
        .map(|i| {
            let mut bad_count = 0;
            for j in i+1..N {
                let count = g.common_neighbors(i, j).count_ones();
                let c = count as i32;
                let e = g.contains_edge(j, i) as i32;
                bad_count += ((c - (2 - e)) * (c - (2 - e))) as usize
            }
            bad_count
        }).sum()
    */
    let mut bad_count = 0;
    for i in 0..N {
        for j in i+1..N {
            let count = g.common_neighbors(i, j).count_ones();
            let c = count as i32;
            let e = g.contains_edge(i, j) as i32;
            bad_count += ((c - (2 - e)) * (c - (2 - e))) as usize;
        }
    }
    bad_count
}

/// Randomly add/remove edge.
#[inline(always)]
fn improve_pair<const N: usize>(
    g: &mut Graph<N>) -> (usize, usize) {
    let choice0 = rand::random::<usize>() % N;
    let choice1 = rand::random::<usize>() % N;
    g.flip_edge(choice0, choice1);
    (choice0, choice1)
}

/// Create an G(n, p) graph, that is a graph that has n vertices and edges
/// are added to the graph with probability p.
fn gnp<const N: usize>(p: f32) -> Graph<N> {
    // Although there may be room for parallelization, this function is only
    // called once and takes a negligible time in the long run.
    let mut g = Graph::<N>::new();
    for i in 0..N {
        for j in i+1..N {
            if rand::random::<f32>() < p {
                g.add_edge(i, j);
            }
        }
    }
    g
}

/// Called when the edge added/removed from improve_pair was bad, revert 
/// it by removing/adding the edge.
#[inline(always)]
fn revert<const N:usize>(g: &mut Graph<N>, 
                         pair: (usize, usize)) {
    // False indicates add, true indicates delete the edge.
    let (fst, snd) = pair;
    g.flip_edge(fst, snd);
}

fn main() -> io::Result<()> {
    const N: usize = 99;
    let mut best_graph = gnp::<N>(14.0/99.0);
    let mut best_fitness = usize::MAX;
    while best_fitness > 0 {
        let mut handlers = Vec::new();
        for _ in 0..thread::available_parallelism()?.get() {
            let mut g = best_graph.clone();
            handlers.push(thread::spawn(move || {
                let now = Instant::now();
                let mut prev_fitness: usize = 0;
                let mut fitness = usize::MAX;
                let mut penalty_factor = 0.5;
                let mut skip_score = false;
                const COOLING_FACTOR: f32 = 5.0;
                const HEATING_FACTOR: f32 = 0.01;
                const PENALTY_UPPER: f32 = 10.0;
                // While the perfect graph has not been found.
                while fitness >= best_fitness && now.elapsed().as_secs() < 120 {
                    if !skip_score {
                        fitness = score(&g);
                        skip_score = true;
                    }
                    // Randomly add/remove edges in hopes of improving the
                    // graph.
                    let choice = improve_pair(&mut g);
                    let new_score = score(&g);
                    // Keep track of the best fitness found so far.
                    if best_fitness > fitness {
                        best_fitness = fitness;
                    }
                    // Add a slowly increasing penalty factor that "heats
                    // up" when no improvement has been found and "cools
                    // down" when a better solution has been found.
                    // Make sure penalty never goes negative, and keep it
                    // small (i.e. <= 10).
                    if prev_fitness == fitness  {
                        penalty_factor = f32::min(
                            penalty_factor + HEATING_FACTOR, PENALTY_UPPER
                        );
                    } else {
                        penalty_factor = f32::max(
                            penalty_factor - COOLING_FACTOR, 0.0
                        );
                    }
                    // Check if the improvement was actually bad, and if 
                    // so, revert the graph
                    prev_fitness = fitness;
                    if new_score > fitness + penalty_factor as usize {
                        revert(&mut g, choice);
                        skip_score = false;
                    }
                }
                (g, fitness)
            }));
        }
        let mut best_graphs: Vec<(Graph<N>, usize)> = Vec::new();
        for handler in handlers {
            best_graphs.push(handler.join().unwrap());
        }
        let mut new_fit = usize::MAX;
        for (g, fit) in best_graphs {
            println!("{fit:?}");
            if fit < new_fit {
                best_graph = g;
                new_fit = fit;
            }
        }
        if new_fit < best_fitness {
            best_fitness = new_fit;
        }
        println!("Fit {best_fitness:?}");
    }

    println!("serialized = {:?}", best_graph.adj);
    Ok(())
}
