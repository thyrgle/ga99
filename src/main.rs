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

use core::array::from_fn;
use rayon::prelude::*;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use rand::RngCore;
use index_set::set::IntersectionIndices;
use index_set::IndexSet;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Datastructure used to represent the graph.
struct Graph<const N: usize> {
    adj: [IndexSet<usize>; N]
}


impl<const N: usize> Graph<N> {

    /// Initializes a graph with no edges and vertices.
    fn new() -> Self {
        let mut g: [IndexSet<usize>; N] 
            = from_fn(|_| IndexSet::<usize>::default());
        for entry in g.iter_mut() {
            *entry = IndexSet::<usize>::default();
        }
        Self {
            adj: g
        }
    }
    
    /// Adds the edge {i, j}.
    #[inline(always)]
    fn add_edge(&mut self, i: usize, j: usize) {
        let to_add = &mut self.adj;
        to_add[i].insert_index(j);
        to_add[j].insert_index(i);
    }
    
    /// Removes the edge {i, j}.
    #[inline(always)]
    fn remove_edge(&mut self, i: usize, j: usize) {
        let to_del = &mut self.adj;
        to_del[i].remove_index(j);
        to_del[j].remove_index(i);
    }
    
    /// Checks if {i, j} is an edge in the graph.
    #[inline(always)]
    fn contains_edge(&self, i: usize, j: usize) -> bool {
        let to_check = &self.adj;
        to_check[i].contains_index(j)
    }
    
    /// Retrieves the common neighbors that two vertices have.
    #[inline(always)]
    fn common_neighbors(&self, i: usize, j: usize) 
        -> IntersectionIndices<'_, usize> {
        let adj = &self.adj;
        adj[i].intersection_indices(&adj[j])
    }
}

/// Computes the score/fitness of the graph.
/// The ideal fitness is a value of 0, and the way it is scored is as follows:
/// If {i,j} is an edge, then i and j are neighbors, so they should be contained
/// in a unique triangle. Thus, they should have 1 neighbor in common.
/// Thus, 
/// If {i,j} is not an edge, then i and j are not neighbors, so they should be
/// contained in a unique square. Thus, they should have 2 neighbors in common.
///
/// Thus, the penalty is either (neighbors in common - 1)^2 or 
/// (neighbors in common - 2)^2 to see how far away from the correct number of
/// neighbors in common the two vertices really are.
#[inline(always)]
fn score<const N: usize>(g: &Graph<N>) -> usize {
    // Check all pairs of vertices in parallel.
    (0..N).into_par_iter()
        .map(|i| {
            let mut bad_count = 0;
            for j in i+1..N {
                let count = g.common_neighbors(i, j).count();
                let c = count as i32;
                if g.contains_edge(i, j) {
                    bad_count += ((c - 1) * (c - 1)) as usize
                    // Should have two neighbors in common.
                } else {
                    bad_count += ((c - 2) * (c - 2)) as usize
                }
            }
            bad_count
        }).sum()
    /*
    let mut bad_count = 0;
    for i in 0..N {
        for j in i+1..N {
            let count = g.common_neighbors(i, j).count();
            // Should have one neighbor in common.
            if g.contains_edge(i, j) {
                bad_count += ((count as i32 - 1) * (count as i32 - 1)) as usize
                // Should have two neighbors in common.
            } else {
                bad_count += ((count as i32 - 2) * (count as i32 - 2)) as usize
            }
        }
    }
    bad_count
    */
}

/// Randomly add/remove edge.
#[inline(always)]
fn improve_pair<const N: usize>(
    g: &mut Graph<N>, 
    rng: &mut Xoshiro256StarStar) -> ((usize, usize), bool) {
    let choice0 = (rng.next_u64() % (N as u64)) as usize;
    let choice1 = (rng.next_u64() % (N as u64)) as usize;
    if rng.next_u64() % 2 == 1 {
        g.add_edge(choice0, choice1);
        ((choice0, choice1), true)
    } else {
        g.remove_edge(choice0, choice1);
        ((choice0, choice1), false)
    }
}

/// Create an G(n, p) graph, that is a graph that has n vertices and edges are
/// added to the graph with probability p.
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

/// Called when the edge added/removed from improve_pair was bad, revert it by
/// removing/adding the edge.
#[inline(always)]
fn revert<const N:usize>(g: &mut Graph<N>, 
                         pair: (usize, usize),
                         add_or_del: bool) {
    // False indicates add, true indicates delete the edge.
    let (fst, snd) = pair;
    if add_or_del {
        g.remove_edge(fst, snd);
    } else {
        g.add_edge(fst, snd);
    }
}

fn main() {
    let mut rng = Xoshiro256StarStar::seed_from_u64(12345);
    // Number of vertices in the graph.
    const N: usize = 99;
    let mut g = gnp::<N>(14.0/99.0);
    // Higher fitness is worse, make a trivial upperbound that is essentially
    // infinity.
    let mut best_fitness: usize = 10000000;
    let mut prev_fitness: usize = 0;
    let mut fitness = 1;
    let mut penalty_factor = 0.5;
    const COOLING_FACTOR: f32 = 2.0;
    const HEATING_FACTOR: f32 = 0.0005;
    const PENALTY_UPPER: f32 = 2.0;
    // While the perfect graph has not been found.
    while fitness > 0 {
        fitness = score(&g);
        // Randomly add/remove edges in hopes of improving the graph.
        let (choice, add_or_del) = improve_pair(&mut g, &mut rng);
        let new_score = score(&g);
        // Keep track of the best fitness found so far.
        if best_fitness > fitness {
            best_fitness = fitness;
            println!("Fit {fitness:?} Penalty {penalty_factor:?}");
            // A great solution has been found! Reset the penalty factor.
        }
        // Add a slowly increasing penalty factor that "heats up" when no
        // Add a slowly increasing penalty factor that "heats up" when no
        // improvement has been found and "cools down" when a better solution
        // has been found.
        // Make sure penalty never goes negative, and keep it small (i.e. <= 3)
        if prev_fitness == fitness  {
            penalty_factor = f32::min(
                penalty_factor + HEATING_FACTOR, PENALTY_UPPER);
        } else {
            penalty_factor = f32::max(penalty_factor 
                                      - COOLING_FACTOR, 0.0);
        }
        // Check if the improvement was actually bad, and if so, revert the 
        // graph
        prev_fitness = fitness;
        if new_score > fitness + penalty_factor as usize {
            revert(&mut g, choice, add_or_del);
        }
    }
    println!("serialized = {:?}", g.adj);
}
