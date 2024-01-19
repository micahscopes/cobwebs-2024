mod crossings;
mod swapper;
// mod grid_wanderer;

pub use crossings::*;
pub use swapper::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        // let result = add(2, 2);
        // assert_eq!(result, 4);
    }
}
