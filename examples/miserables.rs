
use cobwebs_2024::{crossings, crossings_with_permutation, Swapper};
use permutation::Permutation;
use ::rand::distributions::Uniform;
use fdg_sim::{
    petgraph::Graph, Center, Force, ForceGraph, FruchtermanReingold,
    FruchtermanReingoldConfiguration, Node, Translate,
};
use macroquad::prelude::*;
use nalgebra::vector;

use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;

#[derive(Debug, Deserialize)]
struct NodeInfo {
    id: String,
    // group: u32,
    // order: u32,
}

#[derive(Debug, Deserialize)]
struct LinkInfo {
    source: String,
    target: String,
}

#[derive(Debug, Deserialize)]
struct GraphData {
    nodes: Vec<NodeInfo>,
    links: Vec<LinkInfo>,
}

#[macroquad::main("fdg demo")]
async fn main() {
    let file = File::open("examples/miserables.json").unwrap();
    let reader = BufReader::new(file);

    let graph_data: GraphData = serde_json::from_reader(reader).unwrap();

    let mut graph = Graph::<String, String>::new();
    let mut nodes = HashMap::new();

    for node in graph_data.nodes {
        nodes.insert(node.id.clone(), graph.add_node(node.id));
    }

    for link in graph_data.links {
        let source = nodes.get(&link.source).unwrap();
        let target = nodes.get(&link.target).unwrap();
        graph.add_edge(*source, *target, "".to_string());
    }

    let mut force_graph: ForceGraph<f32, 2, String, String> =
        fdg_sim::init_force_graph(graph, Uniform::new(-200.0, 200.0));

    let mut center = Center::default();
    let mut translate = Translate::new(vector![0.0, -100.0]);
    let mut force = FruchtermanReingold {
        conf: FruchtermanReingoldConfiguration {
            scale: 400.0,
            cooloff_factor: 0.5,

            ..Default::default()
        },
        ..Default::default()
    };

    let mut swapper = Swapper::new(3, 5, 30, 10);

    let mut i = 0;
    let skip = 2;

    loop {
        // apply the fruchterman-reingold force 4 times
        for _ in 0..10 {
            force.apply(&mut force_graph);
        }

        if i % skip == 1 {
            swapper.apply(&mut force_graph);
        }
        i += 1;

        // let crossings = crossings(&force_graph).unwrap();
        let permutation = Permutation::one(force_graph.node_count());
        let crossings = crossings_with_permutation(&force_graph, &permutation).unwrap();

        // move the graph mean position to 0,0
        center.apply(&mut force_graph);

        // translate the whole graph up 100 units
        translate.apply(&mut force_graph);

        clear_background(WHITE);

        for idx in force_graph.edge_indices() {
            let (Node(_, source), Node(_, target)) = force_graph
                .edge_endpoints(idx)
                .map(|(a, b)| {
                    (
                        force_graph.node_weight(a).unwrap(),
                        force_graph.node_weight(b).unwrap(),
                    )
                })
                .unwrap();

            draw_line(
                translate_x(source.coords.column(0)[0]),
                translate_y(source.coords.column(0)[1]),
                translate_x(target.coords.column(0)[0]),
                translate_y(target.coords.column(0)[1]),
                4.0,
                BLACK,
            );
        }

        for Node(name, pos) in force_graph.node_weights() {
            let x = translate_x(pos.coords.column(0)[0]);
            let y = translate_y(pos.coords.column(0)[1]);

            draw_circle(x, y, 20.0, RED);
            draw_text(name, x - 30.0, y - 30.0, 40.0, BLACK);
        }

        draw_text(&format!("crossings: {}", crossings), 50.0, 50.0, 40.0, BLACK);

        next_frame().await
    }
}

fn translate_x(x: f32) -> f32 {
    (screen_width() / 2.0) + x
}

fn translate_y(y: f32) -> f32 {
    (screen_height() / 2.0) + y
}
