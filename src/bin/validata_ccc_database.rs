//! Validate model-derived spatial influence edges against an optional CCC database.
//! Database evidence is orthogonal annotation and never defines model relationships.
use csv::{ReaderBuilder, WriterBuilder};
use rand::{Rng, SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use serde::Deserialize;
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    expression_mtx: PathBuf,
    cell_ids: PathBuf,
    gene_ids: PathBuf,
    coordinates: PathBuf,
    model_edges: PathBuf,
    database_relations: PathBuf,
    output_dir: PathBuf,
    maximum_distance: f64,
    #[serde(default = "tol")]
    distance_tolerance_fraction: f64,
    #[serde(default)]
    expression_threshold: f32,
    #[serde(default = "seed")]
    random_seed: u64,
    #[serde(default = "yes")]
    write_edge_support: bool,
}
fn tol() -> f64 {
    0.1
}
fn seed() -> u64 {
    20_260_812
}
fn yes() -> bool {
    true
}
#[derive(Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
struct Key {
    cluster: usize,
    source: String,
    target: String,
}
struct Edge {
    source: usize,
    receiver: usize,
    distance: f64,
    source_id: String,
    receiver_id: String,
}
struct Relation {
    ligand: Vec<usize>,
    receptor: Vec<usize>,
    label: String,
    databases: String,
    ligand_names: Vec<String>,
    receptor_names: Vec<String>,
}
struct Expression {
    positive: Vec<Vec<usize>>,
}
impl Expression {
    #[inline]
    fn has(&self, c: usize, g: usize) -> bool {
        self.positive[c].binary_search(&g).is_ok()
    }
}
struct Block {
    key: Key,
    model_edges: usize,
    background_edges: usize,
    model: Vec<usize>,
    background: Vec<usize>,
    hits: Vec<(String, String, usize)>,
}

fn main() -> Result<(), String> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: validata_ccc_database CONFIG.json")?;
    let c: Config = serde_json::from_str(&fs::read_to_string(path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    if c.maximum_distance <= 0.0 || !c.maximum_distance.is_finite() {
        return Err("maximum_distance must be finite and > 0".into());
    }
    fs::create_dir_all(&c.output_dir).map_err(|e| e.to_string())?;
    let cells = cell_attention::data::read_ids(&c.cell_ids)?;
    let genes = cell_attention::data::read_ids(&c.gene_ids)?;
    let ci: HashMap<&str, usize> = cells
        .iter()
        .enumerate()
        .map(|(i, x)| (x.as_str(), i))
        .collect();
    let gi: HashMap<&str, usize> = genes
        .iter()
        .enumerate()
        .map(|(i, x)| (x.as_str(), i))
        .collect();
    let xy = read_coordinates(&c.coordinates, cells.len())?;
    let edges = read_edges(&c.model_edges, &ci)?;
    let (rel, mut local) = read_relations(&c.database_relations, &gi)?;
    // Background matching requires every model source gene, including genes that
    // never occur as a ligand or receptor in the optional database.
    for key in edges.keys() {
        let raw = *gi
            .get(key.source.as_str())
            .ok_or_else(|| format!("source gene {} absent", key.source))?;
        if !local.contains_key(&raw) {
            let next = local.len();
            local.insert(raw, next);
        }
    }
    eprintln!(
        "stage=expression cells={} genes={} measurable_lr={}",
        cells.len(),
        genes.len(),
        rel.len()
    );
    let x = read_expression(
        &c.expression_mtx,
        cells.len(),
        genes.len(),
        &local,
        c.expression_threshold,
    )?;
    let inv = inverted(&rel, local.len());
    let tasks: Vec<_> = edges.into_iter().collect();
    let blocks: Result<Vec<_>, String> = tasks
        .par_iter()
        .map(|(k, e)| analyze(k, e, &rel, &inv, &x, &xy, &gi, &local, &c))
        .collect();
    let mut blocks = blocks?;
    blocks.sort_by(|a, b| a.key.cmp(&b.key));
    write_results(&c, &rel, &blocks)?;
    eprintln!(
        "stage=complete triplets={} measurable_lr={}",
        blocks.len(),
        rel.len()
    );
    Ok(())
}
fn field<'a>(h: &'a csv::StringRecord, n: &str) -> Result<usize, String> {
    h.iter()
        .position(|x| x == n)
        .ok_or_else(|| format!("missing column {n}"))
}
fn read_coordinates(p: &Path, n: usize) -> Result<Vec<[f64; 2]>, String> {
    let mut out = Vec::new();
    for line in BufReader::new(fs::File::open(p).map_err(|e| e.to_string())?).lines() {
        let line = line.map_err(|e| e.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        let v: Vec<f64> = line
            .split(|c: char| c == ',' || c == '\t' || c.is_whitespace())
            .filter(|x| !x.is_empty())
            .map(str::parse)
            .collect::<Result<_, _>>()
            .map_err(|e| format!("coordinate: {e}"))?;
        if v.len() < 2 {
            return Err("coordinates require two columns".into());
        }
        out.push([v[0], v[1]])
    }
    if out.len() != n {
        return Err(format!("coordinate rows {} != cells {n}", out.len()));
    }
    Ok(out)
}
fn read_edges(p: &Path, ci: &HashMap<&str, usize>) -> Result<BTreeMap<Key, Vec<Edge>>, String> {
    let mut r = ReaderBuilder::new()
        .from_path(p)
        .map_err(|e| e.to_string())?;
    let h = r.headers().map_err(|e| e.to_string())?.clone();
    let (cc, sg, tg, sc, rc, dc) = (
        field(&h, "cluster")?,
        field(&h, "source_gene")?,
        field(&h, "target_gene")?,
        field(&h, "sender_cell_id")?,
        field(&h, "receiver_cell_id")?,
        field(&h, "distance")?,
    );
    let mut out: BTreeMap<Key, Vec<Edge>> = BTreeMap::new();
    for row in r.records() {
        let row = row.map_err(|e| e.to_string())?;
        let sid = &row[sc];
        let rid = &row[rc];
        out.entry(Key {
            cluster: row[cc].parse::<usize>().map_err(|e| e.to_string())?,
            source: row[sg].into(),
            target: row[tg].into(),
        })
        .or_default()
        .push(Edge {
            source: *ci.get(sid).ok_or_else(|| format!("unknown cell {sid}"))?,
            receiver: *ci.get(rid).ok_or_else(|| format!("unknown cell {rid}"))?,
            distance: row[dc].parse::<f64>().map_err(|e| e.to_string())?,
            source_id: sid.into(),
            receiver_id: rid.into(),
        })
    }
    Ok(out)
}
fn split(x: &str) -> Vec<String> {
    x.split(';')
        .map(str::trim)
        .filter(|x| !x.is_empty())
        .map(str::to_string)
        .collect()
}
fn read_relations(
    p: &Path,
    gi: &HashMap<&str, usize>,
) -> Result<(Vec<Relation>, HashMap<usize, usize>), String> {
    let mut r = ReaderBuilder::new()
        .from_path(p)
        .map_err(|e| e.to_string())?;
    let h = r.headers().map_err(|e| e.to_string())?.clone();
    let (d, l, q) = (
        field(&h, "database")?,
        field(&h, "ligand_components")?,
        field(&h, "receptor_components")?,
    );
    let mut unique: BTreeMap<(Vec<String>, Vec<String>), HashSet<String>> = BTreeMap::new();
    for row in r.records() {
        let row = row.map_err(|e| e.to_string())?;
        let (a, b) = (split(&row[l]), split(&row[q]));
        if !a.is_empty() && !b.is_empty() && a.iter().chain(&b).all(|g| gi.contains_key(g.as_str()))
        {
            unique.entry((a, b)).or_default().insert(row[d].to_string());
        }
    }
    let needed: HashSet<usize> = unique
        .keys()
        .flat_map(|(a, b)| a.iter().chain(b))
        .map(|g| gi[g.as_str()])
        .collect();
    let mut ids: Vec<_> = needed.into_iter().collect();
    ids.sort_unstable();
    let local: HashMap<usize, usize> = ids.into_iter().enumerate().map(|(l, r)| (r, l)).collect();
    let rel = unique
        .into_iter()
        .map(|((a, b), db)| {
            let mut db: Vec<_> = db.into_iter().collect();
            db.sort();
            Relation {
                label: format!("{} -> {}", a.join("+"), b.join("+")),
                ligand: a.iter().map(|g| local[&gi[g.as_str()]]).collect(),
                receptor: b.iter().map(|g| local[&gi[g.as_str()]]).collect(),
                databases: db.join(";"),
                ligand_names: a,
                receptor_names: b,
            }
        })
        .collect();
    Ok((rel, local))
}
fn read_expression(
    p: &Path,
    nc: usize,
    ng: usize,
    relevant: &HashMap<usize, usize>,
    threshold: f32,
) -> Result<Expression, String> {
    let mut lines = BufReader::new(fs::File::open(p).map_err(|e| e.to_string())?).lines();
    let dimensions = loop {
        let z = lines
            .next()
            .ok_or("missing MatrixMarket dimensions")?
            .map_err(|e| e.to_string())?;
        if !z.starts_with('%') && !z.trim().is_empty() {
            break z;
        }
    };
    let d: Vec<usize> = dimensions
        .split_whitespace()
        .map(str::parse::<usize>)
        .collect::<Result<_, _>>()
        .map_err(|e| e.to_string())?;
    if d.len() != 3 {
        return Err("invalid MatrixMarket dimensions".into());
    }
    let tr = d[0] == ng && d[1] == nc;
    if !tr && (d[0] != nc || d[1] != ng) {
        return Err(format!("matrix {}x{} != {nc}x{ng}", d[0], d[1]));
    }
    let mut positive = vec![Vec::new(); nc];
    for line in lines {
        let line = line.map_err(|e| e.to_string())?;
        let f: Vec<_> = line.split_whitespace().collect();
        if f.len() != 3 {
            continue;
        }
        let (a, b, v) = (
            f[0].parse::<usize>().map_err(|e| e.to_string())? - 1,
            f[1].parse::<usize>().map_err(|e| e.to_string())? - 1,
            f[2].parse::<f32>().map_err(|e| e.to_string())?,
        );
        if v <= threshold {
            continue;
        }
        let (c, g) = if tr { (b, a) } else { (a, b) };
        if let Some(&g) = relevant.get(&g) {
            positive[c].push(g)
        }
    }
    positive.par_iter_mut().for_each(|v| {
        v.sort_unstable();
        v.dedup()
    });
    Ok(Expression { positive })
}
fn inverted(r: &[Relation], n: usize) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::new(); n];
    for (i, x) in r.iter().enumerate() {
        for &g in &x.ligand {
            out[g].push(i)
        }
    }
    out
}
#[inline]
fn supported(s: usize, t: usize, r: &[Relation], inv: &[Vec<usize>], x: &Expression) -> Vec<usize> {
    let mut candidates = Vec::new();
    for &g in &x.positive[s] {
        candidates.extend_from_slice(&inv[g])
    }
    candidates.sort_unstable();
    candidates.dedup();
    candidates
        .into_iter()
        .filter(|&i| {
            r[i].ligand.iter().all(|&g| x.has(s, g)) && r[i].receptor.iter().all(|&g| x.has(t, g))
        })
        .collect()
}
struct Grid {
    bins: HashMap<[i64; 2], Vec<usize>>,
    width: f64,
}
impl Grid {
    fn new(cells: impl Iterator<Item = usize>, xy: &[[f64; 2]], width: f64) -> Self {
        let mut bins: HashMap<[i64; 2], Vec<usize>> = HashMap::new();
        for c in cells {
            bins.entry([
                (xy[c][0] / width).floor() as i64,
                (xy[c][1] / width).floor() as i64,
            ])
            .or_default()
            .push(c)
        }
        Self { bins, width }
    }
    fn candidates(&self, r: usize, xy: &[[f64; 2]]) -> Vec<usize> {
        let k = [
            (xy[r][0] / self.width).floor() as i64,
            (xy[r][1] / self.width).floor() as i64,
        ];
        let mut out = Vec::new();
        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(v) = self.bins.get(&[k[0] + dx, k[1] + dy]) {
                    out.extend_from_slice(v)
                }
            }
        }
        out
    }
}
fn hash(x: &str) -> u64 {
    x.bytes().fold(1469598103934665603, |h, b| {
        (h ^ b as u64).wrapping_mul(1099511628211)
    })
}
fn analyze(
    k: &Key,
    edges: &[Edge],
    r: &[Relation],
    inv: &[Vec<usize>],
    x: &Expression,
    xy: &[[f64; 2]],
    gi: &HashMap<&str, usize>,
    local: &HashMap<usize, usize>,
    c: &Config,
) -> Result<Block, String> {
    let raw = *gi
        .get(k.source.as_str())
        .ok_or_else(|| format!("source gene {} absent", k.source))?;
    let source_local = local.get(&raw).copied();
    let grid = Grid::new(
        (0..x.positive.len()).filter(|&i| source_local.is_some_and(|g| x.has(i, g))),
        xy,
        c.maximum_distance,
    );
    let mut rng = StdRng::seed_from_u64(
        c.random_seed
            ^ (k.cluster as u64).wrapping_mul(0x9E3779B97F4A7C15)
            ^ hash(&k.source)
            ^ hash(&k.target),
    );
    let (mut model, mut background) = (vec![0; r.len()], vec![0; r.len()]);
    let mut hits = Vec::new();
    let mut bn = 0;
    let tolerance = c.maximum_distance * c.distance_tolerance_fraction;
    for edge in edges {
        for i in supported(edge.source, edge.receiver, r, inv, x) {
            model[i] += 1;
            if c.write_edge_support {
                hits.push((edge.source_id.clone(), edge.receiver_id.clone(), i))
            }
        }
        let mut pool = grid.candidates(edge.receiver, xy);
        pool.retain(|&s| {
            if s == edge.receiver {
                return false;
            }
            let dx = xy[s][0] - xy[edge.receiver][0];
            let dy = xy[s][1] - xy[edge.receiver][1];
            let distance = (dx * dx + dy * dy).sqrt();
            distance <= c.maximum_distance && (distance - edge.distance).abs() <= tolerance
        });
        if !pool.is_empty() {
            let s = pool[rng.random_range(0..pool.len())];
            for i in supported(s, edge.receiver, r, inv, x) {
                background[i] += 1
            }
            bn += 1
        }
    }
    if bn == 0 {
        return Err(format!(
            "no distance-matched background edges for cluster {}: {} -> {}",
            k.cluster, k.source, k.target
        ));
    }
    Ok(Block {
        key: k.clone(),
        model_edges: edges.len(),
        background_edges: bn,
        model,
        background,
        hits,
    })
}
fn log_choose(n: usize, k: usize) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    let k = k.min(n - k);
    (1..=k)
        .map(|i| ((n - k + i) as f64).ln() - (i as f64).ln())
        .sum()
}
fn hypergeom(a: usize, row: usize, col: usize, total: usize) -> f64 {
    (log_choose(col, a) + log_choose(total - col, row - a) - log_choose(total, row)).exp()
}
fn fisher(a: usize, b: usize, c: usize, d: usize) -> f64 {
    let row = a + b;
    let col = a + c;
    let total = row + c + d;
    let lo = row.saturating_sub(total - col);
    let hi = row.min(col);
    let observed = hypergeom(a, row, col, total);
    (lo..=hi)
        .map(|v| hypergeom(v, row, col, total))
        .filter(|&p| p <= observed * (1.0 + 1e-10))
        .sum::<f64>()
        .min(1.0)
}
fn bh(p: &[f64]) -> Vec<f64> {
    let mut order: Vec<_> = (0..p.len()).collect();
    order.sort_by(|&a, &b| p[a].total_cmp(&p[b]));
    let mut q = vec![1.0; p.len()];
    let mut running = 1.0f64;
    for (rank, &i) in order.iter().enumerate().rev() {
        running = running.min(p[i] * p.len() as f64 / (rank + 1) as f64);
        q[i] = running.min(1.0)
    }
    q
}
fn write_results(c: &Config, r: &[Relation], blocks: &[Block]) -> Result<(), String> {
    let mut w = WriterBuilder::new()
        .from_path(c.output_dir.join("lr_distance_matched_enrichment_long.csv"))
        .map_err(|e| e.to_string())?;
    w.write_record([
        "cluster",
        "source_gene",
        "target_gene",
        "lr_pair",
        "databases",
        "model_supported_edges",
        "model_edges",
        "model_fraction",
        "background_supported_edges",
        "background_edges",
        "background_fraction",
        "log2_odds_ratio",
        "fisher_p",
        "fdr_q_within_triplet",
        "direct_model_pair_match",
    ])
    .map_err(|e| e.to_string())?;
    for b in blocks {
        let p: Vec<_> = (0..r.len())
            .map(|i| {
                fisher(
                    b.model[i],
                    b.model_edges - b.model[i],
                    b.background[i],
                    b.background_edges - b.background[i],
                )
            })
            .collect();
        let q = bh(&p);
        for (i, x) in r.iter().enumerate() {
            let (m, bg) = (b.model[i], b.background[i]);
            if m == 0 && bg == 0 {
                continue;
            }
            let odds = ((m as f64 + 0.5) / ((b.model_edges - m) as f64 + 0.5))
                / ((bg as f64 + 0.5) / ((b.background_edges - bg) as f64 + 0.5));
            let direct = x.ligand_names.len() == 1
                && x.receptor_names.len() == 1
                && x.ligand_names[0] == b.key.source
                && x.receptor_names[0] == b.key.target;
            w.write_record([
                b.key.cluster.to_string(),
                b.key.source.clone(),
                b.key.target.clone(),
                x.label.clone(),
                x.databases.clone(),
                m.to_string(),
                b.model_edges.to_string(),
                format!("{:.8}", m as f64 / b.model_edges.max(1) as f64),
                bg.to_string(),
                b.background_edges.to_string(),
                format!("{:.8}", bg as f64 / b.background_edges.max(1) as f64),
                format!("{:.8}", odds.log2()),
                format!("{:.8}", p[i]),
                format!("{:.8}", q[i]),
                direct.to_string(),
            ])
            .map_err(|e| e.to_string())?
        }
    }
    w.flush().map_err(|e| e.to_string())?;
    if c.write_edge_support {
        let mut w = WriterBuilder::new()
            .from_path(c.output_dir.join("model_edge_lr_support.csv"))
            .map_err(|e| e.to_string())?;
        w.write_record([
            "cluster",
            "source_gene",
            "target_gene",
            "sender_cell_id",
            "receiver_cell_id",
            "lr_pair",
            "databases",
        ])
        .map_err(|e| e.to_string())?;
        for b in blocks {
            for (s, t, i) in &b.hits {
                w.write_record([
                    b.key.cluster.to_string(),
                    b.key.source.clone(),
                    b.key.target.clone(),
                    s.clone(),
                    t.clone(),
                    r[*i].label.clone(),
                    r[*i].databases.clone(),
                ])
                .map_err(|e| e.to_string())?
            }
        }
        w.flush().map_err(|e| e.to_string())?
    }
    fs::write(c.output_dir.join("metadata.json"),serde_json::to_string_pretty(&serde_json::json!({"program":"validata_ccc_database","analysis":"optional CCC database validation of model-derived spatial influence edges","measurable_unique_lr_pairs":r.len(),"triplets":blocks.len(),"expression_threshold":c.expression_threshold,"maximum_distance":c.maximum_distance,"distance_tolerance_fraction":c.distance_tolerance_fraction,"random_seed":c.random_seed,"interpretation":"orthogonal biological plausibility annotation; does not define or alter model relationships"})).unwrap()).map_err(|e|e.to_string())?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fisher_is_symmetric() {
        let a = fisher(8, 2, 1, 9);
        assert!(a < 0.01);
        assert!((a - fisher(1, 9, 8, 2)).abs() < 1e-12)
    }
}
