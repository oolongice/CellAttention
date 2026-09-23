use ndarray::Array2;
use std::{
    fs,
    io::{BufWriter, Write},
    path::Path,
};

#[derive(Clone, Debug)]
pub struct ExpressionData {
    pub raw_matrix: Array2<f32>,
    pub matrix: Array2<f32>,
    pub cell_ids: Vec<String>,
    pub gene_ids: Vec<String>,
    pub means: Vec<f32>,
    pub standard_deviations: Vec<f32>,
}

pub fn read_ids(path: impl AsRef<Path>) -> Result<Vec<String>, String> {
    let text = fs::read_to_string(path.as_ref())
        .map_err(|error| format!("failed to read ID file: {error}"))?;
    let ids: Vec<String> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(|line| line.split([',', '\t']).next().unwrap().to_string())
        .collect();
    if ids.is_empty() {
        return Err("ID file is empty".to_string());
    }
    Ok(ids)
}

pub fn read_expression(
    path: impl AsRef<Path>,
    num_cells: usize,
    num_genes: usize,
) -> Result<Array2<f32>, String> {
    let path = path.as_ref();
    let text = fs::read_to_string(path)
        .map_err(|error| format!("failed to read expression matrix: {error}"))?;
    let first = text
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("");
    if path.extension().and_then(|value| value.to_str()) == Some("mtx")
        || first.starts_with("%%MatrixMarket")
    {
        read_matrix_market(&text, num_cells, num_genes)
    } else {
        read_dense(&text, num_cells, num_genes)
    }
}

fn read_dense(text: &str, num_cells: usize, num_genes: usize) -> Result<Array2<f32>, String> {
    let mut values = Vec::with_capacity(num_cells * num_genes);
    let mut rows = 0;
    for line in text.lines().map(str::trim).filter(|line| !line.is_empty()) {
        let row: Result<Vec<f32>, _> = line
            .split(|character: char| {
                character == ',' || character == '\t' || character.is_whitespace()
            })
            .filter(|value| !value.is_empty())
            .map(str::parse::<f32>)
            .collect();
        let row = row.map_err(|error| format!("invalid dense matrix value: {error}"))?;
        if row.len() != num_genes {
            return Err(format!(
                "dense matrix row {} has {} columns; expected {}",
                rows + 1,
                row.len(),
                num_genes
            ));
        }
        values.extend(row);
        rows += 1;
    }
    if rows != num_cells {
        return Err(format!(
            "dense matrix has {rows} rows; expected {num_cells}"
        ));
    }
    Array2::from_shape_vec((num_cells, num_genes), values)
        .map_err(|error| format!("invalid dense matrix shape: {error}"))
}

fn read_matrix_market(
    text: &str,
    num_cells: usize,
    num_genes: usize,
) -> Result<Array2<f32>, String> {
    let mut lines = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('%'));
    let dimensions: Vec<usize> = lines
        .next()
        .ok_or_else(|| "MatrixMarket file has no dimensions".to_string())?
        .split_whitespace()
        .map(str::parse::<usize>)
        .collect::<Result<_, _>>()
        .map_err(|error| format!("invalid MatrixMarket dimensions: {error}"))?;
    if dimensions.len() != 3 {
        return Err("MatrixMarket dimensions must be: rows columns nonzeros".to_string());
    }
    let transposed = dimensions[0] == num_genes && dimensions[1] == num_cells;
    if !transposed && (dimensions[0] != num_cells || dimensions[1] != num_genes) {
        return Err(format!(
            "MatrixMarket shape {}x{} does not match cells x genes {}x{}",
            dimensions[0], dimensions[1], num_cells, num_genes
        ));
    }
    let mut matrix = Array2::<f32>::zeros((num_cells, num_genes));
    for line in lines {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() != 3 {
            return Err(format!("invalid MatrixMarket entry: {line}"));
        }
        let row = fields[0]
            .parse::<usize>()
            .map_err(|error| error.to_string())?
            - 1;
        let column = fields[1]
            .parse::<usize>()
            .map_err(|error| error.to_string())?
            - 1;
        let value = fields[2]
            .parse::<f32>()
            .map_err(|error| error.to_string())?;
        let (cell, gene) = if transposed {
            (column, row)
        } else {
            (row, column)
        };
        matrix[[cell, gene]] = value;
    }
    Ok(matrix)
}

pub fn load_and_standardize(
    matrix_path: impl AsRef<Path>,
    cell_ids_path: impl AsRef<Path>,
    gene_ids_path: impl AsRef<Path>,
) -> Result<ExpressionData, String> {
    let cell_ids = read_ids(cell_ids_path)?;
    let gene_ids = read_ids(gene_ids_path)?;
    let mut matrix = read_expression(matrix_path, cell_ids.len(), gene_ids.len())?;
    let raw_matrix = matrix.clone();
    let mut means = vec![0.0; gene_ids.len()];
    let mut standard_deviations = vec![0.0; gene_ids.len()];
    for gene in 0..gene_ids.len() {
        let mean = matrix.column(gene).sum() / cell_ids.len() as f32;
        let standard_deviation = (matrix
            .column(gene)
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f32>()
            / cell_ids.len() as f32)
            .sqrt()
            .max(1e-6);
        means[gene] = mean;
        standard_deviations[gene] = standard_deviation;
        for cell in 0..cell_ids.len() {
            matrix[[cell, gene]] = (matrix[[cell, gene]] - mean) / standard_deviation;
        }
    }
    Ok(ExpressionData {
        raw_matrix,
        matrix,
        cell_ids,
        gene_ids,
        means,
        standard_deviations,
    })
}

pub fn write_matrix(path: impl AsRef<Path>, matrix: &Array2<f32>) -> Result<(), String> {
    let file =
        fs::File::create(path).map_err(|error| format!("failed to create matrix: {error}"))?;
    let mut output = BufWriter::new(file);
    for row in matrix.rows() {
        for (column, value) in row.iter().enumerate() {
            if column > 0 {
                output
                    .write_all(b",")
                    .map_err(|error| format!("failed to write matrix: {error}"))?;
            }
            write!(output, "{value:.7}")
                .map_err(|error| format!("failed to write matrix: {error}"))?;
        }
        output
            .write_all(b"\n")
            .map_err(|error| format!("failed to write matrix: {error}"))?;
    }
    output
        .flush()
        .map_err(|error| format!("failed to flush matrix: {error}"))
}

pub fn write_ids(path: impl AsRef<Path>, ids: &[String]) -> Result<(), String> {
    fs::write(path, ids.join("\n") + "\n").map_err(|error| format!("failed to write IDs: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_dense_and_sparse_with_same_orientation() {
        let dense = read_dense("1,0\n0,2\n", 2, 2).unwrap();
        let sparse = read_matrix_market(
            "%%MatrixMarket matrix coordinate real general\n2 2 2\n1 1 1\n2 2 2\n",
            2,
            2,
        )
        .unwrap();
        assert_eq!(dense, sparse);
    }
}
